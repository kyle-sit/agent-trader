#!/usr/bin/env python3
"""
Signal Executor — Phase 4 of Market Intelligence Pipeline.

Takes trade signals from the Correlation Engine and:
  1. Sends alerts to Discord
  2. Executes paper trades on Alpaca
  3. Manages open positions (trailing stops, take profit)

Usage:
    python3 signal_executor.py                       # Full pipeline → alerts + paper trades
    python3 signal_executor.py --alerts-only         # Full GPT/LLM analysis + alerts, no trades
    python3 signal_executor.py --dry-run             # Heuristic smoke test: no LLM, alerts only, no trades
    python3 signal_executor.py --watchlist iran       # Focus on Iran watchlist
    python3 signal_executor.py --positions            # Show current paper positions
    python3 signal_executor.py --close AAPL           # Close a position
    python3 signal_executor.py --status               # Full system status
"""

import json
import os
import sys
import time
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from correlation_engine import run_pipeline
from dotenv import load_dotenv

load_dotenv(Path.home() / ".hermes" / ".env")

# ── Config ──────────────────────────────────────────────────

# Alpaca
os.environ['APCA_API_KEY_ID'] = os.getenv('APCA_API_KEY_ID', '')
os.environ['APCA_API_SECRET_KEY'] = os.getenv('APCA_API_SECRET_KEY', '')
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

# Discord alerts are sent directly via Hermes agent — no webhook needed

# Trade config
from portfolio_risk import (
    load_risk_rules,
    check_new_trade,
    calculate_trade_risk_pct,
    get_position_cap_pct,
    get_theme,
    get_sector,
    get_portfolio_data,
)

RISK_RULES = load_risk_rules()
MAX_OPEN_POSITIONS = int(RISK_RULES.get("max_open_positions", 20))
MIN_CONVICTION = "medium"        # Minimum conviction to auto-trade
MIN_RISK_REWARD = float(RISK_RULES.get("min_risk_reward", 1.0))
TRADE_LOG_FILE = Path.home() / "market-intel" / "data" / "trade_log.jsonl"
RUNTIME_LOG_FILE = Path.home() / "market-intel" / "data" / "runtime_log.jsonl"


# ── Discord Alerts ──────────────────────────────────────────

def send_discord_alert(content: str, webhook_url: str = None):
    """Send a message to Discord via webhook."""
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        print("  ⚠️  No Discord webhook configured — skipping alert")
        return False

    try:
        # Discord has 2000 char limit per message
        chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
        for chunk in chunks:
            data = json.dumps({"content": chunk}).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  ⚠️  Discord alert failed: {e}")
        return False


def format_discord_message(results: dict) -> str:
    """Format correlation results as a Discord message."""
    regime = results.get("market_regime", "unknown")
    regime_emoji = {"risk-on": "🟢", "risk-off": "🔴", "mixed": "🟡", "neutral": "⚪"}.get(regime, "❓")

    lines = []
    lines.append(f"# 📊 Market Intelligence Report")
    lines.append(f"**{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
    lines.append(f"")
    lines.append(f"{regime_emoji} **Regime: {regime.upper()}** — {results.get('regime_reasoning', '')[:200]}")
    lines.append(f"Headlines: {results['headlines_analyzed']} | Events: {results['events_detected']} | Tickers: {results['tickers_analyzed']}")
    lines.append("")

    signals = results.get("trade_signals", [])
    enter = [s for s in signals if s["action"] == "enter"]
    watch = [s for s in signals if s["action"] == "watch"]

    if enter:
        lines.append(f"## 🎯 ENTER — {len(enter)} Trades")
        for s in enter:
            dir_emoji = "🟢 LONG" if s["direction"] == "long" else "🔴 SHORT"
            lines.append(f"**{dir_emoji} {s['ticker']}** | Conviction: **{s['conviction'].upper()}** | Score: {s['score']:+.1f}")
            lines.append(f"  📰 `{s['news_catalyst'][:80]}`")
            lines.append(f"  📊 TA: {s['ta_signal']} ({s['ta_strength']})")
            if s.get("entry"):
                lines.append(f"  💰 Entry: ${s['entry']} | Stop: ${s['stop_loss']} | Target: ${s['target']} | R:R: {s['risk_reward']}")
            lines.append("")

    if watch:
        lines.append(f"## 👀 WATCH — {len(watch)} Setups")
        for s in watch[:5]:
            dir_emoji = "🟢" if s["direction"] == "long" else "🔴" if s["direction"] == "short" else "⚪"
            lines.append(f"{dir_emoji} **{s['ticker']}** ({s['conviction']}) | {s['news_catalyst'][:60]}")
        lines.append("")

    return "\n".join(lines)


# ── Alpaca Paper Trading ───────────────────────────────────

def get_alpaca_api():
    """Get Alpaca REST API client."""
    from alpaca_trade_api.rest import REST
    return REST()


def get_account_info() -> dict:
    """Get paper account status."""
    api = get_alpaca_api()
    account = api.get_account()
    return {
        "status": account.status,
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "equity": float(account.equity),
        "long_market_value": float(account.long_market_value),
        "short_market_value": float(account.short_market_value),
    }


def get_positions() -> list[dict]:
    """Get current open positions."""
    api = get_alpaca_api()
    positions = api.list_positions()
    result = []
    for p in positions:
        result.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "side": p.side,
            "avg_entry": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) * 100,
            "change_today": float(p.change_today) * 100,
        })
    return result


def get_open_orders() -> list[dict]:
    """Get open/pending orders."""
    api = get_alpaca_api()
    orders = api.list_orders(status="open")
    result = []
    for o in orders:
        result.append({
            "id": o.id,
            "symbol": o.symbol,
            "side": o.side,
            "type": o.type,
            "qty": float(o.qty) if o.qty else 0,
            "limit_price": float(o.limit_price) if o.limit_price else None,
            "stop_price": float(o.stop_price) if o.stop_price else None,
            "status": o.status,
        })
    return result


def calculate_position_size(portfolio_value: float, price: float, symbol: str = None, stop_loss: float = None) -> int:
    """Calculate shares using asset cap plus stop-risk cap."""
    if not portfolio_value or not price or price <= 0:
        return 0

    position_cap_pct = get_position_cap_pct(symbol) if symbol else RISK_RULES.get("max_stock_position_pct", 5.0)
    max_dollars = portfolio_value * (position_cap_pct / 100)
    cap_shares = int(max_dollars / price)

    risk_shares = cap_shares
    if stop_loss and stop_loss > 0:
        per_share_risk = abs(price - stop_loss)
        if per_share_risk > 0:
            max_risk_dollars = portfolio_value * (RISK_RULES.get("max_risk_per_trade_pct", 0.5) / 100)
            risk_shares = int(max_risk_dollars / per_share_risk)

    shares = min(cap_shares, risk_shares)
    return shares if shares > 0 else 0


def execute_trade(signal: dict, portfolio_value: float, qty_override: int = None) -> dict:
    """Execute a paper trade on Alpaca based on a trade signal."""
    api = get_alpaca_api()
    ticker = signal["ticker"]
    direction = signal["direction"]
    entry = signal.get("entry")
    stop_loss = signal.get("stop_loss")
    target = signal.get("target")

    if not entry or entry <= 0:
        return {"success": False, "error": "No valid entry price"}

    # Position sizing
    qty = qty_override if qty_override is not None else calculate_position_size(portfolio_value, entry, symbol=ticker, stop_loss=stop_loss)
    qty = int(qty or 0)
    if qty <= 0:
        return {"success": False, "error": "Position size reduced to zero by risk limits", "symbol": ticker}

    side = "buy" if direction == "long" else "sell"

    try:
        # Submit market order
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )

        result = {
            "success": True,
            "order_id": order.id,
            "symbol": ticker,
            "side": side,
            "qty": qty,
            "type": "market",
            "status": order.status,
            "submitted_at": str(order.submitted_at),
        }

        # Submit stop loss as separate order
        if stop_loss and stop_loss > 0:
            try:
                stop_side = "sell" if direction == "long" else "buy"
                stop_order = api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=stop_side,
                    type="stop",
                    time_in_force="gtc",
                    stop_price=str(stop_loss),
                )
                result["stop_order_id"] = stop_order.id
                result["stop_price"] = stop_loss
            except Exception as e:
                result["stop_error"] = str(e)

        # Submit take profit as separate order
        if target and target > 0:
            try:
                tp_side = "sell" if direction == "long" else "buy"
                tp_order = api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=tp_side,
                    type="limit",
                    time_in_force="gtc",
                    limit_price=str(target),
                )
                result["target_order_id"] = tp_order.id
                result["target_price"] = target
            except Exception as e:
                result["target_error"] = str(e)

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "symbol": ticker}


def close_position(symbol: str) -> dict:
    """Close an open position."""
    api = get_alpaca_api()
    try:
        api.close_position(symbol)
        return {"success": True, "symbol": symbol, "action": "closed"}
    except Exception as e:
        return {"success": False, "symbol": symbol, "error": str(e)}


def reduce_position(symbol: str, qty: int, position_side: str) -> dict:
    """Reduce an existing position by submitting an offsetting market order."""
    api = get_alpaca_api()
    if qty <= 0:
        return {"success": False, "symbol": symbol, "error": "Reduction qty must be > 0"}
    try:
        side = "sell" if position_side == "long" else "buy"
        order = api.submit_order(
            symbol=symbol,
            qty=int(qty),
            side=side,
            type="market",
            time_in_force="day",
        )
        return {
            "success": True,
            "symbol": symbol,
            "action": "reduced",
            "qty": int(qty),
            "side": side,
            "order_id": str(order.id),
            "status": order.status,
        }
    except Exception as e:
        return {"success": False, "symbol": symbol, "error": str(e)}


def cancel_symbol_orders(symbol: str) -> dict:
    """Cancel open orders for a symbol."""
    api = get_alpaca_api()
    cancelled = []
    errors = []
    try:
        for order in api.list_orders(status="open"):
            if order.symbol != symbol:
                continue
            try:
                api.cancel_order(order.id)
                cancelled.append(str(order.id))
            except Exception as e:
                errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    return {
        "success": len(errors) == 0,
        "symbol": symbol,
        "cancelled_order_ids": cancelled,
        "errors": errors,
    }


def compute_adjusted_qty(base_qty: int, size_fraction: float) -> int:
    """Convert LLM size fraction into a concrete qty."""
    frac = float(size_fraction or 0)
    if frac <= 0:
        return 0
    return max(int(round(base_qty * frac)), 1)


def log_trade(trade_data: dict):
    """Append trade to log file."""
    TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    trade_data["logged_at"] = datetime.now().isoformat()
    with open(TRADE_LOG_FILE, "a") as f:
        f.write(json.dumps(trade_data) + "\n")


def log_runtime(runtime_data: dict):
    """Append runtime metrics for a pipeline run."""
    RUNTIME_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    runtime_data["logged_at"] = datetime.now().isoformat()
    with open(RUNTIME_LOG_FILE, "a") as f:
        f.write(json.dumps(runtime_data) + "\n")


def build_portfolio_context(signal: dict, positions: list[dict], open_orders: list[dict], portfolio_snapshot: dict,
                            proposed_qty: int, proposed_size_pct: float, trade_risk_pct: float) -> str:
    """Build a compact but rich execution context string for LLM validation."""
    ticker = signal.get("ticker")
    matching_position = next((p for p in positions if p["symbol"] == ticker), None)
    matching_orders = [o for o in open_orders if o.get("symbol") == ticker]

    context = {
        "portfolio_summary": {
            "portfolio_value": portfolio_snapshot.get("portfolio_value"),
            "cash_pct": portfolio_snapshot.get("cash_pct"),
            "num_positions": portfolio_snapshot.get("num_positions"),
            "top5_exposure_pct": portfolio_snapshot.get("top5_exposure_pct"),
            "total_open_risk_pct": portfolio_snapshot.get("total_open_risk_pct"),
        },
        "hard_rules": {
            "max_open_positions": RISK_RULES.get("max_open_positions"),
            "max_stock_position_pct": RISK_RULES.get("max_stock_position_pct"),
            "max_etf_position_pct": RISK_RULES.get("max_etf_position_pct"),
            "max_sector_exposure_pct": RISK_RULES.get("max_sector_exposure_pct"),
            "max_theme_exposure_pct": RISK_RULES.get("max_theme_exposure_pct"),
            "cash_floor_pct": RISK_RULES.get("cash_floor_pct"),
            "max_top5_exposure_pct": RISK_RULES.get("max_top5_exposure_pct"),
            "max_risk_per_trade_pct": RISK_RULES.get("max_risk_per_trade_pct"),
            "max_total_open_risk_pct": RISK_RULES.get("max_total_open_risk_pct"),
        },
        "signal_symbol": ticker,
        "signal_direction": signal.get("direction"),
        "signal_sector": get_sector(ticker),
        "signal_theme": get_theme(ticker),
        "proposed_qty": proposed_qty,
        "proposed_size_pct": round(proposed_size_pct, 3),
        "proposed_trade_risk_pct": round(trade_risk_pct, 3),
        "existing_position": matching_position,
        "matching_open_orders": matching_orders,
        "sector_exposure": portfolio_snapshot.get("sector_exposure", {}),
        "theme_exposure": portfolio_snapshot.get("theme_exposure", {}),
    }
    return json.dumps(context, separators=(",", ":"))


# ── Display ─────────────────────────────────────────────────

def print_positions():
    """Display current positions."""
    account = get_account_info()
    positions = get_positions()
    orders = get_open_orders()

    print(f"\n{'=' * 70}")
    print(f"  PAPER ACCOUNT STATUS")
    print(f"{'=' * 70}")
    print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
    print(f"  Cash:            ${account['cash']:,.2f}")
    print(f"  Buying Power:    ${account['buying_power']:,.2f}")
    print(f"  Long Exposure:   ${account['long_market_value']:,.2f}")
    print(f"  Short Exposure:  ${account['short_market_value']:,.2f}")

    if positions:
        print(f"\n{'─' * 70}")
        print(f"  OPEN POSITIONS ({len(positions)})")
        print(f"{'─' * 70}")
        print(f"  {'Symbol':8s} {'Qty':>6s} {'Entry':>10s} {'Current':>10s} {'P&L':>10s} {'P&L%':>8s}")
        print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for p in positions:
            pl_icon = "🟢" if p["unrealized_pl"] >= 0 else "🔴"
            print(f"  {pl_icon}{p['symbol']:7s} {p['qty']:6.0f} ${p['avg_entry']:9.2f} ${p['current_price']:9.2f} ${p['unrealized_pl']:9.2f} {p['unrealized_plpc']:+7.2f}%")
    else:
        print(f"\n  No open positions.")

    if orders:
        print(f"\n{'─' * 70}")
        print(f"  OPEN ORDERS ({len(orders)})")
        print(f"{'─' * 70}")
        for o in orders:
            price_str = f"Stop: ${o['stop_price']}" if o['stop_price'] else f"Limit: ${o['limit_price']}" if o['limit_price'] else "Market"
            print(f"  {o['symbol']:8s} {o['side']:5s} {o['qty']:5.0f} {o['type']:8s} {price_str}")


def print_status():
    """Full system status."""
    print(f"\n{'=' * 70}")
    print(f"  MARKET INTELLIGENCE SYSTEM STATUS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")

    # Data sources
    print(f"\n  📡 Data Sources:")
    from news_analyzer import fetch_rss_headlines, fetch_finnhub_news, fetch_alpaca_news, fetch_twitter_trending
    rss = fetch_rss_headlines(1)
    print(f"     RSS (blogwatcher):  {'✅' if rss else '❌'}")
    fh = fetch_finnhub_news(1)
    print(f"     Finnhub:            {'✅' if fh else '⚠️  No data'}")
    alp = fetch_alpaca_news(1)
    print(f"     Alpaca/Benzinga:    {'✅' if alp else '⚠️  No data'}")
    tw = fetch_twitter_trending(3)
    print(f"     Twitter/X:          {'✅' if tw else '❌'}")

    # Account
    try:
        account = get_account_info()
        positions = get_positions()
        print(f"\n  💰 Paper Account:")
        print(f"     Status:     {account['status']}")
        print(f"     Value:      ${account['portfolio_value']:,.2f}")
        print(f"     Cash:       ${account['cash']:,.2f}")
        print(f"     Positions:  {len(positions)}")
    except Exception as e:
        print(f"\n  💰 Paper Account: ❌ {e}")

    # LLM
    print(f"\n  🧠 LLM Backend:")
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print(f"     Ollama: ✅ Running")
    except Exception:
        print(f"     Ollama: ❌ Not running")
    print(f"     Anthropic: {'✅ Key set' if os.getenv('ANTHROPIC_TOKEN') else '❌ No key'}")

    # Discord
    print(f"\n  📢 Discord Alerts:")
    print(f"     Mode: Hermes agent (direct to channel)")

    # Trade log
