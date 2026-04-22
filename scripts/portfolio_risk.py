#!/usr/bin/env python3
"""
Portfolio Risk Manager — sector, theme, sizing, cash, and stop-risk checks.

Usage:
    python3 portfolio_risk.py                  # Full risk report
    python3 portfolio_risk.py --check XOM      # Check if adding XOM would violate rules
    python3 portfolio_risk.py --json           # JSON output
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from env_utils import configure_alpaca_env, warn_missing_credentials

ALPACA_ENV = configure_alpaca_env()
warn_missing_credentials(ALPACA_ENV["missing"], context="Portfolio risk / Alpaca")

import yfinance as yf
from alpaca_trade_api.rest import REST


DATA_DIR = Path.home() / "market-intel" / "data"
RISK_RULES_FILE = DATA_DIR / "portfolio_risk_rules.json"

DEFAULT_RISK_RULES = {
    "max_open_positions": 20,
    "max_stock_position_pct": 5.0,
    "max_etf_position_pct": 7.0,
    "max_sector_exposure_pct": 20.0,
    "max_theme_exposure_pct": 12.0,
    "cash_floor_pct": 5.0,
    "max_top5_exposure_pct": 35.0,
    "max_risk_per_trade_pct": 0.5,
    "max_total_open_risk_pct": 5.0,
    "min_risk_reward": 1.0,
}


# ── Sector Mapping ──────────────────────────────────────────

SECTOR_MAP = {
    # Energy
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy", "XLE": "Energy",
    "USO": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "VLO": "Energy", "MPC": "Energy", "EQT": "Energy", "RRC": "Energy", "UNG": "Energy",
    # Defense
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "GD": "Defense", "BA": "Defense",
    # Technology / AI infra
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "META": "Technology", "AMZN": "Technology", "TSLA": "Technology", "QQQ": "Technology",
    "XLK": "Technology", "VRT": "Technology", "EQIX": "Technology", "DLR": "Technology",
    # Financials
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials", "XLF": "Financials",
    "MS": "Financials", "WFC": "Financials", "KBE": "Financials",
    # Healthcare
    "XLV": "Healthcare", "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "LLY": "Healthcare",
    # Consumer
    "XLP": "Consumer Staples", "XLY": "Consumer Discretionary", "WMT": "Consumer Staples",
    # Industrials / transport
    "XLI": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    "DAL": "Industrials", "UAL": "Industrials",
    # Homebuilders / real estate
    "ITB": "Real Estate", "XHB": "Real Estate", "DHI": "Real Estate", "LEN": "Real Estate",
    # Utilities
    "XLU": "Utilities",
    # Broad Market
    "SPY": "Broad Market", "DIA": "Broad Market", "IWM": "Broad Market",
    # Bonds
    "TLT": "Bonds", "SHY": "Bonds",
    # International
    "EIDO": "International", "EWJ": "International",
    # Materials / ag
    "ADM": "Materials", "BG": "Materials", "WEAT": "Materials",
}

THEME_MAP = {
    # Energy complex
    "XOM": "Energy Complex", "CVX": "Energy Complex", "OXY": "Energy Complex", "XLE": "Energy Complex",
    "USO": "Energy Complex", "COP": "Energy Complex", "SLB": "Energy Complex", "EOG": "Energy Complex",
    "VLO": "Energy Complex", "MPC": "Energy Complex", "EQT": "Energy Complex", "RRC": "Energy Complex", "UNG": "Energy Complex",
    # Defense / conflict
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "GD": "Defense", "BA": "Defense",
    # AI / data center / semis
    "NVDA": "AI Infra", "QQQ": "AI Infra", "VRT": "AI Infra", "EQIX": "AI Infra", "DLR": "AI Infra", "XLK": "AI Infra",
    # Housing / homebuilders
    "ITB": "Homebuilders", "XHB": "Homebuilders", "DHI": "Homebuilders", "LEN": "Homebuilders",
    # Airlines / travel
    "DAL": "Airlines", "UAL": "Airlines",
    # Rates / bonds
    "TLT": "Rates", "SHY": "Rates",
    # Broad index risk
    "SPY": "Index Beta", "IWM": "Index Beta", "DIA": "Index Beta",
}


# ── Rule Loading ────────────────────────────────────────────

def load_risk_rules() -> dict:
    """Load risk rules from disk, backfilling missing keys."""
    rules = DEFAULT_RISK_RULES.copy()
    if RISK_RULES_FILE.exists():
        try:
            loaded = json.loads(RISK_RULES_FILE.read_text())
            if isinstance(loaded, dict):
                rules.update(loaded)
        except Exception:
            pass
    else:
        save_risk_rules(rules)
    return rules


def save_risk_rules(rules: dict):
    """Persist risk rules to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RISK_RULES_FILE.write_text(json.dumps(rules, indent=2) + "\n")


RISK_RULES = load_risk_rules()


# ── Metadata Helpers ────────────────────────────────────────

def get_security_info(symbol: str) -> dict:
    """Fetch lightweight security metadata with sensible fallbacks."""
    quote_type = None
    sector = None
    short_name = None

    try:
        info = yf.Ticker(symbol).info
        quote_type = info.get("quoteType") or info.get("quote_type")
        sector = info.get("sector")
        short_name = info.get("shortName") or info.get("longName")
    except Exception:
        pass

    return {
        "symbol": symbol,
        "quote_type": quote_type,
        "sector": sector,
        "short_name": short_name,
    }


def is_etf(symbol: str) -> bool:
    if symbol in {"SPY", "QQQ", "IWM", "DIA", "TLT", "XLE", "XLI", "XLP", "XLY", "XLK", "XLV", "XLF", "XLU", "XHB", "ITB", "USO", "UNG", "EIDO", "EWJ", "WEAT"}:
        return True
    meta = get_security_info(symbol)
    qt = (meta.get("quote_type") or "").lower()
    return qt in {"etf", "fund", "mutualfund"}


def get_sector(symbol: str) -> str:
    """Get sector for a symbol."""
    if symbol in SECTOR_MAP:
        return SECTOR_MAP[symbol]
    meta = get_security_info(symbol)
    return meta.get("sector") or "Unknown"


def get_theme(symbol: str) -> str:
    """Get correlated theme bucket for a symbol."""
    if symbol in THEME_MAP:
        return THEME_MAP[symbol]
    return get_sector(symbol)


def get_position_cap_pct(symbol: str) -> float:
    """Return the max allowed capital allocation for this symbol."""
    return RISK_RULES["max_etf_position_pct"] if is_etf(symbol) else RISK_RULES["max_stock_position_pct"]


# ── Portfolio State ─────────────────────────────────────────

def get_open_orders() -> list[dict]:
    """Return open Alpaca orders in normalized form."""
    api = REST()
    try:
        orders = api.list_orders(status="open")
    except Exception:
        return []

    result = []
    for o in orders:
        result.append({
            "id": str(o.id),
            "symbol": o.symbol,
            "side": o.side,
            "type": o.type,
            "qty": float(o.qty) if o.qty else 0.0,
            "status": o.status,
            "limit_price": float(o.limit_price) if getattr(o, "limit_price", None) else None,
            "stop_price": float(o.stop_price) if getattr(o, "stop_price", None) else None,
        })
    return result


def get_portfolio_data() -> dict:
    """Get current portfolio with sector/theme breakdown and stop-risk estimates."""
    api = REST()
    account = api.get_account()
    positions = api.list_positions()
    open_orders = get_open_orders()

    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    equity = float(account.equity)

    stop_orders_by_symbol = {}
    for order in open_orders:
        if order.get("type") == "stop" and order.get("stop_price"):
            stop_orders_by_symbol[order["symbol"]] = order

    pos_data = []
    sector_exposure = defaultdict(float)
    theme_exposure = defaultdict(float)
    total_exposure = 0.0
    total_open_risk = 0.0

    for p in positions:
        symbol = p.symbol
        market_value = abs(float(p.market_value))
        pct_of_portfolio = (market_value / portfolio_value) * 100 if portfolio_value else 0.0
        current_price = float(p.current_price)
        side = p.side
        sector = get_sector(symbol)
        theme = get_theme(symbol)
        cap_pct = get_position_cap_pct(symbol)

        sector_exposure[sector] += pct_of_portfolio
        theme_exposure[theme] += pct_of_portfolio
        total_exposure += pct_of_portfolio

        stop_price = None
        open_risk_pct = None
        stop_order = stop_orders_by_symbol.get(symbol)
        if stop_order and stop_order.get("stop_price"):
            stop_price = float(stop_order["stop_price"])
            try:
                qty = abs(float(p.qty))
                risk_dollars = abs(current_price - stop_price) * qty
                open_risk_pct = (risk_dollars / portfolio_value) * 100 if portfolio_value else 0.0
                total_open_risk += open_risk_pct
            except Exception:
                open_risk_pct = None

        pos_data.append({
            "symbol": symbol,
            "side": side,
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "current_price": current_price,
            "market_value": market_value,
            "pct_of_portfolio": round(pct_of_portfolio, 2),
            "sector": sector,
            "theme": theme,
            "asset_type": "ETF" if is_etf(symbol) else "Stock",
            "max_position_pct": cap_pct,
            "pl_pct": round(float(p.unrealized_plpc) * 100, 2),
            "pl_dollars": round(float(p.unrealized_pl), 2),
            "stop_price": stop_price,
            "open_risk_pct": round(open_risk_pct, 3) if open_risk_pct is not None else None,
        })

    top5_exposure = sum(sorted((p["pct_of_portfolio"] for p in pos_data), reverse=True)[:5])

    return {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "cash_pct": round((cash / portfolio_value) * 100, 2) if portfolio_value else 0.0,
        "equity": equity,
        "total_exposure_pct": round(total_exposure, 2),
        "top5_exposure_pct": round(top5_exposure, 2),
        "total_open_risk_pct": round(total_open_risk, 3),
        "positions": pos_data,
        "open_orders": open_orders,
        "sector_exposure": dict(sector_exposure),
        "theme_exposure": dict(theme_exposure),
        "num_positions": len(positions),
        "num_sectors": len(sector_exposure),
        "num_themes": len(theme_exposure),
    }


# ── Risk Checks ─────────────────────────────────────────────

def calculate_trade_risk_pct(entry: float, stop_loss: float, qty: float, portfolio_value: float) -> float:
    """Estimate max portfolio loss-at-stop for a proposed trade."""
    if not entry or not stop_loss or not qty or not portfolio_value:
        return 0.0
    risk_dollars = abs(entry - stop_loss) * abs(qty)
    return (risk_dollars / portfolio_value) * 100


def check_risk_violations(portfolio: dict) -> list[dict]:
    """Check current portfolio against all hard rules."""
    violations = []

    for sector, pct in portfolio["sector_exposure"].items():
        if pct > RISK_RULES["max_sector_exposure_pct"]:
            violations.append({
                "rule": "max_sector_exposure_pct",
                "severity": "high",
                "message": f"🚨 {sector} exposure {pct:.1f}% exceeds {RISK_RULES['max_sector_exposure_pct']}% limit",
                "sector": sector,
                "current": pct,
                "limit": RISK_RULES["max_sector_exposure_pct"],
            })

    for theme, pct in portfolio["theme_exposure"].items():
        if pct > RISK_RULES["max_theme_exposure_pct"]:
            violations.append({
                "rule": "max_theme_exposure_pct",
                "severity": "high",
                "message": f"🚨 {theme} theme exposure {pct:.1f}% exceeds {RISK_RULES['max_theme_exposure_pct']}% limit",
                "theme": theme,
                "current": pct,
                "limit": RISK_RULES["max_theme_exposure_pct"],
            })

    for pos in portfolio["positions"]:
        if pos["pct_of_portfolio"] > pos["max_position_pct"]:
            violations.append({
                "rule": "max_position_pct",
                "severity": "high",
                "message": f"🚨 {pos['symbol']} is {pos['pct_of_portfolio']:.1f}% of portfolio (limit: {pos['max_position_pct']}% for {pos['asset_type']})",
                "symbol": pos["symbol"],
                "current": pos["pct_of_portfolio"],
                "limit": pos["max_position_pct"],
            })

    if portfolio["cash_pct"] < RISK_RULES["cash_floor_pct"]:
        violations.append({
            "rule": "cash_floor_pct",
            "severity": "medium",
            "message": f"⚠️  Cash {portfolio['cash_pct']:.1f}% is below {RISK_RULES['cash_floor_pct']}% floor",
            "current": portfolio["cash_pct"],
            "limit": RISK_RULES["cash_floor_pct"],
        })

    if portfolio["num_positions"] >= RISK_RULES["max_open_positions"]:
        violations.append({
            "rule": "max_open_positions",
            "severity": "medium",
            "message": f"⚠️  At position limit ({portfolio['num_positions']}/{RISK_RULES['max_open_positions']})",
            "current": portfolio["num_positions"],
            "limit": RISK_RULES["max_open_positions"],
        })

    if portfolio["top5_exposure_pct"] > RISK_RULES["max_top5_exposure_pct"]:
        violations.append({
            "rule": "max_top5_exposure_pct",
            "severity": "high",
            "message": f"🚨 Top-5 exposure {portfolio['top5_exposure_pct']:.1f}% exceeds {RISK_RULES['max_top5_exposure_pct']}% limit",
            "current": portfolio["top5_exposure_pct"],
            "limit": RISK_RULES["max_top5_exposure_pct"],
        })

    if portfolio["total_open_risk_pct"] > RISK_RULES["max_total_open_risk_pct"]:
        violations.append({
            "rule": "max_total_open_risk_pct",
            "severity": "high",
            "message": f"🚨 Open stop-risk {portfolio['total_open_risk_pct']:.2f}% exceeds {RISK_RULES['max_total_open_risk_pct']}% limit",
            "current": portfolio["total_open_risk_pct"],
            "limit": RISK_RULES["max_total_open_risk_pct"],
        })

    return violations


def check_new_trade(symbol: str, side: str = "long", size_pct: float = None,
                    trade_risk_pct: float = 0.0, pending_orders: list[dict] | None = None) -> dict:
    """Check if adding a new trade would violate hard rules."""
    portfolio = get_portfolio_data()
    sector = get_sector(symbol)
    theme = get_theme(symbol)
    cap_pct = get_position_cap_pct(symbol)
    proposed_size_pct = min(size_pct if size_pct is not None else cap_pct, cap_pct)
    pending_orders = pending_orders if pending_orders is not None else portfolio.get("open_orders", [])

    would_violate = []

    current_sector_pct = portfolio["sector_exposure"].get(sector, 0.0)
    new_sector_pct = current_sector_pct + proposed_size_pct
    if new_sector_pct > RISK_RULES["max_sector_exposure_pct"]:
        would_violate.append({
            "rule": "max_sector_exposure_pct",
            "message": f"{sector} would be {new_sector_pct:.1f}% (limit: {RISK_RULES['max_sector_exposure_pct']}%)",
        })

    current_theme_pct = portfolio["theme_exposure"].get(theme, 0.0)
    new_theme_pct = current_theme_pct + proposed_size_pct
    if new_theme_pct > RISK_RULES["max_theme_exposure_pct"]:
        would_violate.append({
            "rule": "max_theme_exposure_pct",
            "message": f"{theme} would be {new_theme_pct:.1f}% (limit: {RISK_RULES['max_theme_exposure_pct']}%)",
        })

    existing = next((p for p in portfolio["positions"] if p["symbol"] == symbol), None)

    if portfolio["num_positions"] >= RISK_RULES["max_open_positions"] and not existing:
        would_violate.append({
            "rule": "max_open_positions",
            "message": f"Already at {portfolio['num_positions']} positions (limit: {RISK_RULES['max_open_positions']})",
        })

    cash_after_trade = portfolio["cash_pct"] - proposed_size_pct
    if cash_after_trade < RISK_RULES["cash_floor_pct"]:
        would_violate.append({
            "rule": "cash_floor_pct",
            "message": f"Cash would fall to {cash_after_trade:.1f}% (floor: {RISK_RULES['cash_floor_pct']}%)",
        })
    if existing:
        total_in_symbol = existing["pct_of_portfolio"] + proposed_size_pct
        if total_in_symbol > cap_pct:
            would_violate.append({
                "rule": "max_position_pct",
                "message": f"{symbol} would be {total_in_symbol:.1f}% of portfolio (limit: {cap_pct}% for {existing['asset_type']})",
            })

    top5_candidate = sum(sorted([p["pct_of_portfolio"] for p in portfolio["positions"]] + [proposed_size_pct], reverse=True)[:5])
    if top5_candidate > RISK_RULES["max_top5_exposure_pct"]:
        would_violate.append({
            "rule": "max_top5_exposure_pct",
            "message": f"Top-5 exposure would be {top5_candidate:.1f}% (limit: {RISK_RULES['max_top5_exposure_pct']}%)",
        })

    total_open_risk_after = portfolio["total_open_risk_pct"] + trade_risk_pct
    if trade_risk_pct > RISK_RULES["max_risk_per_trade_pct"]:
        would_violate.append({
            "rule": "max_risk_per_trade_pct",
            "message": f"Trade risk at stop would be {trade_risk_pct:.2f}% (limit: {RISK_RULES['max_risk_per_trade_pct']}%)",
        })
    if total_open_risk_after > RISK_RULES["max_total_open_risk_pct"]:
        would_violate.append({
            "rule": "max_total_open_risk_pct",
            "message": f"Total open risk would be {total_open_risk_after:.2f}% (limit: {RISK_RULES['max_total_open_risk_pct']}%)",
        })

    conflicting_orders = [
        o for o in pending_orders
        if o.get("symbol") == symbol and o.get("type") == "market" and o.get("status") in {"new", "accepted", "pending_new", "partially_filled"}
    ]
    if conflicting_orders:
        would_violate.append({
            "rule": "duplicate_pending_order",
            "message": f"{symbol} already has {len(conflicting_orders)} pending entry order(s)",
        })

    allowed = len(would_violate) == 0
    return {
        "symbol": symbol,
        "sector": sector,
        "theme": theme,
        "asset_type": "ETF" if is_etf(symbol) else "Stock",
        "side": side,
        "size_pct": round(proposed_size_pct, 3),
        "trade_risk_pct": round(trade_risk_pct, 3),
        "allowed": allowed,
        "violations": would_violate,
        "current_sector_pct": round(current_sector_pct, 2),
        "new_sector_pct": round(new_sector_pct, 2),
        "current_theme_pct": round(current_theme_pct, 2),
        "new_theme_pct": round(new_theme_pct, 2),
        "cash_after_trade_pct": round(cash_after_trade, 2),
        "total_open_risk_after_pct": round(total_open_risk_after, 3),
        "position_cap_pct": cap_pct,
        "portfolio": portfolio,
    }


# ── Display ─────────────────────────────────────────────────

def print_risk_report(portfolio: dict, violations: list[dict]):
    """Pretty-print risk report."""
    print(f"\n{'=' * 70}")
    print(f"  PORTFOLIO RISK REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")

    print(f"\n  💰 Portfolio: ${portfolio['portfolio_value']:,.2f}")
    print(f"  💵 Cash: ${portfolio['cash']:,.2f} ({portfolio['cash_pct']:.1f}%)")
    print(f"  📊 Deployed: {portfolio['total_exposure_pct']:.1f}%")
    print(f"  📈 Positions: {portfolio['num_positions']} / {RISK_RULES['max_open_positions']}")
    print(f"  🏭 Sectors: {portfolio['num_sectors']} | Themes: {portfolio['num_themes']}")
    print(f"  🧮 Top-5 Exposure: {portfolio['top5_exposure_pct']:.1f}% / {RISK_RULES['max_top5_exposure_pct']}%")
    print(f"  🛑 Open Stop Risk: {portfolio['total_open_risk_pct']:.2f}% / {RISK_RULES['max_total_open_risk_pct']}%")

    if portfolio["sector_exposure"]:
        print(f"\n  {'─' * 60}")
