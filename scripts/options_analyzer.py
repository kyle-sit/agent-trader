#!/usr/bin/env python3
"""
Options Analysis Layer — Options chain analysis for hedging and leveraged plays.

Uses yfinance for options chain data. Provides:
  - Options chain overview (calls/puts with greeks)
  - Unusual options activity detection (volume vs open interest)
  - Put/call ratio analysis (market sentiment)
  - Suggested hedges for open positions
  - Optimal strike selection for directional plays

Usage:
    python3 options_analyzer.py AAPL                    # Full options overview
    python3 options_analyzer.py SPY --unusual            # Unusual activity scan
    python3 options_analyzer.py XOM --hedge              # Suggest hedges
    python3 options_analyzer.py SPY --pcr                # Put/call ratio
    python3 options_analyzer.py --watchlist default --pcr # PCR for all watchlist
"""

import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from ta_engine import WATCHLISTS


# ── Core Functions ──────────────────────────────────────────

def get_options_chain(symbol: str, expiry_idx: int = 0) -> dict:
    """Get options chain for a symbol. expiry_idx=0 is nearest expiry."""
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    if not expirations:
        return {"error": f"No options available for {symbol}"}

    expiry = expirations[min(expiry_idx, len(expirations) - 1)]
    chain = ticker.option_chain(expiry)

    # Get current price
    info = ticker.fast_info
    current_price = info.last_price

    calls = chain.calls
    puts = chain.puts

    # Add moneyness
    calls = calls.copy()
    puts = puts.copy()
    calls['moneyness'] = ((calls['strike'] - current_price) / current_price * 100).round(2)
    puts['moneyness'] = ((puts['strike'] - current_price) / current_price * 100).round(2)

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "expiry": expiry,
        "expirations_available": list(expirations),
        "calls": calls,
        "puts": puts,
        "total_call_volume": int(calls['volume'].sum()) if 'volume' in calls else 0,
        "total_put_volume": int(puts['volume'].sum()) if 'volume' in puts else 0,
        "total_call_oi": int(calls['openInterest'].sum()) if 'openInterest' in calls else 0,
        "total_put_oi": int(puts['openInterest'].sum()) if 'openInterest' in puts else 0,
    }


def calculate_pcr(chain: dict) -> dict:
    """Calculate put/call ratio — a sentiment indicator."""
    call_vol = chain.get("total_call_volume", 0)
    put_vol = chain.get("total_put_volume", 0)
    call_oi = chain.get("total_call_oi", 0)
    put_oi = chain.get("total_put_oi", 0)

    vol_pcr = put_vol / call_vol if call_vol > 0 else 0
    oi_pcr = put_oi / call_oi if call_oi > 0 else 0

    # Interpretation
    # PCR > 1.0 = more puts than calls = bearish sentiment (or hedging)
    # PCR < 0.7 = more calls than puts = bullish sentiment (or complacency)
    # PCR 0.7-1.0 = neutral
    if vol_pcr > 1.2:
        sentiment = "very_bearish"
        note = "High put buying — fear/hedging dominant"
    elif vol_pcr > 1.0:
        sentiment = "bearish"
        note = "More puts than calls — cautious market"
    elif vol_pcr > 0.7:
        sentiment = "neutral"
        note = "Balanced put/call activity"
    elif vol_pcr > 0.5:
        sentiment = "bullish"
        note = "More calls than puts — bullish sentiment"
    else:
        sentiment = "very_bullish"
        note = "Extreme call buying — could signal complacency"

    return {
        "symbol": chain["symbol"],
        "volume_pcr": round(vol_pcr, 3),
        "oi_pcr": round(oi_pcr, 3),
        "call_volume": call_vol,
        "put_volume": put_vol,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "sentiment": sentiment,
        "note": note,
    }


def find_unusual_activity(chain: dict, min_vol_oi_ratio: float = 2.0) -> list[dict]:
    """Find options with unusually high volume relative to open interest."""
    unusual = []

    for option_type, df in [("call", chain["calls"]), ("put", chain["puts"])]:
        if df.empty:
            continue

        for _, row in df.iterrows():
            vol = row.get('volume', 0)
            oi = row.get('openInterest', 0)

            if pd.isna(vol) or pd.isna(oi) or vol == 0:
                continue

            vol_oi_ratio = vol / oi if oi > 0 else vol
            if vol_oi_ratio >= min_vol_oi_ratio and vol >= 100:
                unusual.append({
                    "type": option_type,
                    "strike": row['strike'],
                    "moneyness": row.get('moneyness', 0),
                    "volume": int(vol),
                    "open_interest": int(oi),
                    "vol_oi_ratio": round(vol_oi_ratio, 2),
                    "last_price": row.get('lastPrice', 0),
                    "implied_vol": round(row.get('impliedVolatility', 0) * 100, 1),
                    "bid": row.get('bid', 0),
                    "ask": row.get('ask', 0),
                })

    # Sort by volume/OI ratio
    unusual.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
    return unusual


def suggest_hedge(symbol: str, side: str = "long", position_value: float = 5000) -> dict:
    """Suggest protective options for an existing position."""
    chain = get_options_chain(symbol)
    if chain.get("error"):
        return chain

    price = chain["current_price"]
    puts = chain["puts"]

    if side == "long":
        # For long positions, buy puts for protection
        # Look for puts 5-10% OTM
        otm_puts = puts[(puts['moneyness'] >= -10) & (puts['moneyness'] <= -3)]
        if otm_puts.empty:
            otm_puts = puts[puts['moneyness'] < 0].tail(5)

        suggestions = []
        shares = int(position_value / price)
        contracts_needed = max(1, shares // 100)

        for _, row in otm_puts.iterrows():
            ask = row.get('ask', 0)
            if pd.isna(ask) or ask == 0:
                continue

            cost = ask * 100 * contracts_needed
            protection_level = row['strike']
            max_loss_pct = ((protection_level - price) / price) * 100
            cost_pct = (cost / position_value) * 100

            suggestions.append({
                "type": "protective_put",
                "strike": row['strike'],
                "moneyness_pct": row.get('moneyness', 0),
                "ask": ask,
                "contracts": contracts_needed,
                "total_cost": round(cost, 2),
                "cost_pct_of_position": round(cost_pct, 2),
                "protection_level": protection_level,
                "max_loss_with_hedge": round(max_loss_pct + cost_pct, 2),
                "implied_vol": round(row.get('impliedVolatility', 0) * 100, 1),
                "expiry": chain["expiry"],
            })

        return {
            "symbol": symbol,
            "current_price": price,
            "position_side": side,
            "position_value": position_value,
            "shares": shares,
            "suggestions": suggestions[:5],
        }

    return {"error": "Short hedges not yet implemented"}


# ── Display ─────────────────────────────────────────────────

def print_overview(chain: dict):
    """Print options chain overview."""
    print(f"\n{'=' * 70}")
    print(f"  OPTIONS: {chain['symbol']} @ ${chain['current_price']}")
    print(f"  Expiry: {chain['expiry']} | Available: {len(chain['expirations_available'])} expirations")
    print(f"{'=' * 70}")

    pcr = calculate_pcr(chain)
    sent_icon = {"very_bullish": "🟢🟢", "bullish": "🟢", "neutral": "⚪", "bearish": "🔴", "very_bearish": "🔴🔴"}
    print(f"\n  {sent_icon.get(pcr['sentiment'], '❓')} Put/Call Ratio: {pcr['volume_pcr']:.3f} — {pcr['note']}")
    print(f"  Call Vol: {pcr['call_volume']:,} | Put Vol: {pcr['put_volume']:,}")

    # Near-the-money options
    price = chain['current_price']
    calls = chain['calls']
    puts = chain['puts']

    ntm_calls = calls[abs(calls['moneyness']) < 5].head(5)
    ntm_puts = puts[abs(puts['moneyness']) < 5].head(5)

    if not ntm_calls.empty:
        print(f"\n  CALLS (near the money):")
        print(f"  {'Strike':>8s} {'Last':>8s} {'Bid':>8s} {'Ask':>8s} {'Vol':>8s} {'OI':>8s} {'IV':>6s}")
        for _, r in ntm_calls.iterrows():
            print(f"  ${r['strike']:7.2f} ${r.get('lastPrice',0):7.2f} ${r.get('bid',0):7.2f} ${r.get('ask',0):7.2f} {int(r.get('volume',0) or 0):8d} {int(r.get('openInterest',0) or 0):8d} {r.get('impliedVolatility',0)*100:5.1f}%")

    if not ntm_puts.empty:
        print(f"\n  PUTS (near the money):")
        print(f"  {'Strike':>8s} {'Last':>8s} {'Bid':>8s} {'Ask':>8s} {'Vol':>8s} {'OI':>8s} {'IV':>6s}")
        for _, r in ntm_puts.iterrows():
            print(f"  ${r['strike']:7.2f} ${r.get('lastPrice',0):7.2f} ${r.get('bid',0):7.2f} ${r.get('ask',0):7.2f} {int(r.get('volume',0) or 0):8d} {int(r.get('openInterest',0) or 0):8d} {r.get('impliedVolatility',0)*100:5.1f}%")


def print_unusual(symbol: str, unusual: list[dict]):
    """Print unusual options activity."""
    if not unusual:
        print(f"\n  {symbol}: No unusual activity detected")
        return

    print(f"\n{'=' * 70}")
    print(f"  UNUSUAL OPTIONS ACTIVITY: {symbol}")
    print(f"{'=' * 70}")
    for u in unusual[:10]:
        icon = "📈" if u["type"] == "call" else "📉"
        print(f"  {icon} {u['type'].upper()} ${u['strike']} | Vol: {u['volume']:,} vs OI: {u['open_interest']:,} ({u['vol_oi_ratio']:.1f}x) | IV: {u['implied_vol']}%")


def print_pcr_watchlist(results: list[dict]):
    """Print PCR for a watchlist."""
    print(f"\n{'=' * 70}")
    print(f"  PUT/CALL RATIO SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")
    print(f"  {'Symbol':8s} {'PCR':>7s} {'Sentiment':15s} {'Call Vol':>10s} {'Put Vol':>10s}")
    print(f"  {'─'*8} {'─'*7} {'─'*15} {'─'*10} {'─'*10}")
    for r in sorted(results, key=lambda x: x.get("volume_pcr", 0), reverse=True):
        icon = {"very_bullish": "🟢🟢", "bullish": "🟢", "neutral": "⚪", "bearish": "🔴", "very_bearish": "🔴🔴"}.get(r.get("sentiment", ""), "❓")
        print(f"  {icon}{r['symbol']:7s} {r.get('volume_pcr', 0):7.3f} {r.get('sentiment', ''):15s} {r.get('call_volume', 0):10,d} {r.get('put_volume', 0):10,d}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Options Analysis")
    parser.add_argument("symbols", nargs="*", help="Symbols to analyze")
    parser.add_argument("--watchlist", "-w", type=str, help="Use a predefined watchlist")
    parser.add_argument("--unusual", "-u", action="store_true", help="Scan for unusual activity")
    parser.add_argument("--pcr", action="store_true", help="Put/call ratio analysis")
    parser.add_argument("--hedge", action="store_true", help="Suggest hedges")
    parser.add_argument("--expiry", type=int, default=0, help="Expiry index (0=nearest)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.watchlist:
        symbols = WATCHLISTS.get(args.watchlist, WATCHLISTS["default"])
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        print("Provide symbol(s) or --watchlist")
        return

    # PCR scan mode
    if args.pcr:
        results = []
        for sym in symbols:
            try:
                chain = get_options_chain(sym)
                if not chain.get("error"):
                    pcr = calculate_pcr(chain)
                    results.append(pcr)
            except Exception:
                continue
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_pcr_watchlist(results)
        return

    # Per-symbol analysis
    for sym in symbols:
        try:
            chain = get_options_chain(sym, args.expiry)
            if chain.get("error"):
                print(f"  ⚠️  {sym}: {chain['error']}")
                continue

            if args.unusual:
                unusual = find_unusual_activity(chain)
                if args.json:
                    print(json.dumps(unusual, indent=2))
                else:
                    print_unusual(sym, unusual)

            elif args.hedge:
                hedge = suggest_hedge(sym)
                if args.json:
                    print(json.dumps(hedge, indent=2))
                else:
                    print(f"\n  🛡️  HEDGE SUGGESTIONS: {sym}")
                    for s in hedge.get("suggestions", []):
                        print(f"     Buy {s['contracts']}x {sym} ${s['strike']} Put @ ${s['ask']:.2f} (cost: ${s['total_cost']:.2f}, {s['cost_pct_of_position']:.1f}% of position)")
            else:
                if args.json:
                    print(json.dumps({"symbol": sym, "pcr": calculate_pcr(chain)}, indent=2))
                else:
                    print_overview(chain)

        except Exception as e:
            print(f"  ❌ {sym}: {e}")


if __name__ == "__main__":
    main()
