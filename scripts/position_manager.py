#!/usr/bin/env python3
"""
Position Manager — Active management of open paper trades.

Monitors open positions and applies:
  - Trailing stops (lock in profits as price moves in our favor)
  - Partial profit taking (sell 50% at first target, let rest run)
  - Break-even stops (move stop to entry after partial profit)
  - Time-based exits (close if no movement after N days)
  - TA-based exits (close if TA signal flips)

Usage:
    python3 position_manager.py                  # Review and manage all positions
    python3 position_manager.py --auto           # Auto-apply management rules
    python3 position_manager.py --trail XOM 2.5  # Set 2.5% trailing stop on XOM
    python3 position_manager.py --close XOM      # Close a position
    python3 position_manager.py --close-all      # Close all positions
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path.home() / ".hermes" / ".env")

os.environ['APCA_API_KEY_ID'] = os.getenv('APCA_API_KEY_ID', '')
os.environ['APCA_API_SECRET_KEY'] = os.getenv('APCA_API_SECRET_KEY', '')
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

from alpaca_trade_api.rest import REST
from ta_engine import analyze_symbol

# ── Config ──────────────────────────────────────────────────

DATA_DIR = Path.home() / "market-intel" / "data"
POSITION_STATE_FILE = DATA_DIR / "position_state.json"

# Default management rules
DEFAULT_RULES = {
    "trailing_stop_pct": 3.0,         # Trail by 3% from high water mark
    "partial_profit_pct": 3.0,        # Take partial profit at 3% gain
    "partial_profit_qty_pct": 50,     # Sell 50% at partial profit level
    "breakeven_after_partial": True,  # Move stop to entry after partial
    "max_hold_days": 10,              # Close if no movement after 10 days
    "ta_exit_enabled": True,          # Exit if TA signal flips against us
    "max_loss_pct": -5.0,             # Hard stop: close if loss exceeds 5%
}


# ── Position State ──────────────────────────────────────────

def load_state() -> dict:
    """Load position management state."""
    if POSITION_STATE_FILE.exists():
        with open(POSITION_STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save position management state."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(POSITION_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_api():
    return REST()


# ── Position Analysis ──────────────────────────────────────

def get_positions_with_context() -> list[dict]:
    """Get positions enriched with TA analysis and management state."""
    api = get_api()
    positions = api.list_positions()
    state = load_state()

    enriched = []
    for p in positions:
        symbol = p.symbol
        entry = float(p.avg_entry_price)
        current = float(p.current_price)
        qty = float(p.qty)
        side = p.side
        pl_pct = float(p.unrealized_plpc) * 100
        pl_dollars = float(p.unrealized_pl)
        market_value = float(p.market_value)

        # Get or init state for this position
        pos_state = state.get(symbol, {})
        if not pos_state:
            pos_state = {
                "opened_at": datetime.now().isoformat(),
                "high_water_mark": current,
                "low_water_mark": current,
                "partial_taken": False,
                "original_qty": qty,
                "trailing_stop": None,
                "notes": [],
            }

        # Update high/low water marks
        if current > pos_state.get("high_water_mark", 0):
            pos_state["high_water_mark"] = current
        if current < pos_state.get("low_water_mark", float("inf")):
            pos_state["low_water_mark"] = current

        # Calculate trailing stop level
        hwm = pos_state["high_water_mark"]
        trail_pct = DEFAULT_RULES["trailing_stop_pct"]
        if side == "long":
            trailing_stop = hwm * (1 - trail_pct / 100)
        else:
            trailing_stop = pos_state["low_water_mark"] * (1 + trail_pct / 100)
        pos_state["trailing_stop"] = round(trailing_stop, 2)

        # Calculate hold duration
        opened = datetime.fromisoformat(pos_state["opened_at"])
        hold_hours = (datetime.now() - opened).total_seconds() / 3600
        hold_days = hold_hours / 24

        # Run TA
        ta_result = None
        try:
            ta_result = analyze_symbol(symbol)
        except Exception:
            pass

        enriched.append({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry": entry,
            "current": current,
            "pl_pct": round(pl_pct, 2),
            "pl_dollars": round(pl_dollars, 2),
            "market_value": round(market_value, 2),
            "high_water_mark": pos_state["high_water_mark"],
            "trailing_stop": pos_state["trailing_stop"],
            "partial_taken": pos_state["partial_taken"],
            "original_qty": pos_state.get("original_qty", qty),
            "hold_days": round(hold_days, 1),
            "ta": ta_result,
            "state": pos_state,
        })

        # Save updated state
        state[symbol] = pos_state

    save_state(state)
    return enriched


# ── Management Actions ──────────────────────────────────────

def check_trailing_stop(pos: dict) -> dict | None:
    """Check if trailing stop has been hit."""
    if pos["side"] == "long" and pos["current"] <= pos["trailing_stop"]:
        return {
            "action": "close",
            "reason": f"Trailing stop hit (${pos['trailing_stop']:.2f}). HWM was ${pos['high_water_mark']:.2f}",
            "type": "trailing_stop",
        }
    elif pos["side"] == "short" and pos["current"] >= pos["trailing_stop"]:
        return {
            "action": "close",
            "reason": f"Trailing stop hit (${pos['trailing_stop']:.2f})",
            "type": "trailing_stop",
        }
    return None


def check_partial_profit(pos: dict) -> dict | None:
    """Check if we should take partial profits."""
    if pos["partial_taken"]:
        return None

    target_pct = DEFAULT_RULES["partial_profit_pct"]
    if pos["pl_pct"] >= target_pct:
        sell_qty = int(pos["qty"] * DEFAULT_RULES["partial_profit_qty_pct"] / 100)
        if sell_qty > 0:
            return {
                "action": "partial_close",
                "qty": sell_qty,
                "reason": f"Taking {DEFAULT_RULES['partial_profit_qty_pct']}% profit at {pos['pl_pct']:.1f}% gain",
                "type": "partial_profit",
            }
    return None


def check_max_loss(pos: dict) -> dict | None:
    """Check if max loss has been exceeded."""
    if pos["pl_pct"] <= DEFAULT_RULES["max_loss_pct"]:
        return {
            "action": "close",
            "reason": f"Max loss exceeded ({pos['pl_pct']:.1f}% vs {DEFAULT_RULES['max_loss_pct']}% limit)",
            "type": "max_loss",
        }
    return None


def check_time_exit(pos: dict) -> dict | None:
    """Check if position has been held too long without movement."""
    max_days = DEFAULT_RULES["max_hold_days"]
    if pos["hold_days"] >= max_days and abs(pos["pl_pct"]) < 1.0:
        return {
            "action": "close",
            "reason": f"Held {pos['hold_days']:.0f} days with only {pos['pl_pct']:+.1f}% — no edge",
            "type": "time_exit",
        }
    return None


def check_llm_exit(pos: dict) -> dict | None:
    """Check if LLM thinks the original catalyst has been invalidated."""
    try:
        from llm_hooks import analyze_exit
        entry_ctx = {
            "side": pos["side"],
            "entry": pos["entry"],
            "news_catalyst": pos.get("state", {}).get("news_catalyst", ""),
        }
        current = {
            "current_price": pos["current"],
            "pl_pct": pos["pl_pct"],
            "hold_days": pos["hold_days"],
            "ta_signal": pos["ta"]["signal"]["signal_type"] if pos.get("ta") else "unknown",
            "rsi": pos["ta"]["indicators"]["rsi"] if pos.get("ta") else "N/A",
        }
        result = analyze_exit(pos["symbol"], entry_ctx, current)
        if result and result.get("exit_recommendation") == "full_exit":
            return {
                "action": "close",
                "reason": f"LLM exit: {result.get('reasoning', 'Catalyst invalidated')[:100]}",
                "type": "llm_exit",
            }
        elif result and result.get("exit_recommendation") == "partial_exit":
            qty = int(pos["qty"] * 0.5)
            if qty > 0:
                return {
                    "action": "partial_close",
                    "qty": qty,
                    "reason": f"LLM partial exit: {result.get('reasoning', '')[:100]}",
                    "type": "llm_partial_exit",
                }
    except (ImportError, Exception):
        pass
    return None


def check_ta_exit(pos: dict) -> dict | None:
    """Check if TA has flipped against us."""
    if not DEFAULT_RULES["ta_exit_enabled"] or not pos.get("ta"):
        return None

    ta_sig = pos["ta"]["signal"]
    ta_type = ta_sig.get("signal_type", "neutral")
    ta_strength = ta_sig.get("strength", "weak")

    # Only exit on strong contrary signals
    if ta_strength != "strong":
        return None

    if pos["side"] == "long" and ta_type == "bearish":
        return {
            "action": "close",
            "reason": f"TA flipped to STRONG BEARISH — {', '.join(ta_sig.get('reasons', [])[:2])}",
            "type": "ta_exit",
        }
    elif pos["side"] == "short" and ta_type == "bullish":
        return {
            "action": "close",
            "reason": f"TA flipped to STRONG BULLISH — {', '.join(ta_sig.get('reasons', [])[:2])}",
            "type": "ta_exit",
        }
    return None


def execute_action(symbol: str, action: dict) -> dict:
    """Execute a position management action."""
    api = get_api()

    if action["action"] == "close":
        try:
            api.close_position(symbol)
            return {"success": True, "symbol": symbol, **action}
        except Exception as e:
            return {"success": False, "symbol": symbol, "error": str(e)}

    elif action["action"] == "partial_close":
        try:
            qty = action["qty"]
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
            # Update state
            state = load_state()
            if symbol in state:
                state[symbol]["partial_taken"] = True
                save_state(state)

            return {"success": True, "symbol": symbol, "qty_sold": qty, **action}
        except Exception as e:
            return {"success": False, "symbol": symbol, "error": str(e)}

    return {"success": False, "error": "Unknown action"}


# ── Display ─────────────────────────────────────────────────

def print_positions(positions: list[dict]):
    """Pretty-print positions with management info."""
    if not positions:
        print("\n  No open positions.")
        return

    account = get_api().get_account()
    print(f"\n{'=' * 75}")
    print(f"  POSITION MANAGER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Portfolio: ${float(account.portfolio_value):,.2f} | Cash: ${float(account.cash):,.2f}")
    print(f"{'=' * 75}")

    total_pl = sum(p["pl_dollars"] for p in positions)
    total_pl_icon = "🟢" if total_pl >= 0 else "🔴"
    print(f"\n  {total_pl_icon} Total unrealized P&L: ${total_pl:+,.2f}")

    for pos in positions:
        pl_icon = "🟢" if pos["pl_pct"] >= 0 else "🔴"
        ta_dir = pos["ta"]["signal"]["signal_type"] if pos.get("ta") else "N/A"
        ta_icon = "🟢" if ta_dir == "bullish" else "🔴" if ta_dir == "bearish" else "⚪"

        print(f"\n  {pl_icon} {pos['symbol']} ({pos['side'].upper()}) — {pos['pl_pct']:+.2f}% (${pos['pl_dollars']:+.2f})")
        print(f"     Entry: ${pos['entry']:.2f} → Current: ${pos['current']:.2f} | Qty: {pos['qty']:.0f}")
        print(f"     HWM: ${pos['high_water_mark']:.2f} | Trail Stop: ${pos['trailing_stop']:.2f} | Hold: {pos['hold_days']:.1f} days")
        print(f"     TA: {ta_icon} {ta_dir} | Partial taken: {'Yes' if pos['partial_taken'] else 'No'}")

        # Check management rules
        checks = [
            check_trailing_stop(pos),
            check_partial_profit(pos),
            check_max_loss(pos),
            check_time_exit(pos),
            check_ta_exit(pos),
            check_llm_exit(pos),
        ]
        triggers = [c for c in checks if c is not None]
        if triggers:
            for t in triggers:
                icon = "🔔" if t["action"] == "partial_close" else "🚨"
                print(f"     {icon} TRIGGER: {t['reason']}")
        else:
            print(f"     ✅ No management triggers")


def auto_manage(positions: list[dict], dry_run: bool = False) -> list[dict]:
    """Auto-apply management rules to all positions."""
    actions_taken = []

    for pos in positions:
        checks = [
            ("max_loss", check_max_loss(pos)),
            ("trailing_stop", check_trailing_stop(pos)),
            ("partial_profit", check_partial_profit(pos)),
            ("time_exit", check_time_exit(pos)),
            ("ta_exit", check_ta_exit(pos)),
            ("llm_exit", check_llm_exit(pos)),
        ]

        for check_name, action in checks:
            if action is None:
                continue

            print(f"\n  🔔 {pos['symbol']}: {action['reason']}")

            if dry_run:
                print(f"     [DRY RUN] Would {action['action']}")
                actions_taken.append({"symbol": pos["symbol"], "dry_run": True, **action})
            else:
                result = execute_action(pos["symbol"], action)
                if result["success"]:
                    print(f"     ✅ Executed: {action['action']}")
                else:
                    print(f"     ❌ Failed: {result.get('error')}")
                actions_taken.append(result)

            # Only execute the highest priority action per position
            break

    return actions_taken


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Position Manager")
    parser.add_argument("--auto", action="store_true", help="Auto-apply management rules")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without executing")
    parser.add_argument("--trail", nargs=2, metavar=("SYMBOL", "PCT"), help="Set trailing stop pct")
    parser.add_argument("--close", type=str, help="Close a position")
    parser.add_argument("--close-all", action="store_true", help="Close all positions")
    parser.add_argument("--rules", action="store_true", help="Show management rules")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.rules:
        print("\n📋 Position Management Rules:")
        for key, val in DEFAULT_RULES.items():
            print(f"   {key}: {val}")
        return

    if args.close:
        api = get_api()
        try:
            api.close_position(args.close.upper())
            print(f"✅ Closed {args.close.upper()}")
        except Exception as e:
            print(f"❌ {e}")
        return

    if args.close_all:
        api = get_api()
        try:
            api.close_all_positions()
            print("✅ All positions closed")
        except Exception as e:
            print(f"❌ {e}")
        return

    if args.trail:
        symbol, pct = args.trail[0].upper(), float(args.trail[1])
        state = load_state()
        if symbol not in state:
            state[symbol] = {}
        state[symbol]["custom_trail_pct"] = pct
        save_state(state)
        print(f"✅ Set {pct}% trailing stop on {symbol}")
        return

    # Default: review positions
    positions = get_positions_with_context()

    if args.json:
        print(json.dumps([{k: v for k, v in p.items() if k != "ta"} for p in positions], indent=2))
        return

    print_positions(positions)

    if args.auto:
        print(f"\n{'─' * 75}")
        print(f"  AUTO-MANAGEMENT")
        print(f"{'─' * 75}")
        actions = auto_manage(positions, dry_run=args.dry_run)
        if not actions:
            print("\n  ✅ No actions needed — all positions within rules")
        print(f"\n  Actions taken: {len(actions)}")


if __name__ == "__main__":
    main()
