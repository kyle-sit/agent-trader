#!/usr/bin/env python3
"""
Rebalance Portfolio — separate remediation script for portfolio constraint breaches.

Flow:
1. Load current portfolio and active hard-rule violations
2. Ask GPT for the best trim/close actions to move the portfolio back toward limits
3. Execute those reductions/closes (or show them in --dry-run mode)
4. Report the before/after state

Usage:
    python3 rebalance_portfolio.py              # Live rebalance execution
    python3 rebalance_portfolio.py --dry-run    # Show suggested actions only
    python3 rebalance_portfolio.py --json       # JSON output
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path.home() / ".hermes" / ".env")

os.environ['APCA_API_KEY_ID'] = os.getenv('APCA_API_KEY_ID', '')
os.environ['APCA_API_SECRET_KEY'] = os.getenv('APCA_API_SECRET_KEY', '')
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

from portfolio_risk import get_portfolio_data, check_risk_violations, load_risk_rules
from signal_executor import reduce_position, close_position, cancel_symbol_orders
from llm_hooks import recommend_rebalance_actions

DATA_DIR = Path.home() / "market-intel" / "data"
REBALANCE_LOG_FILE = DATA_DIR / "rebalance_log.jsonl"


def log_rebalance(entry: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry["logged_at"] = datetime.now().isoformat()
    with open(REBALANCE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_current_state() -> tuple[dict, list[dict]]:
    portfolio = get_portfolio_data()
    portfolio["rules"] = load_risk_rules()
    violations = check_risk_violations(portfolio)
    return portfolio, violations


def normalize_actions(plan: dict) -> list[dict]:
    actions = plan.get("actions", []) if isinstance(plan, dict) else []
    normalized = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        symbol = str(action.get("symbol", "")).upper().strip()
        act = str(action.get("action", "hold")).lower().strip()
        reduce_fraction = float(action.get("reduce_fraction", 0.0) or 0.0)
        cancel_open_orders = bool(action.get("cancel_open_orders", False))
        reason = str(action.get("reason", "")).strip()
        if not symbol:
            continue
        if act not in {"reduce", "close", "hold"}:
            continue
        if act == "close":
            reduce_fraction = 1.0
        elif act == "hold":
            reduce_fraction = 0.0
        else:
            reduce_fraction = max(0.0, min(1.0, reduce_fraction))
        normalized.append({
            "symbol": symbol,
            "action": act,
            "reduce_fraction": reduce_fraction,
            "cancel_open_orders": cancel_open_orders,
            "reason": reason,
        })
    return normalized


def execute_plan(actions: list[dict], portfolio: dict, dry_run: bool = False) -> list[dict]:
    positions_by_symbol = {p["symbol"]: p for p in portfolio.get("positions", [])}
    results = []

    for action in actions:
        symbol = action["symbol"]
        pos = positions_by_symbol.get(symbol)
        if not pos:
            results.append({
                "symbol": symbol,
                "success": False,
                "action": action["action"],
                "error": "No current position for symbol",
            })
            continue

        record = {
            "symbol": symbol,
            "requested_action": action["action"],
            "reduce_fraction": action["reduce_fraction"],
            "reason": action.get("reason", ""),
            "cancel_open_orders": action.get("cancel_open_orders", False),
            "dry_run": dry_run,
        }

        if action.get("cancel_open_orders"):
            if dry_run:
                record["cancel_orders_result"] = {"dry_run": True}
            else:
                record["cancel_orders_result"] = cancel_symbol_orders(symbol)

        if action["action"] == "hold":
            record["success"] = True
            record["result"] = {"action": "hold"}
            results.append(record)
            continue

        qty_abs = int(abs(pos.get("qty", 0)))
        if qty_abs <= 0:
            record["success"] = False
            record["error"] = "Position qty is zero"
            results.append(record)
            continue

        if action["action"] == "close" or action["reduce_fraction"] >= 0.999:
            if dry_run:
                record["success"] = True
                record["result"] = {"action": "close", "dry_run": True}
            else:
                record["result"] = close_position(symbol)
                record["success"] = bool(record["result"].get("success"))
            results.append(record)
            continue

        reduce_qty = max(int(round(qty_abs * action["reduce_fraction"])), 1)
        if dry_run:
            record["success"] = True
            record["result"] = {"action": "reduce", "qty": reduce_qty, "dry_run": True}
        else:
            record["result"] = reduce_position(symbol, reduce_qty, pos.get("side", "long"))
            record["success"] = bool(record["result"].get("success"))
        results.append(record)

    return results


def print_report(before_portfolio: dict, before_violations: list[dict], plan: dict, actions: list[dict], exec_results: list[dict], after_portfolio: dict | None = None, after_violations: list[dict] | None = None, dry_run: bool = False):
    print(f"\n{'=' * 70}")
    print(f"  PORTFOLIO REBALANCE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")
    print(f"\n  Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
    print(f"  Portfolio Value: ${before_portfolio.get('portfolio_value', 0):,.2f}")
    print(f"  Cash: {before_portfolio.get('cash_pct', 0):.1f}%")
    print(f"  Positions: {before_portfolio.get('num_positions', 0)}")
    print(f"  Top-5 Exposure: {before_portfolio.get('top5_exposure_pct', 0):.1f}%")
    print(f"\n  Violations before: {len(before_violations)}")
    for v in before_violations[:12]:
        print(f"    • {v.get('message', '')}")

    print(f"\n  LLM Summary: {plan.get('summary', 'No summary')}" )
    print(f"\n  Suggested actions ({len(actions)}):")
    if not actions:
        print("    • None")
    for a in actions:
        print(f"    • {a['symbol']}: {a['action']} {a['reduce_fraction']:.2f} — {a.get('reason', '')}")

    if exec_results:
        print(f"\n  Execution results:")
        for r in exec_results:
            status = '✅' if r.get('success') else '❌'
            print(f"    {status} {r['symbol']}: {r.get('requested_action')} — {r.get('reason', '')[:120]}")
            if r.get('error'):
                print(f"       error: {r['error']}")

    if after_portfolio is not None and after_violations is not None:
        print(f"\n  Violations after: {len(after_violations)}")
        for v in after_violations[:12]:
            print(f"    • {v.get('message', '')}")


def run_rebalance(dry_run: bool = False) -> dict:
    started = time.time()
    before_portfolio, before_violations = load_current_state()

    plan = recommend_rebalance_actions(before_portfolio, before_violations)
    actions = normalize_actions(plan)
    exec_results = execute_plan(actions, before_portfolio, dry_run=dry_run)

    after_portfolio = None
    after_violations = None
    if not dry_run:
        # Give broker state a moment to reflect submitted market orders
        time.sleep(1)
        after_portfolio, after_violations = load_current_state()

    result = {
        "started_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "before": {
            "portfolio": before_portfolio,
            "violations": before_violations,
        },
        "plan": plan,
        "actions": actions,
        "execution_results": exec_results,
        "after": {
            "portfolio": after_portfolio,
            "violations": after_violations,
        } if after_portfolio is not None else None,
        "duration_seconds": round(time.time() - started, 2),
    }

    log_rebalance(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Portfolio rebalance helper")
    parser.add_argument("--dry-run", action="store_true", help="Show and log the rebalance plan without executing orders")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    args = parser.parse_args()

    result = run_rebalance(dry_run=args.dry_run)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(
            result["before"]["portfolio"],
            result["before"]["violations"],
            result["plan"],
            result["actions"],
            result["execution_results"],
            result.get("after", {}).get("portfolio") if result.get("after") else None,
            result.get("after", {}).get("violations") if result.get("after") else None,
            dry_run=args.dry_run,
        )
        print(f"\n  Duration: {result['duration_seconds']:.2f}s")
        print(f"  Log file: {REBALANCE_LOG_FILE}")


if __name__ == "__main__":
    main()
