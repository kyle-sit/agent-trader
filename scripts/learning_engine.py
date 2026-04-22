#!/usr/bin/env python3
"""
Learning Engine — Self-improving feedback loop for the Market Intelligence Pipeline.

Tracks trade outcomes against initial predictions and builds a pattern database
that improves future signal scoring.

Flow:
  1. When a trade is opened → record full context (news, TA, conviction, score)
  2. When a trade is closed → record outcome (P&L, hit target/stop, duration)
  3. Compare prediction vs reality → grade the signal
  4. Build pattern database: which signal types, news categories, TA setups
     actually lead to profitable trades
  5. Feed learnings back into correlation engine scoring

Usage:
    python3 learning_engine.py --review              # Review all open trades, grade closed ones
    python3 learning_engine.py --report              # Performance report with learnings
    python3 learning_engine.py --patterns             # Show learned patterns
    python3 learning_engine.py --grade AAPL           # Manually grade a specific trade
    python3 learning_engine.py --adjust-weights       # Recalculate scoring weights from data
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path.home() / ".hermes" / ".env")

os.environ['APCA_API_KEY_ID'] = os.getenv('APCA_API_KEY_ID', '')
os.environ['APCA_API_SECRET_KEY'] = os.getenv('APCA_API_SECRET_KEY', '')
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'


# ── File Paths ──────────────────────────────────────────────

DATA_DIR = Path.home() / "market-intel" / "data"
TRADE_LOG = DATA_DIR / "trade_log.jsonl"
OUTCOMES_LOG = DATA_DIR / "trade_outcomes.jsonl"
PATTERNS_DB = DATA_DIR / "learned_patterns.json"
WEIGHTS_FILE = DATA_DIR / "scoring_weights.json"


# ── Default Scoring Weights ─────────────────────────────────
# These start as defaults and get adjusted by learning

DEFAULT_WEIGHTS = {
    # News signal weights
    "news_impact_multiplier": 10.0,
    "confidence_multipliers": {"high": 1.0, "medium": 0.7, "low": 0.4},
    "timeframe_multipliers": {"immediate": 1.0, "short-term": 0.8, "medium-term": 0.5, "long-term": 0.3},

    # TA signal weights
    "ta_base_score": 50.0,
    "strength_multipliers": {"strong": 1.0, "moderate": 0.6, "weak": 0.3},

    # Alignment bonuses
    "confirmed_bonus": 1.3,       # Multiply score when news + TA agree
    "conflicting_penalty": 0.5,   # Reduce score when they disagree

    # Pattern-based adjustments (learned over time)
    "news_category_adjustments": {},   # e.g., {"iran": 1.2, "economy": 0.8}
    "ta_pattern_adjustments": {},      # e.g., {"RSI oversold": 1.1, "MACD crossover": 0.9}
    "ticker_adjustments": {},          # e.g., {"XOM": 1.15, "SPY": 0.95}
    "timeofday_adjustments": {},       # e.g., {"morning": 1.1, "afternoon": 0.9}
    "alignment_accuracy": {            # Win rate by alignment type
        "confirmed": {"trades": 0, "wins": 0},
        "conflicting": {"trades": 0, "wins": 0},
        "neutral_ta": {"trades": 0, "wins": 0},
    },
}


def load_weights() -> dict:
    """Load learned scoring weights, or return defaults."""
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            saved = json.load(f)
        # Merge with defaults to handle new fields
        merged = DEFAULT_WEIGHTS.copy()
        merged.update(saved)
        return merged
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict):
    """Save learned scoring weights."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)


# ── Trade Outcome Tracking ──────────────────────────────────

def load_trade_log() -> list[dict]:
    """Load all trades from the log."""
    if not TRADE_LOG.exists():
        return []
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trades


def load_outcomes() -> list[dict]:
    """Load graded trade outcomes."""
    if not OUTCOMES_LOG.exists():
        return []
    outcomes = []
    with open(OUTCOMES_LOG) as f:
        for line in f:
            try:
                outcomes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return outcomes


def save_outcome(outcome: dict):
    """Append a graded outcome."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outcome["graded_at"] = datetime.now().isoformat()
    with open(OUTCOMES_LOG, "a") as f:
        f.write(json.dumps(outcome) + "\n")


def get_alpaca_closed_orders() -> list[dict]:
    """Get recently closed/filled orders from Alpaca."""
    from alpaca_trade_api.rest import REST
    api = REST()
    try:
        orders = api.list_orders(status="closed", limit=100)
        result = []
        for o in orders:
            result.append({
                "id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "qty": float(o.qty) if o.qty else 0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                "filled_at": str(o.filled_at) if o.filled_at else None,
                "type": o.type,
                "status": o.status,
            })
        return result
    except Exception as e:
        print(f"⚠️  Alpaca error: {e}")
        return []


def get_current_positions() -> dict:
    """Get current positions as a dict keyed by symbol."""
    from alpaca_trade_api.rest import REST
    api = REST()
    try:
        positions = api.list_positions()
        return {
            p.symbol: {
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "market_value": float(p.market_value),
            }
            for p in positions
        }
    except Exception as e:
        print(f"⚠️  Alpaca error: {e}")
        return {}


# ── Grading System ──────────────────────────────────────────

def grade_trade(signal: dict, actual_pl_pct: float, hit_target: bool,
                hit_stop: bool, duration_hours: float) -> dict:
    """
    Grade a trade by comparing prediction vs outcome.

    Grades:
      A+ : Hit target, direction correct, high conviction was justified
      A  : Profitable, direction correct
      B  : Small profit or breakeven
      C  : Small loss but direction was partially right
      D  : Loss, direction wrong
      F  : Hit stop loss, prediction completely wrong
    """

    direction = signal.get("direction", "long")
    conviction = signal.get("conviction", "low")
    predicted_score = signal.get("score", 0)
    alignment = signal.get("alignment", "neutral_ta")
    news_impact = signal.get("news_impact", 0)

    # Was the direction prediction correct?
    direction_correct = (
        (direction == "long" and actual_pl_pct > 0) or
        (direction == "short" and actual_pl_pct < 0)
    )

    # Grade
    if hit_target and direction_correct:
        grade = "A+"
        grade_score = 100
    elif direction_correct and abs(actual_pl_pct) > 3:
        grade = "A"
        grade_score = 90
    elif direction_correct and abs(actual_pl_pct) > 1:
        grade = "B"
        grade_score = 75
    elif abs(actual_pl_pct) < 1:
        grade = "C"
        grade_score = 50
    elif hit_stop:
        grade = "F"
        grade_score = 0
    elif not direction_correct and abs(actual_pl_pct) > 3:
        grade = "D"
        grade_score = 20
    else:
        grade = "C"
        grade_score = 40

    # Conviction accuracy — did conviction match outcome quality?
    conviction_rank = {"high": 3, "medium": 2, "low": 1}[conviction]
    if grade in ("A+", "A") and conviction_rank >= 2:
        conviction_accuracy = "correct"
    elif grade in ("D", "F") and conviction_rank >= 2:
        conviction_accuracy = "overconfident"
    elif grade in ("A+", "A") and conviction_rank == 1:
        conviction_accuracy = "underconfident"
    else:
        conviction_accuracy = "neutral"

    return {
        "symbol": signal.get("ticker", ""),
        "direction": direction,
        "conviction": conviction,
        "predicted_score": predicted_score,
        "alignment": alignment,
        "actual_pl_pct": round(actual_pl_pct, 2),
        "hit_target": hit_target,
        "hit_stop": hit_stop,
        "duration_hours": round(duration_hours, 1),
        "direction_correct": direction_correct,
        "grade": grade,
        "grade_score": grade_score,
        "conviction_accuracy": conviction_accuracy,
        "news_catalyst": signal.get("news_catalyst", ""),
        "news_impact": news_impact,
        "ta_signal": signal.get("ta_signal", ""),
        "ta_strength": signal.get("ta_strength", ""),
        "ta_reasons": signal.get("ta_reasons", []),
        "topics": signal.get("topics", []),
    }


# ── Pattern Learning ────────────────────────────────────────

def update_patterns(outcomes: list[dict]) -> dict:
    """Analyze all outcomes to find patterns in what works and what doesn't."""

    patterns = {
        "total_trades": len(outcomes),
        "win_rate": 0,
        "avg_grade_score": 0,
        "avg_pl_pct": 0,

        # By conviction level
        "by_conviction": defaultdict(lambda: {"trades": 0, "wins": 0, "avg_pl": 0, "total_pl": 0}),

        # By alignment type
        "by_alignment": defaultdict(lambda: {"trades": 0, "wins": 0, "avg_pl": 0, "total_pl": 0}),

        # By news category/topic
        "by_topic": defaultdict(lambda: {"trades": 0, "wins": 0, "avg_pl": 0, "total_pl": 0}),

        # By TA pattern
        "by_ta_reason": defaultdict(lambda: {"trades": 0, "wins": 0, "avg_pl": 0, "total_pl": 0}),

        # By ticker
        "by_ticker": defaultdict(lambda: {"trades": 0, "wins": 0, "avg_pl": 0, "total_pl": 0}),

        # By news impact score
        "by_news_impact": defaultdict(lambda: {"trades": 0, "wins": 0, "avg_pl": 0, "total_pl": 0}),

        # Best and worst trades
        "best_trades": [],
        "worst_trades": [],

        # Key learnings (human-readable)
        "learnings": [],
    }

    if not outcomes:
        return dict(patterns)

    wins = sum(1 for o in outcomes if o["direction_correct"])
    patterns["win_rate"] = round(wins / len(outcomes) * 100, 1)
    patterns["avg_grade_score"] = round(sum(o["grade_score"] for o in outcomes) / len(outcomes), 1)
    patterns["avg_pl_pct"] = round(sum(o["actual_pl_pct"] for o in outcomes) / len(outcomes), 2)

    for o in outcomes:
        pl = o["actual_pl_pct"]
        win = o["direction_correct"]

        # By conviction
        conv = o["conviction"]
        patterns["by_conviction"][conv]["trades"] += 1
        patterns["by_conviction"][conv]["wins"] += int(win)
        patterns["by_conviction"][conv]["total_pl"] += pl

        # By alignment
        align = o["alignment"]
        patterns["by_alignment"][align]["trades"] += 1
        patterns["by_alignment"][align]["wins"] += int(win)
        patterns["by_alignment"][align]["total_pl"] += pl

        # By topic
        topics = o.get("topics", [])
        if not topics:
            # Try to infer from news catalyst
            catalyst = o.get("news_catalyst", "").lower()
            if "iran" in catalyst: topics = ["iran"]
            elif "fed" in catalyst or "rate" in catalyst: topics = ["economy"]
            elif "tariff" in catalyst or "trade" in catalyst: topics = ["economy"]
        for topic in topics:
            patterns["by_topic"][topic]["trades"] += 1
            patterns["by_topic"][topic]["wins"] += int(win)
            patterns["by_topic"][topic]["total_pl"] += pl

        # By TA reason
        for reason in o.get("ta_reasons", []):
            # Simplify reason to pattern name
            pattern_name = simplify_ta_reason(reason)
            patterns["by_ta_reason"][pattern_name]["trades"] += 1
            patterns["by_ta_reason"][pattern_name]["wins"] += int(win)
            patterns["by_ta_reason"][pattern_name]["total_pl"] += pl

        # By ticker
        ticker = o["symbol"]
        patterns["by_ticker"][ticker]["trades"] += 1
        patterns["by_ticker"][ticker]["wins"] += int(win)
        patterns["by_ticker"][ticker]["total_pl"] += pl

        # By news impact
        impact = str(o.get("news_impact", 0))
        patterns["by_news_impact"][impact]["trades"] += 1
        patterns["by_news_impact"][impact]["wins"] += int(win)
        patterns["by_news_impact"][impact]["total_pl"] += pl

    # Calculate averages
    for category in ["by_conviction", "by_alignment", "by_topic", "by_ta_reason", "by_ticker", "by_news_impact"]:
        for key, data in patterns[category].items():
            if data["trades"] > 0:
                data["avg_pl"] = round(data["total_pl"] / data["trades"], 2)
                data["win_rate"] = round(data["wins"] / data["trades"] * 100, 1)

    # Best and worst trades
    sorted_by_pl = sorted(outcomes, key=lambda o: o["actual_pl_pct"])
    patterns["best_trades"] = [
        {"symbol": o["symbol"], "pl_pct": o["actual_pl_pct"], "grade": o["grade"], "catalyst": o["news_catalyst"][:60]}
        for o in sorted_by_pl[-3:]
    ]
    patterns["worst_trades"] = [
        {"symbol": o["symbol"], "pl_pct": o["actual_pl_pct"], "grade": o["grade"], "catalyst": o["news_catalyst"][:60]}
        for o in sorted_by_pl[:3]
    ]

    # Generate learnings
    patterns["learnings"] = generate_learnings(patterns)

    # Convert defaultdicts to regular dicts for JSON serialization
    for key in ["by_conviction", "by_alignment", "by_topic", "by_ta_reason", "by_ticker", "by_news_impact"]:
        patterns[key] = dict(patterns[key])

    return patterns


def simplify_ta_reason(reason: str) -> str:
    """Simplify a TA reason string to a pattern name."""
    reason_lower = reason.lower()
    if "ema alignment bullish" in reason_lower: return "ema_bullish_alignment"
    if "ema alignment bearish" in reason_lower: return "ema_bearish_alignment"
    if "above 200 sma" in reason_lower: return "above_200sma"
    if "below 200 sma" in reason_lower: return "below_200sma"
    if "macd bullish" in reason_lower: return "macd_bullish_cross"
    if "macd bearish" in reason_lower: return "macd_bearish_cross"
    if "rsi oversold" in reason_lower: return "rsi_oversold"
    if "rsi overbought" in reason_lower: return "rsi_overbought"
    if "rsi strong" in reason_lower: return "rsi_strong"
    if "rsi weak" in reason_lower: return "rsi_weak"
    if "stochrsi oversold" in reason_lower: return "stochrsi_oversold"
    if "stochrsi overbought" in reason_lower: return "stochrsi_overbought"
    if "strong trend" in reason_lower or "strong downtrend" in reason_lower: return "strong_adx"
    if "high volume rally" in reason_lower: return "volume_spike_up"
    if "high volume selloff" in reason_lower: return "volume_spike_down"
    if "obv rising" in reason_lower: return "obv_accumulation"
    if "obv falling" in reason_lower: return "obv_distribution"
    if "below lower bollinger" in reason_lower: return "bb_oversold"
    if "above upper bollinger" in reason_lower: return "bb_overbought"
    if "near support" in reason_lower: return "near_support"
    if "near resistance" in reason_lower: return "near_resistance"
    return reason[:30]


def generate_learnings(patterns: dict) -> list[str]:
    """Generate human-readable learnings from patterns."""
    learnings = []

    # Conviction accuracy
    for conv, data in patterns["by_conviction"].items():
        if data["trades"] >= 3:
            wr = data.get("win_rate", 0)
            if conv == "high" and wr < 50:
                learnings.append(f"⚠️ HIGH conviction trades only winning {wr}% — we're overconfident. Lower conviction threshold or tighten criteria.")
            elif conv == "high" and wr > 70:
                learnings.append(f"✅ HIGH conviction trades winning {wr}% — our high-conviction filter is working well.")
            if conv == "low" and wr > 60:
                learnings.append(f"💡 LOW conviction trades winning {wr}% — we might be underrating these signals.")

    # Alignment patterns
    for align, data in patterns["by_alignment"].items():
        if data["trades"] >= 3:
            wr = data.get("win_rate", 0)
            avg_pl = data.get("avg_pl", 0)
            if align == "confirmed" and wr > 60:
                learnings.append(f"✅ CONFIRMED (news+TA aligned) trades: {wr}% win rate, avg {avg_pl:+.1f}% — alignment strategy is working.")
            if align == "conflicting" and wr < 40:
                learnings.append(f"⚠️ CONFLICTING signal trades: {wr}% win rate — avoid trading when news and TA disagree.")
            if align == "neutral_ta" and wr > 50:
                learnings.append(f"💡 NEUTRAL TA trades still winning {wr}% — news-only signals have value even without TA confirmation.")

    # Topic performance
    for topic, data in patterns["by_topic"].items():
        if data["trades"] >= 3:
            wr = data.get("win_rate", 0)
            avg_pl = data.get("avg_pl", 0)
            if wr > 65:
                learnings.append(f"✅ {topic.upper()} news trades: {wr}% win rate — we read this topic well.")
            elif wr < 35:
                learnings.append(f"⚠️ {topic.upper()} news trades: {wr}% win rate — we're misreading signals from this topic.")

    # TA pattern performance
    best_ta = sorted(
        [(k, v) for k, v in patterns["by_ta_reason"].items() if v["trades"] >= 3],
        key=lambda x: x[1].get("win_rate", 0),
        reverse=True
    )
    if best_ta:
        top = best_ta[0]
        learnings.append(f"📊 Best TA pattern: {top[0]} ({top[1]['win_rate']}% win rate over {top[1]['trades']} trades)")
    if len(best_ta) > 1:
        worst = best_ta[-1]
        if worst[1].get("win_rate", 100) < 40:
            learnings.append(f"📊 Worst TA pattern: {worst[0]} ({worst[1]['win_rate']}% win rate) — consider downweighting")

    # Ticker performance
    for ticker, data in patterns["by_ticker"].items():
        if data["trades"] >= 3:
            wr = data.get("win_rate", 0)
            if wr > 70:
                learnings.append(f"✅ {ticker}: {wr}% win rate — we trade this well.")
            elif wr < 30:
                learnings.append(f"⚠️ {ticker}: {wr}% win rate — avoid or reduce position size.")

    if not learnings:
        learnings.append("📝 Not enough trade data yet. Need 3+ graded trades per category to generate learnings.")

    return learnings


# ── Weight Adjustment ───────────────────────────────────────

def adjust_weights_from_outcomes(outcomes: list[dict]) -> dict:
    """Recalculate scoring weights based on actual trade outcomes."""
    weights = load_weights()
    patterns = update_patterns(outcomes)

    if len(outcomes) < 5:
        print("⚠️  Need at least 5 graded trades to adjust weights. Current:", len(outcomes))
        return weights

    # Adjust news category weights
    for topic, data in patterns.get("by_topic", {}).items():
        if data["trades"] >= 3:
            baseline_wr = patterns["win_rate"] / 100
            topic_wr = data["win_rate"] / 100
