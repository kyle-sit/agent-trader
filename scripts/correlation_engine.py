#!/usr/bin/env python3
"""
Correlation Engine — Phase 3 of Market Intelligence Pipeline.

Combines news signals (LLM analysis) with technical analysis signals
to produce high-conviction trade recommendations.

Flow:
  1. Fetch + analyze news → market impact assessments
  2. Run TA on affected tickers → technical signals
  3. Correlate: news catalyst + TA confirmation = trade signal
  4. Score, rank, and output actionable trades

Usage:
    python3 correlation_engine.py                    # Full pipeline scan
    python3 correlation_engine.py --watchlist iran    # Focus on Iran-related assets
    python3 correlation_engine.py --dry-run           # Skip LLM, use cached/mock news
    python3 correlation_engine.py --json              # JSON output for alerts
"""

import json
import os
import sys
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from ta_engine import analyze_symbol, WATCHLISTS, IndicatorValues
from news_analyzer import (
    fetch_rss_headlines,
    fetch_finnhub_news,
    fetch_alpaca_news,
    fetch_twitter_trending,
    analyze_with_llm,
    TOPIC_ASSETS,
)
from learning_engine import load_weights, simplify_ta_reason

# ── Learned Weights (loaded once at startup) ────────────────

WEIGHTS = load_weights()
TA_MAX_WORKERS = max(1, int(os.getenv("TA_MAX_WORKERS", "8")))
CORRELATION_LLM_MAX_TICKERS = max(0, int(os.getenv("CORRELATION_LLM_MAX_TICKERS", "3")))
MAX_TICKERS_TO_ANALYZE = max(1, int(os.getenv("MAX_TICKERS_TO_ANALYZE", "10")))
MAX_HEADLINES_FOR_LLM = max(1, int(os.getenv("MAX_HEADLINES_FOR_LLM", "20")))


def _headline_priority(item: dict) -> tuple:
    """Rank headlines for expensive LLM analysis."""
    source = str(item.get("source", "") or "")
    relevance = float(item.get("relevance_score") or 0)
    has_symbols = 1 if item.get("symbols") else 0
    age_minutes = float(item.get("age_minutes") or 10_000)

    if source.startswith("twitter_t1_"):
        source_rank = 4
    elif source == "alpaca":
        source_rank = 3
    elif source == "finnhub":
        source_rank = 2
    else:
        source_rank = 1

    return (source_rank, relevance, has_symbols, -age_minutes)


# ── Data Classes ────────────────────────────────────────────

@dataclass
class TradeSignal:
    """A correlated trade signal combining news + TA."""
    ticker: str
    direction: str                  # "long" or "short"
    conviction: str                 # "high", "medium", "low"
    score: float                    # -100 to +100 composite score

    # News side
    news_catalyst: str
    news_impact: int                # -5 to +5
    news_confidence: str
    news_timeframe: str

    # TA side
    ta_signal: str                  # "bullish", "bearish", "neutral"
    ta_strength: str                # "strong", "moderate", "weak"
    ta_reasons: list

    # Trade parameters
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    risk_reward: Optional[float] = None

    # Context
    alignment: str = ""             # "confirmed", "conflicting", "neutral_ta"
    action: str = ""                # "enter", "watch", "avoid"
    notes: list = field(default_factory=list)


# ── Correlation Logic ───────────────────────────────────────

def compute_news_score(event: dict, ticker: str = "") -> float:
    """Convert news event to a -50 to +50 score, adjusted by learned weights."""
    w = WEIGHTS
    impact = event.get("impact_score", 0)  # -5 to +5
    confidence_mult = w.get("confidence_multipliers", {}).get(event.get("confidence", "low"), 0.4)
    timeframe_mult = w.get("timeframe_multipliers", {}).get(event.get("timeframe", "medium-term"), 0.5)
    impact_mult = w.get("news_impact_multiplier", 10.0)

    base_score = impact * impact_mult * confidence_mult * timeframe_mult

    # Apply learned topic adjustments
    topic_adj = 1.0
    for topic in event.get("topics", []):
        adj = w.get("news_category_adjustments", {}).get(topic, 1.0)
        topic_adj *= adj
    base_score *= topic_adj

    # Apply learned ticker adjustment
    ticker_adj = w.get("ticker_adjustments", {}).get(ticker, 1.0)
    base_score *= ticker_adj

    return base_score


def compute_ta_score(signal: dict, ticker: str = "") -> float:
    """Convert TA signal to a -50 to +50 score, adjusted by learned weights."""
    w = WEIGHTS
    direction = 1 if signal.get("signal_type") == "bullish" else -1 if signal.get("signal_type") == "bearish" else 0
    strength_mult = w.get("strength_multipliers", {}).get(signal.get("strength", "weak"), 0.3)
    ta_base = w.get("ta_base_score", 50.0)

    # Count reasons, weighted by learned pattern performance
    reasons = signal.get("reasons", [])
    weighted_count = 0
    for reason in reasons:
        pattern = simplify_ta_reason(reason)
        pattern_adj = w.get("ta_pattern_adjustments", {}).get(pattern, 1.0)
        weighted_count += pattern_adj

    confluence = min(weighted_count / 6, 1.0)  # Normalize to 0-1

    base_score = direction * ta_base * strength_mult * confluence

    # Apply learned ticker adjustment
    ticker_adj = w.get("ticker_adjustments", {}).get(ticker, 1.0)
    base_score *= ticker_adj

    return base_score


def aggregate_events_by_ticker(events: list[dict]) -> dict[str, dict]:
    """Aggregate all news events into one merged catalyst per ticker."""
    ticker_events: dict[str, list[dict]] = {}

    for event in events:
        event_tickers = list(event.get("affected_tickers", []) or [])
        if not event_tickers:
            for topic in event.get("topics", []):
                if topic in TOPIC_ASSETS:
                    event_tickers.extend(TOPIC_ASSETS[topic]["tickers"])
        event_tickers = list(dict.fromkeys(event_tickers))

        for ticker in event_tickers:
            ticker_events.setdefault(ticker, []).append(event)

    aggregated = {}
    for ticker, grouped_events in ticker_events.items():
        if not grouped_events:
            continue

        headline_weights = []
        topics = []
        directions = []
        confidence_rank = {"low": 1, "medium": 2, "high": 3}
        timeframe_rank = {"immediate": 4, "short-term": 3, "medium-term": 2, "long-term": 1}

        total_impact = 0.0
        weighted_impact = 0.0
        total_weight = 0.0
        best_confidence = "low"
        best_timeframe = "long-term"

        for event in grouped_events:
            impact = float(event.get("impact_score", 0) or 0)
            confidence = event.get("confidence", "low")
            timeframe = event.get("timeframe", "medium-term")
            weight = confidence_rank.get(confidence, 1)

            total_impact += impact
            weighted_impact += impact * weight
            total_weight += weight
            directions.append(event.get("direction", "neutral"))

            if confidence_rank.get(confidence, 1) > confidence_rank.get(best_confidence, 1):
                best_confidence = confidence
            if timeframe_rank.get(timeframe, 0) > timeframe_rank.get(best_timeframe, 0):
                best_timeframe = timeframe

            for topic in event.get("topics", []):
                if topic not in topics:
                    topics.append(topic)

            headline = event.get("headline", "")
            if headline:
                headline_weights.append((abs(impact), headline))

        avg_impact = weighted_impact / total_weight if total_weight else 0.0
        if avg_impact > 0.25:
            direction = "bullish"
        elif avg_impact < -0.25:
            direction = "bearish"
        else:
            bullish_count = sum(1 for d in directions if d == "bullish")
            bearish_count = sum(1 for d in directions if d == "bearish")
            if bullish_count > bearish_count:
                direction = "bullish"
            elif bearish_count > bullish_count:
                direction = "bearish"
            else:
                direction = "neutral"

        headline_weights.sort(key=lambda item: item[0], reverse=True)
        top_headlines = [headline for _, headline in headline_weights[:3]]
        catalyst = top_headlines[0] if top_headlines else f"Aggregated catalyst for {ticker}"

        aggregated[ticker] = {
            "ticker": ticker,
            "headline": catalyst,
            "headlines": top_headlines,
            "headline_count": len(grouped_events),
            "impact_score": round(avg_impact, 2),
            "total_impact_score": round(total_impact, 2),
            "confidence": best_confidence,
            "timeframe": best_timeframe,
            "topics": topics,
            "affected_tickers": [ticker],
            "direction": direction,
            "reasoning": f"Merged {len(grouped_events)} event(s) for {ticker}",
            "source_events": grouped_events,
        }

    return aggregated


def correlate_signals(news_event: dict, ta_result: dict, ticker: str, use_llm: bool = True) -> TradeSignal:
    """Combine a news event with TA analysis for a specific ticker."""
    w = WEIGHTS
    ta_sig = ta_result["signal"]
    ta_ind = ta_result["indicators"]

    news_score = compute_news_score(news_event, ticker)
    ta_score = compute_ta_score(ta_sig, ticker)
    composite = news_score + ta_score

    # Determine alignment
    news_dir = news_event.get("direction", "neutral")
    ta_dir = ta_sig.get("signal_type", "neutral")

    if news_dir == ta_dir and news_dir != "neutral":
        alignment = "confirmed"
    elif ta_dir == "neutral":
        alignment = "neutral_ta"
    elif news_dir != "neutral" and ta_dir != "neutral" and news_dir != ta_dir:
        alignment = "conflicting"
    else:
        alignment = "neutral_ta"

    # Apply learned alignment multipliers
    if alignment == "confirmed":
        composite *= w.get("confirmed_bonus", 1.3)
    elif alignment == "conflicting":
        composite *= w.get("conflicting_penalty", 0.5)

    # Direction follows news if aligned, or news if TA neutral
    if alignment == "confirmed":
        direction = "long" if news_dir == "bullish" else "short"
    elif alignment == "neutral_ta":
        direction = "long" if news_dir == "bullish" else "short" if news_dir == "bearish" else "watch"
    elif alignment == "conflicting":
        # News conviction vs TA — go with stronger signal
        if abs(news_score) > abs(ta_score):
            direction = "long" if news_dir == "bullish" else "short"
        else:
            direction = "long" if ta_dir == "bullish" else "short"
    else:
        direction = "watch"

    # Conviction
    abs_score = abs(composite)
    if alignment == "confirmed" and abs_score > 40:
        conviction = "high"
    elif alignment == "confirmed" and abs_score > 20:
        conviction = "medium"
    elif alignment == "conflicting":
        conviction = "low"
    elif abs_score > 30:
        conviction = "medium"
    else:
        conviction = "low"

    # Action
    notes = []
    if alignment == "confirmed" and conviction in ("high", "medium"):
        action = "enter"
        notes.append("News + TA aligned — high probability setup")
    elif alignment == "confirmed":
        action = "watch"
        notes.append("Aligned but weak signals — wait for confirmation")
    elif alignment == "conflicting":
        action = "avoid"
        notes.append(f"NEWS says {news_dir} but TA says {ta_dir} — conflicting signals")
    elif alignment == "neutral_ta":
        if abs(news_score) > 25:
            action = "watch"
            notes.append("Strong news catalyst but no TA confirmation yet — watch for entry")
        else:
            action = "watch"
            notes.append("Weak news signal, no TA confirmation")
    else:
        action = "watch"

    # Risk/reward from TA
    entry = ta_sig.get("entry_zone")
    stop_loss = ta_sig.get("stop_loss")
    target = ta_sig.get("target")
    risk_reward = ta_sig.get("risk_reward")

    # Adjust R:R note
    if risk_reward and risk_reward < 1.0:
        notes.append(f"⚠️ R:R below 1.0 ({risk_reward}) — unfavorable risk/reward")
    elif risk_reward and risk_reward >= 2.0:
        notes.append(f"✅ Strong R:R ({risk_reward})")

    # Volume confirmation
    vol_ratio = ta_ind.get("volume_ratio", 1.0)
    if vol_ratio > 1.5:
        notes.append(f"Volume {vol_ratio}x average — institutional activity")
    elif vol_ratio < 0.5:
        notes.append("Low volume — weak conviction in price move")

    # Note any learned weight adjustments being applied
    ticker_adj = w.get("ticker_adjustments", {}).get(ticker)
    if ticker_adj and ticker_adj != 1.0:
        direction_word = "boosted" if ticker_adj > 1.0 else "reduced"
        notes.append(f"🧠 Learned: {ticker} score {direction_word} {ticker_adj:.2f}x from past performance")

    topic_adjs = []
    for topic in news_event.get("topics", []):
        adj = w.get("news_category_adjustments", {}).get(topic)
        if adj and adj != 1.0:
            topic_adjs.append(f"{topic} {adj:.2f}x")
    if topic_adjs:
        notes.append(f"🧠 Learned topic weights: {', '.join(topic_adjs)}")

    # Optional LLM correlation reasoning
    if use_llm:
        try:
            from llm_hooks import assess_correlation
            llm_reasoning = assess_correlation(news_event, ta_sig, ticker, ta_ind)
            if llm_reasoning:
                score_adj = llm_reasoning.get("score_adjustment", 0)
                composite += score_adj
                if llm_reasoning.get("recommendation") == "avoid":
                    action = "avoid"
                    notes.append(f"🧠 LLM: {llm_reasoning.get('alignment_assessment', '')[:100]}")
                elif llm_reasoning.get("recommendation") == "wait_for_pullback":
                    action = "watch"
                    notes.append(f"🧠 LLM suggests wait for pullback: {llm_reasoning.get('timing', '')[:80]}")
                if score_adj != 0:
                    notes.append(f"🧠 LLM score adjustment: {score_adj:+.0f}")
                for risk in llm_reasoning.get("risk_factors", [])[:2]:
                    notes.append(f"⚠️ {risk}")
        except (ImportError, Exception):
            pass

    return TradeSignal(
        ticker=ticker,
        direction=direction,
        conviction=conviction,
        score=round(composite, 2),
        news_catalyst=news_event.get("headline", "")[:100],
        news_impact=news_event.get("impact_score", 0),
        news_confidence=news_event.get("confidence", "low"),
        news_timeframe=news_event.get("timeframe", "unknown"),
        ta_signal=ta_dir,
        ta_strength=ta_sig.get("strength", "weak"),
        ta_reasons=ta_sig.get("reasons", []),
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        risk_reward=risk_reward,
        alignment=alignment,
        action=action,
        notes=notes,
    )


# ── Pipeline Runner ─────────────────────────────────────────

def run_pipeline(sources: str = "all", limit: int = 10, watchlist: str = None,
                 dry_run: bool = False, topic_filter: str = None, deep: bool = False) -> dict:
    """Run the full correlation pipeline."""

    timestamp = datetime.now().isoformat()

    # ── Step 1: Fetch news ──
    print("\n📡 STEP 1: Fetching news from all sources...")
    headlines = []

    if sources in ("rss", "all"):
        rss = fetch_rss_headlines(limit, extract_body=deep)
        bodies = sum(1 for r in rss if r.get("body"))
        print(f"   RSS: {len(rss)} headlines" + (f" ({bodies} with full text)" if bodies else ""))
        headlines.extend(rss)

    if sources in ("finnhub", "all"):
        fh = fetch_finnhub_news(limit)
        print(f"   Finnhub: {len(fh)} headlines")
        headlines.extend(fh)

    if sources in ("alpaca", "all"):
        alp = fetch_alpaca_news(limit)
        print(f"   Alpaca: {len(alp)} headlines")
        headlines.extend(alp)

    if sources in ("twitter", "all"):
        tw = fetch_twitter_trending(limit)
        print(f"   Twitter: {len(tw)} headlines")
        headlines.extend(tw)

    # Deduplicate
    seen = set()
    unique = []
    for h in headlines:
        key = h.get("headline", "")[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    headlines = unique
    print(f"   Total unique: {len(headlines)}")

    # Reduce headline volume before the expensive LLM pass.
    if len(headlines) > MAX_HEADLINES_FOR_LLM:
        print(f"   Limiting LLM news analysis to top {MAX_HEADLINES_FOR_LLM} headlines")
        headlines = sorted(headlines, key=_headline_priority, reverse=True)[:MAX_HEADLINES_FOR_LLM]

    # ── Step 2: LLM analysis ──
    print("\n🧠 STEP 2: LLM news analysis...")

    if dry_run:
        print("   [DRY RUN] Skipping LLM — using topic-based heuristics")
        # Simple keyword-based fallback when LLM unavailable
        analysis = generate_heuristic_analysis(headlines)
    else:
        analysis = analyze_with_llm(headlines)

    if analysis.get("parse_error"):
        print(f"   ⚠️  LLM unavailable, falling back to heuristic analysis")
        analysis = generate_heuristic_analysis(headlines)

    # ── Step 3: Identify affected tickers ──
    print("\n🎯 STEP 3: Identifying affected tickers...")

    events = analysis.get("events", [])
    aggregated_events = aggregate_events_by_ticker(events)
    affected_tickers = set()

    for event in events:
        tickers = event.get("affected_tickers", [])
        affected_tickers.update(tickers)

    # Also add watchlist tickers
    if watchlist and watchlist in WATCHLISTS:
        affected_tickers.update(WATCHLISTS[watchlist])

    # If no tickers from LLM, use topic-based defaults
    if not affected_tickers:
        for event in events:
            topics = event.get("topics", [])
            for topic in topics:
                if topic in TOPIC_ASSETS:
                    affected_tickers.update(TOPIC_ASSETS[topic]["tickers"])

    # Fallback: default watchlist
    if not affected_tickers:
        affected_tickers = set(WATCHLISTS.get(watchlist or "default", WATCHLISTS["default"]))

    # Prefer the strongest aggregated catalysts if the event graph fans out to too many symbols.
    if aggregated_events:
        ranked_tickers = sorted(
            aggregated_events.keys(),
            key=lambda ticker: (
                abs(float(aggregated_events[ticker].get("total_impact_score", 0) or 0)),
                int(aggregated_events[ticker].get("headline_count", 0) or 0),
            ),
            reverse=True,
        )
        # Keep explicit watchlist additions even if they rank lower.
        watchlist_set = set(WATCHLISTS.get(watchlist, [])) if watchlist and watchlist in WATCHLISTS else set()
        merged_ranked = []
        for ticker in ranked_tickers + sorted(watchlist_set):
            if ticker in affected_tickers and ticker not in merged_ranked:
                merged_ranked.append(ticker)
        affected_tickers = merged_ranked
    else:
        affected_tickers = sorted(affected_tickers)

    if len(affected_tickers) > MAX_TICKERS_TO_ANALYZE:
        print(f"   Limiting TA universe to top {MAX_TICKERS_TO_ANALYZE} tickers by catalyst strength")
        affected_tickers = affected_tickers[:MAX_TICKERS_TO_ANALYZE]

    print(f"   Tickers to analyze: {', '.join(affected_tickers)}")

    # ── Step 4: Run TA on affected tickers ──
    print(f"\n📊 STEP 4: Running TA on {len(affected_tickers)} tickers...")

    ta_results = {}
    ta_errors = []
    for ticker in affected_tickers:
        try:
            ta_results[ticker] = analyze_symbol(ticker)
            status = ta_results[ticker]["signal"]["signal_type"]
            print(f"   ✅ {ticker}: {status}")
        except Exception as e:
            ta_errors.append(ticker)
            print(f"   ❌ {ticker}: {e}")

    # ── Step 5: Correlate news + TA ──
    print(f"\n🔗 STEP 5: Correlating news signals with TA...")

    trade_signals = []
    print(f"   Aggregated into {len(aggregated_events)} unique ticker catalyst(s)")

    # First pass: fast deterministic correlation for all tickers.
    base_signals = []
    for ticker, merged_event in aggregated_events.items():
        if ticker in ta_results:
            signal = correlate_signals(merged_event, ta_results[ticker], ticker, use_llm=False)
            base_signals.append(signal)

    # Optional second pass: only apply expensive LLM reasoning to the top-ranked candidates.
    if dry_run or not base_signals or CORRELATION_LLM_MAX_TICKERS <= 0:
        trade_signals = base_signals
    else:
        ranked = sorted(base_signals, key=lambda s: abs(s.score), reverse=True)
        max_llm = min(CORRELATION_LLM_MAX_TICKERS, len(ranked))
        selected = {signal.ticker for signal in ranked[:max_llm]}
        print(f"   LLM deep reasoning on top {max_llm}/{len(ranked)} ticker(s)")

        llm_overrides = {}
        for signal in ranked[:max_llm]:
            try:
                llm_overrides[signal.ticker] = correlate_signals(
                    aggregated_events[signal.ticker],
                    ta_results[signal.ticker],
                    signal.ticker,
                    True,
                )
            except Exception:
                pass

        trade_signals = [llm_overrides.get(signal.ticker, signal) if signal.ticker in selected else signal for signal in base_signals]

    trade_signals = sorted(trade_signals, key=lambda s: abs(s.score), reverse=True)

    # ── Build result ──
    return {
        "timestamp": timestamp,
        "headlines_analyzed": len(headlines),
        "events_detected": len(events),
        "tickers_analyzed": len(ta_results),
        "market_regime": analysis.get("market_regime", "unknown"),
        "regime_reasoning": analysis.get("regime_reasoning", ""),
        "trade_signals": [asdict(s) for s in trade_signals],
        "ta_errors": ta_errors,
    }


def generate_heuristic_analysis(headlines: list[dict]) -> dict:
    """Fallback analysis when LLM is unavailable. Uses keyword matching."""

    KEYWORD_MAP = {
        "iran": {"topics": ["iran", "geopolitics"], "tickers": ["XOM", "CVX", "OXY", "XLE", "USO", "LMT", "RTX", "GLD", "SPY"]},
        "ceasefire": {"topics": ["iran", "geopolitics"], "tickers": ["XOM", "CVX", "XLE", "USO", "SPY", "GLD"]},
        "hormuz": {"topics": ["iran", "geopolitics"], "tickers": ["XOM", "CVX", "XLE", "USO"]},
        "tariff": {"topics": ["economy", "politics"], "tickers": ["SPY", "QQQ", "IWM", "XLI"]},
        "fed": {"topics": ["economy", "finance"], "tickers": ["SPY", "QQQ", "TLT", "XLF", "JPM", "GS"]},
        "interest rate": {"topics": ["economy"], "tickers": ["SPY", "TLT", "XLF", "XLU"]},
        "inflation": {"topics": ["economy"], "tickers": ["SPY", "TLT", "GLD", "XLP"]},
        "nato": {"topics": ["geopolitics"], "tickers": ["LMT", "RTX", "NOC", "GD", "SPY"]},
        "sanctions": {"topics": ["geopolitics", "iran"], "tickers": ["XOM", "CVX", "XLE", "SPY"]},
        "trump": {"topics": ["politics"], "tickers": ["SPY", "QQQ", "DIA"]},
        "oil": {"topics": ["finance", "iran"], "tickers": ["XOM", "CVX", "OXY", "XLE", "USO"]},
        "bitcoin": {"topics": ["finance"], "tickers": ["BTC-USD"]},
        "crypto": {"topics": ["finance"], "tickers": ["BTC-USD", "ETH-USD"]},
        "defense": {"topics": ["geopolitics"], "tickers": ["LMT", "RTX", "NOC", "GD"]},
        "nuclear": {"topics": ["iran", "geopolitics"], "tickers": ["XOM", "LMT", "GLD", "SPY", "VXX"]},
        "recession": {"topics": ["economy"], "tickers": ["SPY", "QQQ", "TLT", "XLP", "XLU", "GLD"]},
        "gdp": {"topics": ["economy"], "tickers": ["SPY", "QQQ", "IWM"]},
        "supreme court": {"topics": ["politics"], "tickers": ["SPY"]},
        "spacex": {"topics": ["finance"], "tickers": ["SPY"]},
    }

    # Sentiment keywords
    BULLISH_WORDS = ["ceasefire", "peace", "deal", "agreement", "rally", "surge", "soar", "boom", "recovery", "winding down", "end of", "approval"]
    BEARISH_WORDS = ["war", "attack", "strike", "crash", "collapse", "sanctions", "escalation", "recession", "shutdown", "threat", "missile"]

    events = []
    all_tickers = set()

    for h in headlines:
        headline = h.get("headline", "").lower()
        matched_topics = set()
        matched_tickers = set()

        for keyword, mapping in KEYWORD_MAP.items():
            if keyword in headline:
                matched_topics.update(mapping["topics"])
                matched_tickers.update(mapping["tickers"])

        if not matched_topics:
            continue

        # Determine sentiment
        bull_count = sum(1 for w in BULLISH_WORDS if w in headline)
        bear_count = sum(1 for w in BEARISH_WORDS if w in headline)

        if bull_count > bear_count:
            direction = "bullish"
            impact = min(bull_count, 3)
        elif bear_count > bull_count:
            direction = "bearish"
            impact = -min(bear_count, 3)
        else:
            direction = "neutral"
            impact = 0

        all_tickers.update(matched_tickers)

        events.append({
            "headline": h.get("headline", ""),
            "impact_score": impact,
            "confidence": "medium",
            "timeframe": "short-term",
            "topics": list(matched_topics),
            "affected_tickers": list(matched_tickers),
            "direction": direction,
            "reasoning": f"Keyword-based heuristic: {'bullish' if impact > 0 else 'bearish' if impact < 0 else 'neutral'} signal from headline",
            "trade_idea": None,
        })

    # Determine regime
    bull_events = sum(1 for e in events if e["direction"] == "bullish")
    bear_events = sum(1 for e in events if e["direction"] == "bearish")
    if bull_events > bear_events + 2:
        regime = "risk-on"
    elif bear_events > bull_events + 2:
        regime = "risk-off"
    else:
        regime = "mixed"

    return {
        "market_regime": regime,
        "regime_reasoning": f"Heuristic: {bull_events} bullish, {bear_events} bearish events detected",
        "events": events,
        "top_signals": [],
    }


# ── Display ─────────────────────────────────────────────────

def print_results(results: dict):
    """Pretty-print correlation results."""

    regime = results.get("market_regime", "unknown")
    regime_icons = {"risk-on": "🟢", "risk-off": "🔴", "mixed": "🟡", "neutral": "⚪"}

    print(f"\n{'=' * 70}")
    print(f"  MARKET INTELLIGENCE REPORT — {results['timestamp'][:16]}")
    print(f"{'=' * 70}")
    print(f"\n  {regime_icons.get(regime, '❓')} Market Regime: {regime.upper()}")
    print(f"  {results.get('regime_reasoning', '')}")
    print(f"  Headlines: {results['headlines_analyzed']} | Events: {results['events_detected']} | Tickers: {results['tickers_analyzed']}")

    signals = results.get("trade_signals", [])
    if not signals:
        print("\n  No actionable signals detected.")
        return

    # Separate by action
    enter = [s for s in signals if s["action"] == "enter"]
    watch = [s for s in signals if s["action"] == "watch"]
    avoid = [s for s in signals if s["action"] == "avoid"]

    if enter:
        print(f"\n{'─' * 70}")
        print(f"  🎯 ENTER — {len(enter)} actionable trades")
        print(f"{'─' * 70}")
        for s in enter:
            dir_icon = "🟢 LONG" if s["direction"] == "long" else "🔴 SHORT"
            print(f"\n  {dir_icon} {s['ticker']}  |  Conviction: {s['conviction'].upper()}  |  Score: {s['score']:+.1f}")
            print(f"  ├─ News:  [{s['news_impact']:+d}] {s['news_catalyst']}")
            print(f"  ├─ TA:    {s['ta_signal']} ({s['ta_strength']}) — {', '.join(s['ta_reasons'][:3])}")
            print(f"  ├─ Align: {s['alignment'].upper()}")
            if s.get("entry"):
                print(f"  ├─ Entry: ${s['entry']}  |  Stop: ${s['stop_loss']}  |  Target: ${s['target']}  |  R:R: {s['risk_reward']}")
            for note in s.get("notes", []):
                print(f"  └─ {note}")

    if watch:
        print(f"\n{'─' * 70}")
        print(f"  👀 WATCH — {len(watch)} potential setups")
        print(f"{'─' * 70}")
        for s in watch[:10]:  # Limit display
            dir_icon = "🟢" if s["direction"] == "long" else "🔴" if s["direction"] == "short" else "⚪"
            print(f"  {dir_icon} {s['ticker']:6s} | {s['conviction']:6s} | Score: {s['score']:+6.1f} | {s['alignment']:12s} | {s['news_catalyst'][:50]}")

    if avoid:
        print(f"\n{'─' * 70}")
        print(f"  ⛔ AVOID — {len(avoid)} conflicting signals")
        print(f"{'─' * 70}")
        for s in avoid[:5]:
            print(f"  ⚠️  {s['ticker']:6s} | News: {s['news_impact']:+d} ({s['news_confidence']}) vs TA: {s['ta_signal']} — {s['notes'][0] if s['notes'] else ''}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {len(enter)} ENTER | {len(watch)} WATCH | {len(avoid)} AVOID")
    print(f"{'=' * 70}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Correlation Engine — News + TA → Trade Signals")
    parser.add_argument("--source", type=str, default="all", choices=["rss", "finnhub", "alpaca", "twitter", "all"])
    parser.add_argument("--watchlist", "-w", type=str, help="Focus on a watchlist (default, iran, economy, crypto)")
    parser.add_argument("--limit", type=int, default=10, help="Headlines per source")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM, use heuristic analysis")
    parser.add_argument("--deep", action="store_true", help="Extract full article bodies for richer analysis")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--topic", type=str, help="Filter by topic")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  CORRELATION ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Sources: {args.source} | Watchlist: {args.watchlist or 'auto'} | Dry run: {args.dry_run}")
    print("=" * 70)

    results = run_pipeline(
        sources=args.source,
        limit=args.limit,
        watchlist=args.watchlist,
        dry_run=args.dry_run,
        topic_filter=args.topic,
        deep=args.deep,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)

    return results


if __name__ == "__main__":
    main()
