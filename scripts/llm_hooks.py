#!/usr/bin/env python3
"""
LLM Hooks — All LLM call points for the pipeline, using the centralized router.

Each function is a self-contained LLM call that can be imported by any component.
All calls go through llm_router.llm_call() which handles model selection,
fallbacks, and enable/disable logic.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_router import llm_call, extract_json


# ── 1. Article Summarization (deep mode) ───────────────────

def summarize_article(headline: str, body: str) -> str:
    """Summarize a full article body into a concise market-relevant summary."""
    prompt = f"""Summarize this news article in 2-3 sentences focused on market impact.
What happened, who's affected, and what are the financial implications?

Headline: {headline}
Article: {body[:3000]}

Reply with ONLY the summary, no preamble."""

    result = llm_call("article_summarization", prompt)
    if result["success"]:
        return result["text"].strip()
    return ""


# ── 2. TA Interpretation ───────────────────────────────────

def interpret_ta(symbol: str, indicators: dict, signal: dict, support_resistance: dict) -> str:
    """Interpret TA indicators and provide a human-readable analysis."""
    prompt = f"""You are a technical analyst. Interpret these indicators for {symbol} and provide a concise trading assessment.

Price: ${indicators['price']} ({indicators['change_pct']:+.2f}%)
EMAs: 9={indicators['ema_9']}, 21={indicators['ema_21']}, SMA50={indicators['sma_50']}, SMA200={indicators['sma_200']}
RSI: {indicators['rsi']} | StochRSI: {indicators['stoch_rsi']:.0f}
MACD: {indicators['macd']:.4f} | Signal: {indicators['macd_signal']:.4f} | Hist: {indicators['macd_histogram']:.4f}
ADX: {indicators['adx']} | ATR: {indicators['atr']} ({indicators['atr_pct']:.1f}%)
BB: Lower={indicators['bb_lower']}, Mid={indicators['bb_middle']}, Upper={indicators['bb_upper']} (Position: {indicators['bb_pct']:.2f})
Volume: {indicators['volume_ratio']}x 20-day avg | OBV slope: {indicators['obv_slope']:.2f}%
Support: S1={support_resistance['support_1']}, S2={support_resistance['support_2']}
Resistance: R1={support_resistance['resistance_1']}, R2={support_resistance['resistance_2']}

Current signal: {signal['signal_type']} ({signal['strength']})
Reasons: {', '.join(signal['reasons'][:5])}

Provide in JSON format:
{{
    "assessment": "2-3 sentence technical assessment",
    "key_levels": "Most important price levels to watch",
    "pattern": "Any chart pattern you identify (e.g., 'bull flag', 'head and shoulders', 'consolidation')",
    "risk": "Key risk for this setup",
    "conviction_modifier": -0.2 to +0.2 (negative = reduce conviction, positive = increase)
}}"""

    result = llm_call("ta_interpretation", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return parsed
    return None


# ── 3. Correlation Reasoning ───────────────────────────────

def assess_correlation(news_event: dict, ta_signal: dict, ticker: str, indicators: dict) -> dict:
    """Deep reasoning about aggregated news/TA alignment for a specific ticker."""
    headlines = news_event.get("headlines") or [news_event.get("headline", "N/A")]
    source_events = news_event.get("source_events") or []
    headlines_block = "\n".join(f"  - {h}" for h in headlines[:5])
    event_count = news_event.get("headline_count", len(source_events) or 1)

    prompt = f"""You are a senior macro trader. Assess this potential trade setup.

TICKER: {ticker}
CURRENT PRICE: ${indicators.get('price', 'N/A')}

MERGED NEWS CATALYST ({event_count} event(s)):
  Primary catalyst: {news_event.get('headline', 'N/A')}
  Headlines:
{headlines_block}
  Aggregate direction: {news_event.get('direction', 'N/A')}
  Aggregate impact: {news_event.get('impact_score', 'N/A')}/5
  Confidence: {news_event.get('confidence', 'N/A')}
  Timeframe: {news_event.get('timeframe', 'N/A')}
  Topics: {', '.join(news_event.get('topics', []))}

TA SIGNAL:
  Direction: {ta_signal.get('signal_type', 'N/A')} ({ta_signal.get('strength', 'N/A')})
  RSI: {indicators.get('rsi', 'N/A')}
  Trend: ADX {indicators.get('adx', 'N/A')}
  Reasons: {', '.join(ta_signal.get('reasons', [])[:4])}

Reply in JSON:
{{
    "alignment_assessment": "Does the merged news picture support the TA setup? Why or why not?",
    "second_order_effects": "What secondary market effects could this merged catalyst trigger for this ticker?",
    "timing": "Is this the right time to enter, or should we wait?",
    "risk_factors": ["Risk 1", "Risk 2"],
    "score_adjustment": -20 to +20 (adjust the composite score based on your deeper analysis),
    "recommendation": "enter_now | wait_for_pullback | wait_for_confirmation | avoid"
}}"""

    result = llm_call("correlation_reasoning", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return parsed
    return None


# ── 4. Trade Validation ────────────────────────────────────

def _normalize_trade_validation(parsed: dict) -> dict:
    """Backfill missing fields in trade validation responses."""
    if not parsed:
        return None
    if "action" not in parsed:
        parsed["action"] = "open_new" if parsed.get("approved", True) else "skip"
    if "size_fraction" not in parsed:
        suggestion = (parsed.get("position_size_suggestion") or "full").lower()
        parsed["size_fraction"] = {"full": 1.0, "half": 0.5, "quarter": 0.25, "none": 0.0}.get(suggestion, 1.0)
    if "reduce_fraction" not in parsed:
        parsed["reduce_fraction"] = 0.5 if parsed.get("action") == "reduce_position" else 0.0
    if "cancel_existing_orders" not in parsed:
        parsed["cancel_existing_orders"] = False
    if "warnings" not in parsed or not isinstance(parsed.get("warnings"), list):
        parsed["warnings"] = []
    return parsed


def validate_trade(signal: dict, portfolio_context: str = "") -> dict:
    """Final position-aware execution decision before trading."""
    prompt = f"""You are a risk-conscious portfolio manager deciding what execution action to take.

TRADE SIGNAL:
  Ticker: {signal.get('ticker', 'N/A')}
  Direction: {signal.get('direction', 'N/A')}
  Conviction: {signal.get('conviction', 'N/A')}
  Score: {signal.get('score', 'N/A')}
  Entry: ${signal.get('entry', 'N/A')}
  Stop: ${signal.get('stop_loss', 'N/A')}
  Target: ${signal.get('target', 'N/A')}
  R:R: {signal.get('risk_reward', 'N/A')}
  News catalyst: {signal.get('news_catalyst', 'N/A')}
  TA: {signal.get('ta_signal', 'N/A')} ({signal.get('ta_strength', 'N/A')})
  Alignment: {signal.get('alignment', 'N/A')}

{f'PORTFOLIO CONTEXT JSON: {portfolio_context}' if portfolio_context else ''}

Your job:
- Consider current position, direction, matching open orders, portfolio concentration, sector/theme exposure, and hard rules.
- Hard rules are deterministic guardrails and should not be overridden.
- Choose the best action from this enum only:
  - open_new
  - add_to_position
  - hold_existing
  - reduce_position
  - reverse_position
  - skip
- Use add_to_position only if the existing position is in the SAME direction and adding is justified.
- Use reverse_position only if an existing position is in the OPPOSITE direction and the new signal is materially stronger.
- Use reduce_position if an existing position should be trimmed rather than reversed.
- Use hold_existing if we already have the right exposure and should not trade now.
- Use skip if the trade should not be executed.

Reply in JSON only:
{{
    "approved": true/false,
    "action": "open_new | add_to_position | hold_existing | reduce_position | reverse_position | skip",
    "reason": "1-2 sentence explanation",
    "size_fraction": 0.0 | 0.25 | 0.5 | 1.0,
    "position_size_suggestion": "full | half | quarter | none",
    "reduce_fraction": 0.0 | 0.25 | 0.5 | 1.0,
    "cancel_existing_orders": true/false,
    "stop_adjustment": null or suggested stop price,
    "target_adjustment": null or suggested target price,
    "warnings": ["Any concerns to note"]
}}

Guidance:
- If action is hold_existing, reduce_position, or skip, approved may still be false.
- If no trade should be placed, set size_fraction to 0.0.
- Prefer smaller adds when concentration is elevated.
- Be conservative."""

    result = llm_call("trade_validation", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return _normalize_trade_validation(parsed)
    return None


def validate_trades_batch(candidates: list[dict]) -> dict[str, dict]:
    """Batch trade validation for multiple candidate trades in one LLM call."""
    if not candidates:
        return {}

    compact_candidates = []
    for c in candidates:
        signal = c.get("signal", {})
        compact_candidates.append({
            "ticker": signal.get("ticker"),
            "direction": signal.get("direction"),
            "conviction": signal.get("conviction"),
            "score": signal.get("score"),
            "entry": signal.get("entry"),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target"),
            "risk_reward": signal.get("risk_reward"),
            "news_catalyst": signal.get("news_catalyst"),
            "ta_signal": signal.get("ta_signal"),
            "ta_strength": signal.get("ta_strength"),
            "alignment": signal.get("alignment"),
            "portfolio_context": c.get("portfolio_context"),
        })

    prompt = f"""You are a risk-conscious portfolio manager deciding execution actions for a batch of trade candidates.

For each candidate, consider:
- the trade signal
- current position and direction
- matching open orders
- portfolio concentration
- sector/theme exposure
- hard deterministic rules in the provided context

Important:
- Hard rules are guardrails and should not be overridden.
- Return one decision per ticker.
- Choose only from: open_new, add_to_position, hold_existing, reduce_position, reverse_position, skip
- Be conservative.

CANDIDATES JSON:
{json.dumps(compact_candidates, ensure_ascii=False)}

Reply in JSON only, using this shape:
{{
  "decisions": {{
    "TICKER": {{
      "approved": true,
      "action": "open_new | add_to_position | hold_existing | reduce_position | reverse_position | skip",
      "reason": "brief explanation",
      "size_fraction": 0.0 | 0.25 | 0.5 | 1.0,
      "position_size_suggestion": "full | half | quarter | none",
      "reduce_fraction": 0.0 | 0.25 | 0.5 | 1.0,
      "cancel_existing_orders": false,
      "stop_adjustment": null,
      "target_adjustment": null,
      "warnings": []
    }}
  }}
}}"""

    result = llm_call("trade_validation", prompt)
    if not result["success"]:
        return {}
    parsed = extract_json(result["text"])
    if not parsed:
        return {}
    decisions = parsed.get("decisions", parsed if isinstance(parsed, dict) else {})
    normalized = {}
    if isinstance(decisions, dict):
        for ticker, decision in decisions.items():
            if isinstance(decision, dict):
                normalized[str(ticker)] = _normalize_trade_validation(decision)
    return normalized


# ── 5. Exit Analysis ───────────────────────────────────────

def analyze_exit(symbol: str, entry_context: dict, current_state: dict, recent_news: str = "") -> dict:
    """Analyze whether to exit a position based on catalyst invalidation."""
    prompt = f"""You are managing an open position. Should we exit?

POSITION:
  Symbol: {symbol}
  Side: {entry_context.get('side', 'long')}
  Entry: ${entry_context.get('entry', 'N/A')}
  Current: ${current_state.get('current_price', 'N/A')}
  P&L: {current_state.get('pl_pct', 'N/A')}%
  Hold: {current_state.get('hold_days', 'N/A')} days

ORIGINAL CATALYST:
  {entry_context.get('news_catalyst', 'N/A')}

CURRENT TA:
  Signal: {current_state.get('ta_signal', 'N/A')}
  RSI: {current_state.get('rsi', 'N/A')}

{f'RECENT NEWS: {recent_news[:500]}' if recent_news else ''}

Reply in JSON:
{{
    "catalyst_still_valid": true/false,
    "exit_recommendation": "hold | partial_exit | full_exit",
    "reasoning": "Why hold or exit",
    "new_stop": null or suggested new stop price,
    "new_target": null or suggested new target price
}}"""

    result = llm_call("exit_analysis", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return parsed
    return None


# ── 6. Risk Commentary ─────────────────────────────────────

def recommend_rebalance_actions(portfolio_data: dict, violations: list) -> dict:
    """Generate a structured rebalance plan to move the portfolio back toward hard limits."""
    positions = portfolio_data.get("positions", [])
    position_rows = []
    for p in positions:
        position_rows.append({
            "symbol": p.get("symbol"),
            "side": p.get("side"),
            "pct_of_portfolio": p.get("pct_of_portfolio"),
            "sector": p.get("sector"),
            "theme": p.get("theme"),
            "asset_type": p.get("asset_type"),
            "pl_pct": p.get("pl_pct"),
            "max_position_pct": p.get("max_position_pct"),
        })

    prompt = f"""You are a risk-conscious portfolio rebalancing manager.

Goal:
- Recommend the smallest sensible set of trim/close actions needed to move the portfolio back toward its hard portfolio constraints.
- Prioritize reducing the biggest and weakest contributors to violations.
- Prefer trims before full closes unless a position is clearly a poor candidate to keep.
- Do NOT propose opening any new positions.

HARD RULES:
{json.dumps(portfolio_data.get('rules', {}), indent=2)}

CURRENT PORTFOLIO SUMMARY:
{json.dumps({
    'portfolio_value': portfolio_data.get('portfolio_value'),
    'cash_pct': portfolio_data.get('cash_pct'),
    'total_exposure_pct': portfolio_data.get('total_exposure_pct'),
    'top5_exposure_pct': portfolio_data.get('top5_exposure_pct'),
    'num_positions': portfolio_data.get('num_positions'),
    'sector_exposure': portfolio_data.get('sector_exposure', {}),
    'theme_exposure': portfolio_data.get('theme_exposure', {}),
}, indent=2)}

POSITIONS:
{json.dumps(position_rows, indent=2)}

VIOLATIONS:
{json.dumps(violations, indent=2)}

Return JSON only in this exact shape:
{{
  "summary": "brief rebalance thesis",
  "actions": [
    {{
      "symbol": "XOM",
      "action": "reduce | close | hold",
      "reduce_fraction": 0.0 | 0.25 | 0.5 | 0.75 | 1.0,
      "cancel_open_orders": true/false,
      "reason": "brief reason tied to the violations"
    }}
  ]
}}

Rules for the output:
- Include only positions that should be acted on.
- Keep the action list concise; prefer the smallest set that meaningfully improves violations.
- If no action is needed, return an empty actions list.
- For action=close, set reduce_fraction to 1.0.
- For action=hold, reduce_fraction should be 0.0."""

    result = llm_call("risk_commentary", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return parsed
    return {"summary": "No rebalance plan available", "actions": []}


def analyze_portfolio_risk(portfolio_data: dict, violations: list) -> dict:
    """Provide contextual risk analysis and rebalancing suggestions."""
    positions_str = "\n".join(
        f"  {p['symbol']:8s} ({p['sector']:15s}): {p['pct_of_portfolio']:5.1f}% | P&L: {p['pl_pct']:+.2f}%"
        for p in portfolio_data.get("positions", [])
    )
    violations_str = "\n".join(v.get("message", "") for v in violations) if violations else "None"

    prompt = f"""You are a portfolio risk manager. Analyze this portfolio and suggest improvements.

PORTFOLIO:
  Value: ${portfolio_data.get('portfolio_value', 0):,.2f}
  Cash: {portfolio_data.get('cash_pct', 0):.1f}%
  Exposure: {portfolio_data.get('total_exposure_pct', 0):.1f}%
  Sectors: {portfolio_data.get('num_sectors', 0)}

POSITIONS:
{positions_str}

SECTOR EXPOSURE:
{json.dumps(portfolio_data.get('sector_exposure', {}), indent=2)}

RISK VIOLATIONS:
{violations_str}

Reply in JSON:
{{
    "risk_level": "low | moderate | elevated | high | critical",
    "assessment": "2-3 sentence risk assessment",
    "concentration_risk": "Analysis of sector/position concentration",
    "rebalance_suggestions": ["Suggestion 1", "Suggestion 2"],
    "hedging_ideas": ["Hedge idea 1"],
    "position_adjustments": [
        {{"symbol": "XOM", "action": "reduce | hold | increase", "reason": "..."}}
    ]
}}"""

    result = llm_call("risk_commentary", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return parsed
    return None


# ── 7. Learning Synthesis ──────────────────────────────────

def synthesize_learnings(patterns: dict, current_weights: dict) -> dict:
    """Synthesize performance data into actionable strategy adjustments."""
    prompt = f"""You are a quantitative strategy analyst. Analyze these trading performance patterns and suggest concrete strategy adjustments.

PERFORMANCE:
  Total trades: {patterns.get('total_trades', 0)}
  Win rate: {patterns.get('win_rate', 0)}%
  Avg P&L: {patterns.get('avg_pl_pct', 0):+.2f}%

BY CONVICTION:
{json.dumps(patterns.get('by_conviction', {}), indent=2)}

BY ALIGNMENT TYPE:
{json.dumps(patterns.get('by_alignment', {}), indent=2)}

BY NEWS TOPIC:
{json.dumps(patterns.get('by_topic', {}), indent=2)}

BY TA PATTERN:
{json.dumps(dict(list(patterns.get('by_ta_reason', {}).items())[:10]), indent=2)}

CURRENT LEARNINGS:
{chr(10).join(patterns.get('learnings', []))}

CURRENT SCORING WEIGHTS:
  Confirmed bonus: {current_weights.get('confirmed_bonus', 1.3)}
  Conflicting penalty: {current_weights.get('conflicting_penalty', 0.5)}
  Confidence multipliers: {json.dumps(current_weights.get('confidence_multipliers', {}))}

Reply in JSON:
{{
    "strategy_assessment": "Overall assessment of strategy performance",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "weight_recommendations": {{
        "confirmed_bonus": suggested_value,
        "conflicting_penalty": suggested_value,
        "news_category_adjustments": {{"topic": multiplier}},
        "ta_pattern_adjustments": {{"pattern": multiplier}}
    }},
    "behavioral_adjustments": ["What we should do more of", "What we should stop doing"],
    "next_actions": ["Specific action 1", "Specific action 2"]
}}"""

    result = llm_call("learning_synthesis", prompt)
    if result["success"]:
        parsed = extract_json(result["text"])
        if parsed:
            return parsed
    return None
