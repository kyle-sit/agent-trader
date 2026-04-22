#!/usr/bin/env python3
"""
LLM News Analysis Layer for Market Intelligence Pipeline.

Ingests news from all sources (RSS, Finnhub, Alpaca, Twitter),
sends to LLM for interpretation, outputs structured market impact assessments.

Usage:
    python3 news_analyzer.py                     # Analyze latest news from all sources
    python3 news_analyzer.py --source rss        # Only RSS feeds
    python3 news_analyzer.py --source finnhub    # Only Finnhub news
    python3 news_analyzer.py --source alpaca     # Only Alpaca/Benzinga news
    python3 news_analyzer.py --source twitter    # Only Twitter trending
    python3 news_analyzer.py --topic iran        # Filter by topic
    python3 news_analyzer.py --json              # JSON output for pipeline
    python3 news_analyzer.py --headline "Trump announces Iran ceasefire"  # Analyze a single headline
"""

import json
import os
import re
import subprocess
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import anthropic
import finnhub
from env_utils import configure_alpaca_env, load_market_intel_env, warn_missing_credentials

# ── Config ──────────────────────────────────────────────────

BLOGWATCHER = Path.home() / "go" / "bin" / "blogwatcher"
XBIRD = Path.home() / ".local" / "bin" / "xbird"

# Load env
load_market_intel_env()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_TOKEN", "") or os.getenv("ANTHROPIC_API_KEY", "")


# ── Asset Mapping ───────────────────────────────────────────

TOPIC_ASSETS = {
    "iran": {
        "tickers": ["XOM", "CVX", "OXY", "XLE", "USO", "LMT", "RTX", "NOC", "GD", "GLD", "SPY"],
        "sectors": ["Energy", "Defense", "Gold", "Broad Market"],
        "context": "Iran conflict impacts oil prices, defense stocks, safe havens, and broad market risk appetite.",
    },
    "politics": {
        "tickers": ["SPY", "QQQ", "DIA", "IWM", "XLF", "XLV", "XLC"],
        "sectors": ["Broad Market", "Financials", "Healthcare", "Communications"],
        "context": "US political events impact regulation, fiscal policy, and sector-specific legislation.",
    },
    "finance": {
        "tickers": ["SPY", "QQQ", "XLF", "JPM", "GS", "BAC", "BTC-USD"],
        "sectors": ["Financials", "Broad Market", "Crypto"],
        "context": "Financial news impacts bank stocks, fintech, interest rate sensitive sectors, and crypto.",
    },
    "geopolitics": {
        "tickers": ["SPY", "GLD", "SLV", "LMT", "RTX", "XLE", "USO", "VXX"],
        "sectors": ["Defense", "Energy", "Precious Metals", "Volatility"],
        "context": "Geopolitical tensions increase defense spending, oil risk premium, safe haven demand, and volatility.",
    },
    "economy": {
        "tickers": ["SPY", "QQQ", "DIA", "TLT", "XLF", "XLP", "XLU", "IWM"],
        "sectors": ["Broad Market", "Bonds", "Financials", "Consumer Staples", "Utilities"],
        "context": "Economic data impacts Fed policy expectations, rate-sensitive sectors, and growth/value rotation.",
    },
}


# ── News Fetchers ───────────────────────────────────────────

def fetch_rss_headlines(limit: int = 20, extract_body: bool = False) -> list[dict]:
    """Get latest headlines from blogwatcher RSS feeds.
    If extract_body=True, also fetch full article text for top headlines.
    """
    result = subprocess.run(
        [str(BLOGWATCHER), "articles"],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{Path.home()}/go/bin:{os.environ.get('PATH', '')}"}
    )

    articles = []
    lines = result.stdout.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(r'\[(\d+)\]\s+\[new\]\s+(.+)', line)
        if match:
            article = {
                "source": "rss",
                "headline": match.group(2).strip(),
                "id": match.group(1),
            }
            # Look for URL in following lines
            for j in range(i + 1, min(i + 5, len(lines))):
                meta = lines[j].strip()
                if meta.startswith("URL:"):
                    article["url"] = meta.replace("URL:", "").strip()
                elif meta.startswith("Blog:"):
                    article["blog"] = meta.replace("Blog:", "").strip()
            articles.append(article)
        i += 1

    articles = articles[:limit]

    # Optionally extract full article bodies for richer LLM analysis
    if extract_body and articles:
        try:
            from extract_articles import resolve_google_news_url, extract_article_text_direct
            extracted = 0
            for article in articles[:10]:  # Limit body extraction to top 10 to save time
                url = article.get("url", "")
                if not url:
                    continue
                try:
                    actual_url = resolve_google_news_url(url)
                    result = extract_article_text_direct(actual_url)
                    if result and result.get("body"):
                        article["body"] = result["body"][:2000]  # Cap at 2000 chars for LLM context
                        article["body_word_count"] = len(result["body"].split())
                        extracted += 1
                except Exception:
                    continue
            if extracted > 0:
                print(f"   Extracted body text for {extracted} articles", file=sys.stderr)
        except ImportError:
            pass  # extract_articles not available

    # Optionally summarize article bodies via LLM for better analysis
    if extract_body:
        try:
            from llm_hooks import summarize_article
            for article in articles:
                if article.get("body") and len(article["body"]) > 200:
                    summary = summarize_article(article["headline"], article["body"])
                    if summary:
                        article["llm_summary"] = summary
        except (ImportError, Exception):
            pass

    return articles


def fetch_finnhub_news(limit: int = 20) -> list[dict]:
    """Get latest market news from Finnhub."""
    if not FINNHUB_KEY:
        return []

    try:
        client = finnhub.Client(api_key=FINNHUB_KEY)
        news = client.general_news('general', min_id=0)
        articles = []
        for item in news[:limit]:
            articles.append({
                "source": "finnhub",
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "datetime": item.get("datetime", ""),
                "related": item.get("related", ""),
            })
        return articles
    except Exception as e:
        print(f"⚠️  Finnhub error: {e}", file=sys.stderr)
        return []


def fetch_alpaca_news(limit: int = 20) -> list[dict]:
    """Get latest news from Alpaca/Benzinga."""
    alpaca_env = configure_alpaca_env()
    if alpaca_env["missing"]:
        warn_missing_credentials(alpaca_env["missing"], context="News analyzer / Alpaca")
        return []

    try:
        from alpaca_trade_api.rest import REST
        api = REST()
        news = api.get_news(limit=limit)
        articles = []
        for item in news:
            articles.append({
                "source": "alpaca",
                "headline": item.headline,
                "summary": getattr(item, 'summary', ''),
                "url": getattr(item, 'url', ''),
                "symbols": getattr(item, 'symbols', []),
            })
        return articles
    except Exception as e:
        print(f"⚠️  Alpaca error: {e}", file=sys.stderr)
        return []


def fetch_twitter_trending(limit: int = 10) -> list[dict]:
    """Get credibility-filtered Twitter intel using tiered system."""
    try:
        from twitter_intel import scan_twitter, format_for_news_analyzer
        data = scan_twitter(topics=None, include_trump=True, search_limit=limit)
        headlines = format_for_news_analyzer(data)
        return headlines
    except ImportError:
        pass

    # Fallback: basic trending only
    try:
        result = subprocess.run(
            [str(XBIRD), "news", "--ai-only", "--news-only", "-n", str(limit)],
            capture_output=True, text=True, timeout=30
        )
        articles = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("[AI"):
                match = re.match(r'\[AI\s*·\s*(\w+)\]\s+(.+)', line)
                if match:
                    articles.append({
                        "source": "twitter",
                        "category": match.group(1),
                        "headline": match.group(2).strip(),
                    })
        return articles
    except Exception as e:
        print(f"⚠️  Twitter error: {e}", file=sys.stderr)
        return []


# ── LLM Analysis ────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a senior market analyst at a macro hedge fund. Your job is to analyze news headlines and determine their market impact.

For each batch of news, provide a structured analysis in the following JSON format:

{{
  "market_regime": "risk-on | risk-off | mixed | neutral",
  "regime_reasoning": "Brief explanation of overall market conditions",
  "events": [
    {{
      "headline": "The original headline",
      "impact_score": -5 to +5 (negative=bearish, positive=bullish, 0=neutral),
      "confidence": "high | medium | low",
      "timeframe": "immediate | short-term | medium-term | long-term",
      "topics": ["iran", "politics", "finance", "geopolitics", "economy"],
      "affected_tickers": ["XOM", "SPY", ...],
      "affected_sectors": ["Energy", "Defense", ...],
      "direction": "bullish | bearish | neutral",
      "reasoning": "Why this impacts these assets in this direction",
      "trade_idea": "Specific actionable trade idea or null if none"
    }}
  ],
  "top_signals": [
    {{
      "ticker": "XOM",
      "direction": "bullish | bearish",
      "conviction": "high | medium | low",
      "catalyst": "Brief description of the news catalyst",
      "suggested_action": "buy | sell | watch | hedge"
    }}
  ]
}}

Rules:
- Only include events that have REAL market impact. Skip noise (celebrity news, weather, sports unless market-moving).
- impact_score: -5 = catastrophic bearish, -3 = significantly bearish, -1 = slightly bearish, 0 = neutral, +1 = slightly bullish, +3 = significantly bullish, +5 = extremely bullish
- For top_signals, only include tickers with medium or high conviction
- Be specific about WHY something is bullish or bearish
- Consider second-order effects (e.g., Iran conflict → oil up → consumer spending down → retail stocks down)
- If headlines conflict, note the tension

Analyze these headlines:

{headlines}"""


def analyze_with_llm(headlines: list[dict]) -> dict:
    """Send headlines through the centralized LLM router for market impact analysis."""
    if not ANTHROPIC_KEY:
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY in ~/.hermes/.env")

    # Format headlines for the prompt
    headline_text = ""
    for i, h in enumerate(headlines, 1):
        source = h.get("source", "unknown")
        headline = h.get("headline", "")
        summary = h.get("summary", "")
        body = h.get("body", "")
        line = f"{i}. [{source}] {headline}"
        if summary:
            line += f"\n   Summary: {summary[:300]}"
        if body:
            line += f"\n   Full article excerpt: {body[:1500]}"
        headline_text += line + "\n"

    prompt = ANALYSIS_PROMPT.format(headlines=headline_text)

    # Use centralized LLM router (handles model selection, fallbacks, and retries)
    try:
        from llm_router import llm_call
        result = llm_call("news_analysis", prompt)
        if result["success"]:
            text = result["text"]
            print(f"  ✅ Using {result['model_used']}", file=sys.stderr)
        else:
            text = None
            print(f"  ⚠️  Router failed: {result.get('reason', 'unknown')}", file=sys.stderr)
    except ImportError:
        text = None
        print(f"  ⚠️  LLM router not available", file=sys.stderr)

    if not text:
        return {"raw_response": "All LLM backends unavailable. Start Ollama or wait for Anthropic rate limit to clear.", "parse_error": True}

    # Find JSON block
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text
    return {"raw_response": text, "parse_error": True}


# ── Main ────────────────────────────────────────────────────

def print_analysis(analysis: dict):
    """Pretty-print the LLM analysis."""
    if analysis.get("parse_error"):
        print("⚠️  Could not parse structured response. Raw output:")
        print(analysis.get("raw_response", ""))
        return

    # Market regime
    regime = analysis.get("market_regime", "unknown")
    regime_icons = {"risk-on": "🟢", "risk-off": "🔴", "mixed": "🟡", "neutral": "⚪"}
    print(f"\n{regime_icons.get(regime, '❓')} MARKET REGIME: {regime.upper()}")
    print(f"   {analysis.get('regime_reasoning', '')}")

    # Events
    events = analysis.get("events", [])
    if events:
        print(f"\n📰 KEY EVENTS ({len(events)})")
        print("-" * 60)
        for e in events:
            score = e.get("impact_score", 0)
            if score > 0:
                icon = "🟢"
            elif score < 0:
                icon = "🔴"
            else:
                icon = "⚪"

            print(f"\n  {icon} [{score:+d}] {e.get('headline', '')[:80]}")
            print(f"     Direction: {e.get('direction', 'N/A')} | Confidence: {e.get('confidence', 'N/A')} | Timeframe: {e.get('timeframe', 'N/A')}")
            print(f"     Topics: {', '.join(e.get('topics', []))}")
            print(f"     Tickers: {', '.join(e.get('affected_tickers', []))}")
            print(f"     Reasoning: {e.get('reasoning', '')}")
            if e.get("trade_idea"):
                print(f"     💡 Trade: {e['trade_idea']}")

    # Top signals
    signals = analysis.get("top_signals", [])
    if signals:
        print(f"\n🎯 TOP SIGNALS ({len(signals)})")
        print("-" * 60)
        for s in signals:
            icon = "🟢" if s.get("direction") == "bullish" else "🔴"
            print(f"  {icon} {s.get('ticker', 'N/A')}: {s.get('direction', 'N/A').upper()} ({s.get('conviction', 'N/A')})")
            print(f"     Action: {s.get('suggested_action', 'N/A')} | Catalyst: {s.get('catalyst', '')}")


def main():
    parser = argparse.ArgumentParser(description="LLM News Analysis Layer")
    parser.add_argument("--source", type=str, choices=["rss", "finnhub", "alpaca", "twitter", "all"], default="all")
    parser.add_argument("--topic", type=str, help="Filter headlines by topic keyword")
    parser.add_argument("--headline", type=str, help="Analyze a single headline")
    parser.add_argument("--limit", type=int, default=15, help="Max headlines per source")
    parser.add_argument("--deep", action="store_true", help="Extract full article bodies for richer LLM analysis")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    # Single headline mode
    if args.headline:
        headlines = [{"source": "manual", "headline": args.headline}]
    else:
        headlines = []

        if not args.json:
            print("=" * 60)
            print(f"  NEWS ANALYZER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"  Fetching from: {args.source}")
            print("=" * 60)

        # Fetch from selected sources
        if args.source in ("rss", "all"):
            if not args.json:
                print("\n📡 Fetching RSS feeds...")
            rss = fetch_rss_headlines(args.limit, extract_body=args.deep)
            if not args.json:
                bodies = sum(1 for r in rss if r.get("body"))
                print(f"   Got {len(rss)} headlines" + (f" ({bodies} with full text)" if bodies else ""))
            headlines.extend(rss)

        if args.source in ("finnhub", "all"):
            if not args.json:
                print("📊 Fetching Finnhub news...")
            fh = fetch_finnhub_news(args.limit)
            if not args.json:
                print(f"   Got {len(fh)} headlines")
            headlines.extend(fh)

        if args.source in ("alpaca", "all"):
            if not args.json:
                print("📈 Fetching Alpaca/Benzinga news...")
            alp = fetch_alpaca_news(args.limit)
            if not args.json:
                print(f"   Got {len(alp)} headlines")
            headlines.extend(alp)

        if args.source in ("twitter", "all"):
            if not args.json:
                print("🐦 Fetching Twitter trending...")
            tw = fetch_twitter_trending(args.limit)
            if not args.json:
                print(f"   Got {len(tw)} headlines")
            headlines.extend(tw)

    # Filter by topic
    if args.topic:
        topic_lower = args.topic.lower()
        headlines = [h for h in headlines if topic_lower in h.get("headline", "").lower()
                     or topic_lower in h.get("summary", "").lower()
                     or topic_lower in h.get("category", "").lower()]
        if not args.json:
            print(f"\n🔍 Filtered to {len(headlines)} headlines matching '{args.topic}'")

    if not headlines:
        print("\n⚠️  No headlines found.")
        return

    # Deduplicate by headline similarity
    seen = set()
    unique = []
    for h in headlines:
        key = h.get("headline", "")[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    headlines = unique

    if not args.json:
        print(f"\n🧠 Analyzing {len(headlines)} unique headlines with configured LLM route...")
        print("-" * 60)

    # LLM analysis
    analysis = analyze_with_llm(headlines)

    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "headlines_analyzed": len(headlines),
            "sources": list(set(h.get("source", "") for h in headlines)),
            "analysis": analysis,
        }
        print(json.dumps(output, indent=2))
    else:
        print_analysis(analysis)

    # Return for pipeline use
    return analysis


if __name__ == "__main__":
    main()
