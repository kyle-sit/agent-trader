#!/usr/bin/env python3
"""
Tiered Twitter/X Intelligence Layer.

Monitors Twitter/X with a credibility-weighted system:
  Tier 1: Trusted sources (verified journalists, officials, institutions) — full weight
  Tier 2: Credible accounts (high followers, aged accounts) — medium weight
  Tier 3: General public — ignored (noise)

Usage:
    python3 twitter_intel.py                         # Tier 1 direct account scan + GPT relevance filter
    python3 twitter_intel.py --accounts              # List monitored Tier 1 accounts
    python3 twitter_intel.py --json                  # JSON output for pipeline
"""

import json
import os
import re
import subprocess
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

XBIRD = Path.home() / ".local" / "bin" / "xbird"
MAX_TWEET_AGE_MINUTES = int(os.getenv("TWITTER_MAX_AGE_MINUTES", "30"))

# ── Tier 1: Trusted Sources ────────────────────────────────
# These accounts are treated as factual. Full weight in analysis.

TIER1_ACCOUNTS = {
    # ── Wire Services & Major News ──
    "Reuters": "Reuters",
    "AP": "AP",
    "AFP": "AFP",

    # ── Financial News ──
    "Bloomberg": "business",
    "CNBC": "CNBC",
    "WSJ": "WSJ",
    "FT": "FT",
    "MarketWatch": "MarketWatch",

    # ── Broadcast / Print ──
    "BBCWorld": "BBCWorld",
    "BBCBreaking": "BBCBreaking",
    "CNN": "CNN",
    "NPR": "NPR",
    "PBS": "NewsHour",
    "NYT": "nytimes",
    "WashingtonPost": "washingtonpost",
    # Politico / The Hill removed pending handle verification (old entries were invalid)
    "TheEconomist": "TheEconomist",
    "AJEnglish": "AJEnglish",

    # ── US Government / Officials ──
    "POTUS": "POTUS",
    "WhiteHouse": "WhiteHouse",
    "realDonaldTrump": "realDonaldTrump",
    "StateDept": "StateDept",
    # Pentagon press secretary / SecDef removed pending handle verification (old entries were invalid)
    "VP": "VP",
    "USTreasury": "USTreasury",

    # ── Market Analysis ──

    "KobeissiLetter": "KobeissiLetter",
}

# ── Topic-Specific Search Queries ──────────────────────────

TOPIC_QUERIES = {
    "iran": [
        "Iran war OR ceasefire OR Hormuz OR IRGC OR Tehran",
        "Iran oil OR sanctions OR nuclear",
    ],
    "politics": [
        "Congress OR Senate OR \"Supreme Court\" OR \"executive order\"",
        "POTUS OR \"White House\" policy OR legislation",
    ],
    "finance": [
        "\"stock market\" OR S&P500 OR NASDAQ OR \"Wall Street\"",
        "Bitcoin OR crypto OR \"interest rate\" OR Fed",
    ],
    "geopolitics": [
        "NATO OR sanctions OR \"military strike\" OR ceasefire",
        "China OR Russia OR Ukraine OR Taiwan",
    ],
    "economy": [
        "inflation OR GDP OR unemployment OR recession",
        "tariff OR \"trade war\" OR \"consumer spending\"",
    ],
}


# ── Tier 2 Credibility Check ──────────────────────────────

def check_tier2_credibility(tweet: dict) -> tuple[bool, str]:
    """
    Check if a tweet from an unknown account meets Tier 2 criteria.
    Returns (is_credible, reason).

    Tier 2 criteria:
    - 50k+ followers
    - Account verified or notable
    - Like count suggests engagement (not bot)
    """
    author = tweet.get("author", {})
    username = author.get("username", "")

    # If it's a Tier 1 account, skip
    if username in TIER1_ACCOUNTS.values():
        return True, "tier1"

    likes = tweet.get("likeCount", 0)
    retweets = tweet.get("retweetCount", 0)
    replies = tweet.get("replyCount", 0)
    engagement = likes + retweets + replies

    # High engagement suggests credible/noteworthy content
    if engagement > 1000:
        return True, f"high_engagement ({engagement:,})"

    # Check if it's a quote or retweet of a Tier 1 source
    text = tweet.get("text", "")
    for t1_handle in TIER1_ACCOUNTS.values():
        if f"@{t1_handle}" in text:
            return True, f"references_tier1 (@{t1_handle})"

    return False, "insufficient_credibility"


def parse_tweet_timestamp(tweet: dict) -> datetime | None:
    """Parse tweet timestamps from xbird JSON payloads."""
    raw = (
        tweet.get("createdAt")
        or tweet.get("created_at")
        or tweet.get("timestamp")
        or tweet.get("published_at")
    )
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()
    formats = [
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def filter_recent_tweets(tweets: list[dict], max_age_minutes: int = MAX_TWEET_AGE_MINUTES) -> list[dict]:
    """Keep only tweets within the freshness window and annotate age metadata."""
    recent = []
    now = datetime.now(timezone.utc)
    for tweet in tweets:
        dt = parse_tweet_timestamp(tweet)
        if dt is None:
            continue
        age_minutes = (now - dt).total_seconds() / 60
        if age_minutes <= max_age_minutes:
            tweet["_created_at_iso"] = dt.isoformat()
            tweet["_age_minutes"] = round(age_minutes, 1)
            recent.append(tweet)
    return recent


def is_reply_tweet(tweet: dict) -> bool:
    """Detect reply tweets from xbird payloads."""
    if tweet.get("inReplyToStatusId") or tweet.get("in_reply_to_status_id"):
        return True
    text = str(tweet.get("text", "") or "").lstrip()
    return text.startswith("@")


def exclude_reply_tweets(tweets: list[dict]) -> list[dict]:
    """Remove reply tweets from a tweet list."""
    return [tweet for tweet in tweets if not is_reply_tweet(tweet)]


# ── Fetchers ────────────────────────────────────────────────

def fetch_tier1_tweets(accounts: list[str] = None, limit: int = 5) -> list[dict]:
    """Fetch latest tweets from Tier 1 accounts."""
    if accounts is None:
        accounts = list(TIER1_ACCOUNTS.values())

    all_tweets = []

    for handle in accounts:
        try:
            result = subprocess.run(
                [str(XBIRD), "user-tweets", f"@{handle}", "-n", str(limit), "--json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                tweets = json.loads(result.stdout)
                for t in tweets:
                    t["_tier"] = 1
                    t["_source_handle"] = handle
                all_tweets.extend(tweets)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            continue

    return filter_recent_tweets(exclude_reply_tweets(all_tweets))


def fetch_topic_tweets(topic: str, limit: int = 20) -> list[dict]:
    """Search for topic-specific tweets, filter by credibility tier."""
    queries = TOPIC_QUERIES.get(topic, [f"{topic}"])
    all_tweets = []

    for query in queries:
        try:
            result = subprocess.run(
                [str(XBIRD), "search", query, "-n", str(limit), "--json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                tweets = json.loads(result.stdout)
                for t in tweets:
                    username = t.get("author", {}).get("username", "")
                    if username in TIER1_ACCOUNTS.values():
                        t["_tier"] = 1
                    else:
                        is_credible, reason = check_tier2_credibility(t)
                        if is_credible:
                            t["_tier"] = 2
                            t["_credibility_reason"] = reason
                        else:
                            t["_tier"] = 3  # Will be filtered out
                    t["_source_handle"] = username
                all_tweets.extend(tweets)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            continue

    # Filter out Tier 3, then require recency
    credible = [t for t in all_tweets if t.get("_tier", 3) <= 2]
    return filter_recent_tweets(exclude_reply_tweets(credible))


def fetch_trending_news(limit: int = 10) -> list[dict]:
    """Fetch AI-curated trending news from X."""
    try:
        result = subprocess.run(
            [str(XBIRD), "news", "--ai-only", "--news-only", "-n", str(limit), "--json"],
            capture_output=True, text=True, timeout=20
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            # Trending items come as a different format
            items = []
            if isinstance(data, list):
                for item in data:
                    items.append({
                        "text": item.get("title", item.get("text", "")),
                        "category": item.get("category", ""),
                        "posts": item.get("posts", 0),
                        "_tier": 1,  # X's AI curation is Tier 1
                        "_source_handle": "x_trending",
                    })
            return items
    except Exception:
        pass

    # Fallback to non-JSON
    try:
        result = subprocess.run(
            [str(XBIRD), "news", "--ai-only", "--news-only", "-n", str(limit)],
            capture_output=True, text=True, timeout=20
        )
        items = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            match = re.match(r'\[AI\s*·\s*(\w+)\]\s+(.+)', line)
            if match:
                items.append({
                    "text": match.group(2).strip(),
                    "category": match.group(1),
                    "_tier": 1,
                    "_source_handle": "x_trending",
                })
        return items
    except Exception:
        return []


def fetch_trump_posts(limit: int = 5) -> list[dict]:
    """Fetch Trump's latest posts from X (both @realDonaldTrump and @POTUS)."""
    all_posts = []
    for handle in ["realDonaldTrump", "POTUS"]:
        try:
            result = subprocess.run(
                [str(XBIRD), "user-tweets", f"@{handle}", "-n", str(limit), "--json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                tweets = json.loads(result.stdout)
                for t in tweets:
                    t["_tier"] = 1
                    t["_source_handle"] = handle
                    t["_is_trump"] = True
                all_posts.extend(tweets)
        except Exception:
            continue
    return filter_recent_tweets(exclude_reply_tweets(all_posts))


# ── Main Scanner ────────────────────────────────────────────

def classify_tier1_relevance_with_llm(tweets: list[dict]) -> dict:
    """Use one batched LLM call to score Tier 1 tweets for market/geopolitical relevance."""
    if not tweets:
        return {"relevant_ids": [], "classifications": []}

    compact_lines = []
    for idx, tweet in enumerate(tweets, 1):
        compact_lines.append(
            f"{idx}. id={tweet.get('id')} | handle=@{tweet.get('_source_handle', '')} | "
            f"age_min={tweet.get('_age_minutes', 'NA')} | text={str(tweet.get('text', '')).replace(chr(10), ' ')[:280]}"
        )

    prompt = f"""You are filtering tweets for a market intelligence trading pipeline.

Goal: keep only tweets that are relevant to markets, finance, macroeconomics, government policy, geopolitics, trade, energy, commodities, sanctions, war/ceasefire, shipping chokepoints, rates, yields, inflation, jobs, tariffs, Treasury/Fed/White House/State Department policy, or major official statements with likely market impact.

Down-rank or reject tweets that are mostly sports, entertainment, celebrity, culture, lifestyle, human-interest, local crime, or general politics with no likely market/economic/geopolitical consequence.

Return JSON only in this format:
{{
  "classifications": [
    {{
      "id": "tweet id",
      "relevant": true,
      "relevance_score": 0,
      "categories": ["finance_markets", "geopolitics"],
      "reason": "short reason"
    }}
  ]
}}

Rules:
- relevance_score is 0-10
- be conservative: only mark relevant=true if the tweet is actually useful for this trading/news pipeline
- categories can include: finance_markets, economy_macro, politics_policy, geopolitics, energy_commodities, official_statement, general_news, sports, celebrity_entertainment, culture_lifestyle
- keep output compact

Tweets:
{chr(10).join(compact_lines)}"""

    try:
        from llm_router import llm_call, extract_json
        result = llm_call("tweet_relevance", prompt)
        if not result.get("success"):
            return {"relevant_ids": [t.get("id") for t in tweets if t.get("id")], "classifications": [], "error": result.get("reason", "router_failed")}
        parsed = extract_json(result.get("text", ""))
        if not parsed:
            return {"relevant_ids": [t.get("id") for t in tweets if t.get("id")], "classifications": [], "error": "parse_error"}
        classifications = parsed.get("classifications", []) if isinstance(parsed, dict) else []
        relevant_ids = [c.get("id") for c in classifications if c.get("relevant") and float(c.get("relevance_score", 0) or 0) >= 6]
        return {"relevant_ids": relevant_ids, "classifications": classifications}
    except Exception as e:
        return {"relevant_ids": [t.get("id") for t in tweets if t.get("id")], "classifications": [], "error": str(e)}


def scan_twitter(topics: list[str] = None, include_trump: bool = True,
                 tier1_limit: int = 10, search_limit: int = 15) -> dict:
    """
    Full Twitter intelligence scan.
    Returns structured data with credibility-weighted tweets.
    """
    timestamp = datetime.now().isoformat()
    results = {
        "timestamp": timestamp,
        "tier1_tweets": [],
        "relevant_tweets": [],
        "classifications": [],
        "summary": {},
    }

    # 1. Direct monitored Tier 1 account feed (only source)
    print("  🟢 Fetching Tier 1 monitored accounts...")
    tier1 = fetch_tier1_tweets(limit=tier1_limit)
    results["tier1_tweets"] = tier1
    print(f"     Got {len(tier1)} recent Tier 1 tweets")

    # Deduplicate by tweet ID
    seen_ids = set()
    for key in ["tier1_tweets"]:
        unique = []
        for t in results[key]:
            tid = t.get("id", "")
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                unique.append(t)
        results[key] = unique

    # 2. Batched LLM relevance filtering over recent Tier 1 tweets
    print("  🧠 Classifying Tier 1 tweet relevance with GPT...")
    llm_filter = classify_tier1_relevance_with_llm(results["tier1_tweets"])
    relevant_ids = set(llm_filter.get("relevant_ids", []))
    results["classifications"] = llm_filter.get("classifications", [])
    results["relevant_tweets"] = [t for t in results["tier1_tweets"] if t.get("id") in relevant_ids]
    print(f"     Relevant: {len(results['relevant_tweets'])} / {len(results['tier1_tweets'])}")

    # Summary
    results["summary"] = {
        "total_tier1": len(results["tier1_tweets"]),
        "relevant_tier1": len(results["relevant_tweets"]),
        "topics_scanned": [],
    }

    return results


def format_for_news_analyzer(twitter_data: dict) -> list[dict]:
    """Convert relevant Tier 1 Twitter results to the format expected by news_analyzer."""
    headlines = []

    classifications = {
        c.get("id"): c for c in twitter_data.get("classifications", []) if isinstance(c, dict)
    }

    for t in twitter_data.get("relevant_tweets", []):
        text = t.get("text", "")
        handle = t.get("_source_handle", "")
        if text:
            meta = classifications.get(t.get("id"), {})
            headlines.append({
                "source": f"twitter_t1_{handle}",
                "headline": f"[@{handle}] {text[:200]}",
                "tier": 1,
                "weight": 1.0,
                "published_at": t.get("_created_at_iso"),
                "age_minutes": t.get("_age_minutes"),
                "relevance_score": meta.get("relevance_score"),
                "categories": meta.get("categories", []),
                "relevance_reason": meta.get("reason", ""),
            })

    return headlines


# ── Display ─────────────────────────────────────────────────

def print_results(data: dict):
    """Pretty-print Tier 1 Twitter intelligence results."""
    print(f"\n{'=' * 70}")
    print(f"  TWITTER INTELLIGENCE SCAN — {data['timestamp'][:16]}")
    print(f"{'=' * 70}")
    s = data["summary"]
    print(f"  Tier 1 fetched: {s['total_tier1']} | Relevant after GPT filter: {s['relevant_tier1']}")

    t1 = data.get("tier1_tweets", [])
    if t1:
        print(f"\n  🟢 RAW TIER 1 ({len(t1)} tweets)")
        print(f"  {'─' * 60}")
        for t in t1[:10]:
            handle = t.get("_source_handle", "?")
            age = t.get("_age_minutes", "?")
            text = t.get("text", "")[:100].replace('\n', ' ')
            print(f"  @{handle:20s} [{age:>4}m] {text}")

    relevant = data.get("relevant_tweets", [])
    if relevant:
        classifications = {c.get('id'): c for c in data.get('classifications', []) if isinstance(c, dict)}
        print(f"\n  🎯 RELEVANT TIER 1 ({len(relevant)} tweets)")
        print(f"  {'─' * 60}")
        for t in relevant[:15]:
            handle = t.get("_source_handle", "?")
            age = t.get("_age_minutes", "?")
            meta = classifications.get(t.get('id'), {})
            score = meta.get('relevance_score', '?')
            cats = ','.join(meta.get('categories', [])[:3])
            text = t.get("text", "")[:100].replace('\n', ' ')
            print(f"  @{handle:20s} [{age:>4}m] score={score} {cats} | {text}")


def print_accounts():
    """List all monitored Tier 1 accounts."""
    print(f"\n{'=' * 70}")
    print(f"  TIER 1 MONITORED ACCOUNTS ({len(TIER1_ACCOUNTS)})")
    print(f"{'=' * 70}")

    categories = {
        "Wire Services": ["Reuters", "AP", "AFP"],
        "Financial News": ["Bloomberg", "CNBC", "WSJ", "FT", "MarketWatch"],
        "Broadcast/Print": ["BBCWorld", "BBCBreaking", "CNN", "NPR", "PBS", "NYT", "WashingtonPost", "TheEconomist", "AJEnglish"],
        "US Government": ["POTUS", "WhiteHouse", "realDonaldTrump", "StateDept", "VP", "USTreasury"],
        "Market Analysis": ["KobeissiLetter"],
    }

    for cat, accounts in categories.items():
        print(f"\n  {cat}:")
        for name in accounts:
            handle = TIER1_ACCOUNTS.get(name, name)
            print(f"    @{handle}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 1 Twitter/X intelligence with batched GPT relevance filtering")
    parser.add_argument("--topic", type=str, nargs="*", help="Unused in Tier 1-only mode (kept for backward compatibility)")
    parser.add_argument("--accounts", action="store_true", help="List monitored Tier 1 accounts")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--no-trump", action="store_true", help="Unused in Tier 1-only mode (Trump is already part of Tier 1)")
    args = parser.parse_args()

    if args.accounts:
        print_accounts()
        return

    print("🐦 Twitter Intelligence Scanner")
    print("=" * 50)

    data = scan_twitter(
        topics=args.topic,
        include_trump=not args.no_trump,
    )

    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print_results(data)


if __name__ == "__main__":
    main()
