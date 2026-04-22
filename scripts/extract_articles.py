#!/usr/bin/env python3
"""
Article extraction pipeline for market intelligence.
Reads blogwatcher articles, fetches full text via trafilatura, saves as JSONL.

Usage:
    python3 extract_articles.py                  # Extract all unread articles
    python3 extract_articles.py --limit 10       # Extract first N articles
    python3 extract_articles.py --scan           # Scan for new articles first, then extract
    python3 extract_articles.py --source "Iran"  # Filter by source name
"""

import json
import os
import re
import subprocess
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import urllib.request

import trafilatura
from googlenewsdecoder import new_decoderv1

# Config
DATA_DIR = Path.home() / "market-intel" / "data"
ARTICLES_FILE = DATA_DIR / "articles.jsonl"
EXTRACTED_IDS_FILE = DATA_DIR / "extracted_ids.json"
BLOGWATCHER = Path.home() / "go" / "bin" / "blogwatcher"

# Topic classification keywords
TOPIC_KEYWORDS = {
    "iran": ["iran", "tehran", "irgc", "khamenei", "persian gulf", "strait of hormuz",
             "iranian", "hezbollah", "houthi", "nuclear deal", "jcpoa"],
    "politics": ["trump", "biden", "congress", "senate", "supreme court", "democrat",
                 "republican", "election", "whitehouse", "executive order", "legislation",
                 "gop", "dnc", "rnc", "impeach", "veto", "filibuster", "speaker"],
    "finance": ["stock", "market", "wall street", "s&p", "nasdaq", "dow jones",
                "crypto", "bitcoin", "ethereum", "trading", "hedge fund", "ipo",
                "earnings", "revenue", "bonds", "yield", "treasury", "forex"],
    "geopolitics": ["nato", "china", "russia", "ukraine", "taiwan", "sanctions",
                    "military", "war", "conflict", "ceasefire", "diplomacy", "treaty",
                    "un security council", "g7", "g20", "brics", "territorial"],
    "economy": ["inflation", "gdp", "unemployment", "interest rate", "federal reserve",
                "fed", "recession", "tariff", "trade deficit", "cpi", "ppi",
                "consumer spending", "housing market", "debt ceiling", "fiscal"],
}


def classify_topics(title: str, body: str, source: str = "") -> list[str]:
    """Classify article into topic categories based on keyword matching."""
    text = (title + " " + (body or "")).lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            topics.append(topic)
    
    # Source-based classification for non-English content
    source_lower = source.lower()
    if "iran" in source_lower and "iran" not in topics:
        topics.append("iran")
        if "geopolitics" not in topics:
            topics.append("geopolitics")
    
    return topics if topics else ["general"]


def load_extracted_ids() -> set:
    """Load set of already-extracted article URLs to avoid re-processing."""
    if EXTRACTED_IDS_FILE.exists():
        return set(json.loads(EXTRACTED_IDS_FILE.read_text()))
    return set()


def save_extracted_ids(ids: set):
    """Save extracted article IDs."""
    EXTRACTED_IDS_FILE.write_text(json.dumps(list(ids), indent=2))


def get_blogwatcher_articles() -> list[dict]:
    """Parse blogwatcher articles output into structured data."""
    result = subprocess.run(
        [str(BLOGWATCHER), "articles"],
        capture_output=True, text=True
    )
    
    articles = []
    lines = result.stdout.strip().split("\n")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match article lines: [ID] [new] Title - Source
        match = re.match(r'\[(\d+)\]\s+\[new\]\s+(.+)', line)
        if match:
            article_id = match.group(1)
            title = match.group(2).strip()
            
            blog = ""
            url = ""
            published = ""
            
            # Look ahead for metadata lines
            for j in range(i + 1, min(i + 5, len(lines))):
                meta_line = lines[j].strip()
                if meta_line.startswith("Blog:"):
                    blog = meta_line.replace("Blog:", "").strip()
                elif meta_line.startswith("URL:"):
                    url = meta_line.replace("URL:", "").strip()
                elif meta_line.startswith("Published:"):
                    published = meta_line.replace("Published:", "").strip()
            
            if url:
                articles.append({
                    "id": article_id,
                    "title": title,
                    "source": blog,
                    "url": url,
                    "published": published,
                })
        i += 1
    
    return articles


def resolve_google_news_url(url: str) -> str:
    """Resolve Google News redirect URLs to actual article URLs."""
    if "news.google.com/rss/articles/" not in url:
        return url
    
    try:
        result = new_decoderv1(url, interval=5)
        if result and result.get("status") and result.get("decoded_url"):
            return result["decoded_url"]
    except Exception as e:
        print(f"           ⚠️  Google News decode failed: {e}", file=sys.stderr)
    
    return url


def extract_article_text(url: str) -> dict | None:
    """Fetch and extract article body text using trafilatura (with URL resolution)."""
    actual_url = resolve_google_news_url(url)
    if actual_url != url:
        print(f"           → Resolved to: {actual_url[:80]}")
    return extract_article_text_direct(actual_url)


def extract_article_text_direct(url: str) -> dict | None:
    """Fetch and extract article body text using trafilatura (no URL resolution)."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            output_format="txt"
        )
        
        metadata = trafilatura.extract(
            downloaded,
            include_comments=False,
            output_format="json"
        )
        
        meta = {}
        if metadata:
            meta = json.loads(metadata)
        
        return {
            "body": text,
            "author": meta.get("author", ""),
            "hostname": meta.get("hostname", ""),
            "excerpt": meta.get("excerpt", ""),
            "categories": meta.get("categories", ""),
        }
    except Exception as e:
        print(f"  ⚠️  Extraction failed for {url}: {e}", file=sys.stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract full article text from blogwatcher feeds")
    parser.add_argument("--limit", type=int, default=0, help="Max articles to extract (0=all)")
    parser.add_argument("--scan", action="store_true", help="Run blogwatcher scan first")
    parser.add_argument("--source", type=str, default="", help="Filter by source name (partial match)")
    parser.add_argument("--rescan", action="store_true", help="Re-extract already processed articles")
    args = parser.parse_args()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Optionally scan for new articles first
    if args.scan:
        print("📡 Scanning for new articles...")
        subprocess.run([str(BLOGWATCHER), "scan"], capture_output=True)
        print()
    
    # Get articles from blogwatcher
    print("📰 Reading blogwatcher articles...")
    articles = get_blogwatcher_articles()
    print(f"   Found {len(articles)} unread articles")
    
    # Filter by source if specified
    if args.source:
        articles = [a for a in articles if args.source.lower() in a["source"].lower()]
        print(f"   Filtered to {len(articles)} articles from '{args.source}'")
    
    # Skip already extracted
    extracted_ids = load_extracted_ids()
    if not args.rescan:
        articles = [a for a in articles if a["url"] not in extracted_ids]
        print(f"   {len(articles)} new articles to extract")
    
    # Apply limit
    if args.limit > 0:
        articles = articles[:args.limit]
        print(f"   Limited to {args.limit} articles")
    
    if not articles:
        print("\n✅ No new articles to extract.")
        return
    
    # Pre-resolve Google News URLs in parallel
    gnews_articles = [a for a in articles if "news.google.com" in a["url"]]
    if gnews_articles:
        print(f"\n🔗 Resolving {len(gnews_articles)} Google News URLs (parallel)...")
        resolved_map = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(resolve_google_news_url, a["url"]): a["url"] for a in gnews_articles}
            for future in as_completed(futures):
                orig = futures[future]
                try:
                    resolved_map[orig] = future.result()
                except Exception:
                    resolved_map[orig] = orig
        
        # Update article URLs
        for article in articles:
            if article["url"] in resolved_map:
                article["resolved_url"] = resolved_map[article["url"]]
            else:
                article["resolved_url"] = article["url"]
        resolved_count = sum(1 for u, r in resolved_map.items() if u != r)
        print(f"   ✅ Resolved {resolved_count}/{len(gnews_articles)} URLs")
    else:
        for article in articles:
            article["resolved_url"] = article["url"]
    
    # Extract each article
    print(f"\n🔍 Extracting {len(articles)} articles...\n")
    
    extracted_count = 0
    failed_count = 0
    
    with open(ARTICLES_FILE, "a") as f:
        for i, article in enumerate(articles, 1):
            print(f"  [{i}/{len(articles)}] {article['title'][:80]}...")
            
            actual_url = article["resolved_url"]
            if actual_url != article["url"]:
                print(f"           → {actual_url[:80]}")
            
            result = extract_article_text_direct(actual_url)
            
            if result and result["body"]:
                # Build structured record
                url_hash = hashlib.md5(article["url"].encode()).hexdigest()[:12]
                topics = classify_topics(article["title"], result["body"])
                
                record = {
                    "id": url_hash,
                    "title": article["title"],
                    "source": article["source"],
                    "author": result["author"],
                    "hostname": result["hostname"],
                    "url": actual_url,
                    "original_url": article["url"],
                    "published": article["published"],
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "topics": topics,
                    "excerpt": result["excerpt"],
                    "body": result["body"],
                    "word_count": len(result["body"].split()),
                }
                
                f.write(json.dumps(record) + "\n")
                extracted_ids.add(article["url"])
                extracted_count += 1
                
                topic_str = ", ".join(topics)
                print(f"           ✅ {record['word_count']} words | Topics: {topic_str}")
            else:
                failed_count += 1
                extracted_ids.add(article["url"])  # Still mark as processed
                print(f"           ❌ Failed to extract")
            
            # Save progress every 20 articles
            if i % 20 == 0:
                save_extracted_ids(extracted_ids)
                print(f"           💾 Progress saved ({extracted_count} extracted so far)")
            
            # Be polite to servers
            time.sleep(0.3)
    
    # Save progress
    save_extracted_ids(extracted_ids)
    
    print(f"\n{'='*60}")
    print(f"📊 Extraction Summary")
    print(f"   ✅ Extracted: {extracted_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Saved to: {ARTICLES_FILE}")
    print(f"   📂 Total articles in DB: {sum(1 for _ in open(ARTICLES_FILE)) if ARTICLES_FILE.exists() else 0}")


if __name__ == "__main__":
    main()
