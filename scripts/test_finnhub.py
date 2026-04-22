#!/usr/bin/env python3
"""Test Finnhub API connection."""
import finnhub

client = finnhub.Client(api_key="d7717ahr01qtg3nf8pigd7717ahr01qtg3nf8pj0")

# Test quote
print("📊 Finnhub — Real-time Quotes")
print("=" * 50)
for sym in ["AAPL", "TSLA", "SPY", "QQQ"]:
    q = client.quote(sym)
    change = q['dp'] if q['dp'] else 0
    arrow = "🟢" if change >= 0 else "🔴"
    print(f"  {arrow} {sym}: ${q['c']:.2f}  ({change:+.2f}%)  Vol: {q.get('v', 'N/A')}")

# Test market news
print()
print("📰 Finnhub — Latest Market News")
print("=" * 50)
news = client.general_news('general', min_id=0)
for article in news[:5]:
    print(f"  • {article['headline'][:80]}")
    print(f"    Source: {article['source']} | {article['datetime']}")
    print()

# Test company news sentiment
print("🧠 Finnhub — News Sentiment (AAPL)")
print("=" * 50)
sentiment = client.news_sentiment('AAPL')
if sentiment and 'sentiment' in sentiment:
    s = sentiment['sentiment']
    print(f"  Bullish: {s.get('bullishPercent', 'N/A')}")
    print(f"  Bearish: {s.get('bearishPercent', 'N/A')}")
    print(f"  Articles analyzed: {sentiment.get('buzz', {}).get('articlesInLastWeek', 'N/A')}")
