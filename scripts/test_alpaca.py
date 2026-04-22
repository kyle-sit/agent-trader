#!/usr/bin/env python3
"""Test Alpaca API connection."""
import os
os.environ['APCA_API_KEY_ID'] = os.getenv('APCA_API_KEY_ID', '')
os.environ['APCA_API_SECRET_KEY'] = os.getenv('APCA_API_SECRET_KEY', '')
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta

api = REST()

# Test account connection
print("🔗 Alpaca — Account Connection")
print("=" * 50)
account = api.get_account()
print(f"  Status: {account.status}")
print(f"  Buying Power: ${float(account.buying_power):,.2f}")
print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")

# Test market data - latest quotes
print()
print("📊 Alpaca — Latest Quotes")
print("=" * 50)
for sym in ["AAPL", "TSLA", "SPY", "NVDA"]:
    try:
        quote = api.get_latest_trade(sym)
        print(f"  {sym}: ${quote.price:.2f}")
    except Exception as e:
        print(f"  {sym}: Error - {e}")

# Test historical bars
print()
print("📈 Alpaca — Historical Bars (SPY last 5 days)")
print("=" * 50)
end = datetime.now()
start = end - timedelta(days=7)
try:
    bars = api.get_bars("SPY", TimeFrame.Day, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), limit=5).df
    for idx, row in bars.iterrows():
        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        change = ((row['close'] - row['open']) / row['open']) * 100
        arrow = "🟢" if change >= 0 else "🔴"
        print(f"  {arrow} {date_str}: O=${row['open']:.2f} H=${row['high']:.2f} L=${row['low']:.2f} C=${row['close']:.2f} ({change:+.2f}%)")
except Exception as e:
    print(f"  Error: {e}")

# Test crypto
print()
print("🪙 Alpaca — Crypto")
print("=" * 50)
for sym in ["BTC/USD", "ETH/USD"]:
    try:
        quote = api.get_latest_crypto_trade(sym)
        print(f"  {sym}: ${quote.price:,.2f}")
    except Exception as e:
        print(f"  {sym}: Error - {e}")

# Test news
print()
print("📰 Alpaca — News")
print("=" * 50)
try:
    news = api.get_news(symbol="SPY", limit=5)
    for article in news:
        print(f"  • {article.headline[:80]}")
        print(f"    Source: {article.source}")
        print()
except Exception as e:
    print(f"  News Error: {e}")
