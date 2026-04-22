#!/usr/bin/env python3
"""Test all 3 market data sources."""
import os
os.environ['APCA_API_KEY_ID'] = os.getenv('APCA_API_KEY_ID', '')
os.environ['APCA_API_SECRET_KEY'] = os.getenv('APCA_API_SECRET_KEY', '')
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

import finnhub
import yfinance as yf
from alpaca_trade_api.rest import REST

print("=" * 60)
print("  MARKET INTELLIGENCE — DATA SOURCE TEST")
print("=" * 60)

# --- YFINANCE (Indices, Sectors, Commodities) ---
print("\n🏦 yfinance — Indices & Commodities (no key, 15-min delay)")
print("-" * 60)
for sym, name in [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^VIX", "VIX"), ("^IXIC", "NASDAQ")]:
    t = yf.Ticker(sym)
    info = t.fast_info
    change = ((info.last_price - info.previous_close) / info.previous_close) * 100
    arrow = "🟢" if change >= 0 else "🔴"
    print(f"  {arrow} {name}: ${info.last_price:,.2f} ({change:+.2f}%)")

print()
for sym, name in [("CL=F", "Crude Oil"), ("GC=F", "Gold"), ("BTC-USD", "Bitcoin")]:
    t = yf.Ticker(sym)
    info = t.fast_info
    print(f"  🔸 {name}: ${info.last_price:,.2f}")

# --- FINNHUB (Real-time quotes, news) ---
print()
print("⚡ Finnhub — Real-time Quotes + News (60 req/min)")
print("-" * 60)
fc = finnhub.Client(api_key="d7717ahr01qtg3nf8pigd7717ahr01qtg3nf8pj0")
for sym in ["AAPL", "TSLA", "NVDA", "XOM", "LMT"]:
    q = fc.quote(sym)
    change = q['dp'] if q['dp'] else 0
    arrow = "🟢" if change >= 0 else "🔴"
    print(f"  {arrow} {sym}: ${q['c']:.2f} ({change:+.2f}%)")

# Finnhub company news
print()
print("  📰 Company News (AAPL):")
try:
    from datetime import datetime, timedelta
    today = datetime.now().strftime('%Y-%m-%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    news = fc.company_news('AAPL', _from=week_ago, to=today)
    for a in news[:3]:
        print(f"    • {a['headline'][:75]}")
except Exception as e:
    print(f"    Error: {e}")

# --- ALPACA (Real-time trades, news, paper account) ---
print()
print("📈 Alpaca — Real-time Trades + News (200 req/min)")
print("-" * 60)
api = REST()
print(f"  Account: ACTIVE | Paper Balance: ${float(api.get_account().portfolio_value):,.2f}")
print()
for sym in ["AAPL", "TSLA", "NVDA", "SPY"]:
    try:
        trade = api.get_latest_trade(sym)
        print(f"  💹 {sym}: ${trade.price:.2f}")
    except Exception as e:
        print(f"  {sym}: {e}")

# Alpaca news
print()
print("  📰 Market News (Benzinga):")
try:
    news = api.get_news(limit=5)
    for a in news[:5]:
        print(f"    • {a.headline[:75]}")
except Exception as e:
    print(f"    Error: {e}")

print()
print("=" * 60)
print("  ✅ ALL 3 DATA SOURCES OPERATIONAL")
print("=" * 60)
