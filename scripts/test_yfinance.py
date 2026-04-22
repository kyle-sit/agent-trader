#!/usr/bin/env python3
"""Quick test of yfinance - no API key needed."""
import yfinance as yf

# Test major indices
print("📊 Major Indices (yfinance)")
print("=" * 50)
for symbol, name in [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^VIX", "VIX"), ("^IXIC", "NASDAQ")]:
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    print(f"  {name}: ${info.last_price:,.2f}  (prev close: ${info.previous_close:,.2f})")

print()

# Test sector ETFs
print("📈 Sector ETFs")
print("=" * 50)
sectors = {
    "XLF": "Financials", "XLE": "Energy", "XLK": "Technology",
    "XLV": "Healthcare", "XLI": "Industrials", "XLP": "Consumer Staples"
}
for sym, name in sectors.items():
    ticker = yf.Ticker(sym)
    info = ticker.fast_info
    change = ((info.last_price - info.previous_close) / info.previous_close) * 100
    arrow = "🟢" if change >= 0 else "🔴"
    print(f"  {arrow} {name} ({sym}): ${info.last_price:.2f}  ({change:+.2f}%)")

print()

# Test oil (relevant for Iran/geopolitics)
print("🛢️  Commodities")
print("=" * 50)
for sym, name in [("CL=F", "Crude Oil"), ("GC=F", "Gold"), ("BTC-USD", "Bitcoin")]:
    ticker = yf.Ticker(sym)
    info = ticker.fast_info
    print(f"  {name}: ${info.last_price:,.2f}")
