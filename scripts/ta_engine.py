#!/usr/bin/env python3
"""
Technical Analysis Engine for Market Intelligence Pipeline.

Fetches price data via yfinance, calculates indicators, identifies signals.
Designed to be called by the correlation engine or standalone.

Usage:
    python3 ta_engine.py AAPL TSLA SPY          # Analyze specific symbols
    python3 ta_engine.py --watchlist default     # Analyze default watchlist
    python3 ta_engine.py AAPL --json             # JSON output for pipeline
    python3 ta_engine.py SPY --detail            # Full indicator breakdown
"""

import json
import sys
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


# ── Watchlists ──────────────────────────────────────────────

WATCHLISTS = {
    "default": [
        # Indices
        "SPY", "QQQ", "DIA", "IWM",
        # Geopolitics / Defense
        "LMT", "RTX", "NOC", "GD",
        # Energy (Iran/Oil sensitive)
        "XOM", "CVX", "OXY", "XLE", "USO",
        # Financials
        "XLF", "JPM", "GS",
        # Tech megacaps
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
        # Commodities
        "GLD", "SLV",
        # Volatility
        "VXX",
    ],
    "iran": [
        "XOM", "CVX", "OXY", "XLE", "USO",  # Oil
        "LMT", "RTX", "NOC", "GD", "BA",     # Defense
        "GLD", "SLV",                          # Safe havens
        "SPY", "QQQ",                          # Broad market
    ],
    "economy": [
        "SPY", "QQQ", "DIA", "IWM",           # Indices
        "XLF", "JPM", "GS", "BAC",            # Financials
        "TLT", "SHY",                          # Bonds
        "XLP", "XLU",                          # Defensive sectors
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD",
    ],
}


# ── Data Classes ────────────────────────────────────────────

@dataclass
class IndicatorValues:
    """Current indicator values for a symbol."""
    # Price
    price: float
    prev_close: float
    change_pct: float

    # Moving Averages
    ema_9: float
    ema_21: float
    sma_50: float
    sma_200: float

    # Trend
    macd: float
    macd_signal: float
    macd_histogram: float
    adx: float

    # Momentum
    rsi: float
    stoch_rsi: float

    # Volatility
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_pct: float  # Where price is within bands (0=lower, 1=upper)
    atr: float
    atr_pct: float  # ATR as % of price

    # Volume
    volume: int
    volume_sma_20: float
    volume_ratio: float  # Current vol / 20-day avg
    obv_slope: float     # OBV direction (positive = accumulation)


@dataclass
class Signal:
    """A trading signal identified by the TA engine."""
    symbol: str
    signal_type: str       # "bullish", "bearish", "neutral"
    strength: str          # "strong", "moderate", "weak"
    reasons: list
    entry_zone: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    risk_reward: Optional[float] = None


# ── Core Functions ──────────────────────────────────────────

def fetch_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df


# ── Intraday Timeframes ────────────────────────────────────
# yfinance limits: 5m data goes back 60 days, 15m/1h go back 730 days

INTRADAY_CONFIGS = {
    "5m":  {"period": "5d",  "interval": "5m",  "label": "5-Minute"},
    "15m": {"period": "14d", "interval": "15m", "label": "15-Minute"},
    "1h":  {"period": "30d", "interval": "1h",  "label": "1-Hour"},
    "1d":  {"period": "6mo", "interval": "1d",  "label": "Daily"},
}


def calculate_indicators(df: pd.DataFrame) -> IndicatorValues:
    """Calculate all technical indicators from OHLCV data."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Current price
    price = close.iloc[-1]
    prev_close = close.iloc[-2]
    change_pct = ((price - prev_close) / prev_close) * 100

    # Moving Averages
    ema_9 = EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema_21 = EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    sma_50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    sma_200 = SMAIndicator(close, window=min(200, len(close) - 1)).sma_indicator().iloc[-1]

    # MACD
    macd_ind = MACD(close)
    macd_val = macd_ind.macd().iloc[-1]
    macd_signal = macd_ind.macd_signal().iloc[-1]
    macd_hist = macd_ind.macd_diff().iloc[-1]

    # ADX (trend strength)
    adx_ind = ADXIndicator(high, low, close)
    adx = adx_ind.adx().iloc[-1]

    # RSI
    rsi = RSIIndicator(close).rsi().iloc[-1]

    # Stochastic RSI
    stoch_rsi_ind = StochRSIIndicator(close)
    stoch_rsi = stoch_rsi_ind.stochrsi().iloc[-1] * 100

    # Bollinger Bands
    bb = BollingerBands(close)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_middle = bb.bollinger_mavg().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_pct = bb.bollinger_pband().iloc[-1]

    # ATR
    atr_ind = AverageTrueRange(high, low, close)
    atr = atr_ind.average_true_range().iloc[-1]
    atr_pct = (atr / price) * 100

    # Volume
    vol = int(volume.iloc[-1])
    vol_sma_20 = volume.rolling(20).mean().iloc[-1]
    vol_ratio = vol / vol_sma_20 if vol_sma_20 > 0 else 1.0

    # OBV slope (5-day)
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_recent = obv.iloc[-5:]
    obv_slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / abs(obv_recent.iloc[0]) * 100 if obv_recent.iloc[0] != 0 else 0

    return IndicatorValues(
        price=round(price, 2),
        prev_close=round(prev_close, 2),
        change_pct=round(change_pct, 2),
        ema_9=round(ema_9, 2),
        ema_21=round(ema_21, 2),
        sma_50=round(sma_50, 2),
        sma_200=round(sma_200, 2),
        macd=round(macd_val, 4),
        macd_signal=round(macd_signal, 4),
        macd_histogram=round(macd_hist, 4),
        adx=round(adx, 2),
        rsi=round(rsi, 2),
        stoch_rsi=round(stoch_rsi, 2),
        bb_upper=round(bb_upper, 2),
        bb_middle=round(bb_middle, 2),
        bb_lower=round(bb_lower, 2),
        bb_pct=round(bb_pct, 4),
        atr=round(atr, 2),
        atr_pct=round(atr_pct, 2),
        volume=vol,
        volume_sma_20=round(vol_sma_20, 0),
        volume_ratio=round(vol_ratio, 2),
        obv_slope=round(obv_slope, 4),
    )


def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Identify key support and resistance levels using pivot points and price clusters."""
    recent = df.tail(lookback)
    high = recent['High']
    low = recent['Low']
    close = recent['Close']

    # Simple pivot point method
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    pivot = (last_high + last_low + last_close) / 3
    r1 = 2 * pivot - last_low
    r2 = pivot + (last_high - last_low)
    s1 = 2 * pivot - last_high
    s2 = pivot - (last_high - last_low)

    # Recent swing highs/lows (local extrema)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(recent) - 2):
        if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
           high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
            swing_highs.append(round(high.iloc[i], 2))
        if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
           low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
            swing_lows.append(round(low.iloc[i], 2))

    return {
        "pivot": round(pivot, 2),
        "resistance_1": round(r1, 2),
        "resistance_2": round(r2, 2),
        "support_1": round(s1, 2),
        "support_2": round(s2, 2),
        "swing_highs": sorted(swing_highs, reverse=True)[:3],
        "swing_lows": sorted(swing_lows)[:3],
    }


def generate_signal(symbol: str, ind: IndicatorValues, sr: dict) -> Signal:
    """Generate a trading signal based on indicator confluence."""
    bullish = []
    bearish = []
    neutral = []

    # ── Trend ──
    # EMA alignment
    if ind.ema_9 > ind.ema_21 > ind.sma_50:
        bullish.append("EMA alignment bullish (9 > 21 > 50)")
    elif ind.ema_9 < ind.ema_21 < ind.sma_50:
        bearish.append("EMA alignment bearish (9 < 21 < 50)")

    # Price vs 200 SMA
    if ind.price > ind.sma_200:
        bullish.append(f"Above 200 SMA (${ind.sma_200})")
    else:
        bearish.append(f"Below 200 SMA (${ind.sma_200})")

    # MACD
    if ind.macd > ind.macd_signal and ind.macd_histogram > 0:
        bullish.append("MACD bullish crossover")
    elif ind.macd < ind.macd_signal and ind.macd_histogram < 0:
        bearish.append("MACD bearish crossover")

    # ADX trend strength
    if ind.adx > 25:
        if ind.ema_9 > ind.ema_21:
            bullish.append(f"Strong trend (ADX {ind.adx})")
        else:
            bearish.append(f"Strong downtrend (ADX {ind.adx})")

    # ── Momentum ──
    # RSI
    if ind.rsi < 30:
        bullish.append(f"RSI oversold ({ind.rsi})")
    elif ind.rsi > 70:
        bearish.append(f"RSI overbought ({ind.rsi})")
    elif ind.rsi < 45:
        bearish.append(f"RSI weak ({ind.rsi})")
    elif ind.rsi > 55:
        bullish.append(f"RSI strong ({ind.rsi})")

    # Stochastic RSI
    if ind.stoch_rsi < 20:
        bullish.append(f"StochRSI oversold ({ind.stoch_rsi:.0f})")
    elif ind.stoch_rsi > 80:
        bearish.append(f"StochRSI overbought ({ind.stoch_rsi:.0f})")

    # ── Volatility ──
    # Bollinger Band position
    if ind.bb_pct < 0:
        bullish.append("Price below lower Bollinger Band (mean reversion)")
    elif ind.bb_pct > 1:
        bearish.append("Price above upper Bollinger Band (overextended)")

    # ── Volume ──
    if ind.volume_ratio > 2.0:
        if ind.change_pct > 0:
            bullish.append(f"High volume rally ({ind.volume_ratio}x avg)")
        else:
            bearish.append(f"High volume selloff ({ind.volume_ratio}x avg)")
    elif ind.volume_ratio > 1.5:
        neutral.append(f"Above average volume ({ind.volume_ratio}x)")

    # OBV
    if ind.obv_slope > 1:
        bullish.append("OBV rising (accumulation)")
    elif ind.obv_slope < -1:
        bearish.append("OBV falling (distribution)")

    # ── Support/Resistance proximity ──
    price = ind.price
    if sr["support_1"] and abs(price - sr["support_1"]) / price < 0.02:
        bullish.append(f"Near support S1 (${sr['support_1']})")
    if sr["resistance_1"] and abs(price - sr["resistance_1"]) / price < 0.02:
        bearish.append(f"Near resistance R1 (${sr['resistance_1']})")

    # ── Determine signal ──
    bull_count = len(bullish)
    bear_count = len(bearish)
    total = bull_count + bear_count

    if total == 0:
        signal_type = "neutral"
        strength = "weak"
        reasons = neutral if neutral else ["No clear signals"]
    elif bull_count >= bear_count + 3:
        signal_type = "bullish"
        strength = "strong" if bull_count >= 5 else "moderate"
        reasons = bullish
    elif bear_count >= bull_count + 3:
        signal_type = "bearish"
        strength = "strong" if bear_count >= 5 else "moderate"
        reasons = bearish
    elif bull_count > bear_count:
        signal_type = "bullish"
        strength = "weak"
        reasons = bullish
    elif bear_count > bull_count:
        signal_type = "bearish"
        strength = "weak"
        reasons = bearish
    else:
        signal_type = "neutral"
        strength = "weak"
        reasons = bullish + bearish

    # ── Entry, stop, target ──
    entry_zone = None
    stop_loss = None
    target = None
    risk_reward = None

    if signal_type == "bullish":
        entry_zone = round(price, 2)
        stop_loss = round(max(sr["support_1"], price - 2 * ind.atr), 2)
        target = round(min(sr["resistance_1"], price + 3 * ind.atr), 2)
        risk = abs(entry_zone - stop_loss)
        reward = abs(target - entry_zone)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
    elif signal_type == "bearish":
        entry_zone = round(price, 2)
        stop_loss = round(min(sr["resistance_1"], price + 2 * ind.atr), 2)
        target = round(max(sr["support_1"], price - 3 * ind.atr), 2)
        risk = abs(stop_loss - entry_zone)
        reward = abs(entry_zone - target)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

    return Signal(
        symbol=symbol,
        signal_type=signal_type,
        strength=strength,
        reasons=reasons,
        entry_zone=entry_zone,
        stop_loss=stop_loss,
        target=target,
        risk_reward=risk_reward,
    )


# ── Analysis Runner ─────────────────────────────────────────

def analyze_symbol(symbol: str, detail: bool = False, timeframe: str = "1d") -> dict:
    """Run full TA analysis on a symbol. Returns dict with indicators, S/R, and signal."""
    config = INTRADAY_CONFIGS.get(timeframe, INTRADAY_CONFIGS["1d"])
    df = fetch_data(symbol, period=config["period"], interval=config["interval"])
    indicators = calculate_indicators(df)
    sr = find_support_resistance(df)
    signal = generate_signal(symbol, indicators, sr)

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timeframe_label": config["label"],
        "timestamp": datetime.now().isoformat(),
        "indicators": asdict(indicators),
        "support_resistance": sr,
        "signal": asdict(signal),
    }

    # Optional LLM interpretation
    try:
        from llm_hooks import interpret_ta
        interpretation = interpret_ta(symbol, asdict(indicators), asdict(signal), sr)
        if interpretation:
            result["llm_interpretation"] = interpretation
            # Apply conviction modifier if present
            modifier = interpretation.get("conviction_modifier", 0)
            if modifier != 0:
                result["signal"]["llm_conviction_modifier"] = modifier
    except (ImportError, Exception):
        pass

    return result


def analyze_multi_timeframe(symbol: str, timeframes: list[str] = None) -> dict:
    """
    Multi-timeframe analysis. Checks alignment across timeframes.
    Strong signal = all timeframes agree. Mixed = caution.
    """
    if timeframes is None:
        timeframes = ["1h", "1d"]

    results = {}
    for tf in timeframes:
        try:
            results[tf] = analyze_symbol(symbol, timeframe=tf)
        except Exception as e:
            results[tf] = {"error": str(e)}

    # Check alignment across timeframes
    directions = []
    for tf, r in results.items():
        if "error" not in r:
            directions.append(r["signal"]["signal_type"])

    if all(d == "bullish" for d in directions if d != "neutral"):
        mtf_alignment = "bullish_aligned"
    elif all(d == "bearish" for d in directions if d != "neutral"):
        mtf_alignment = "bearish_aligned"
    elif "bullish" in directions and "bearish" in directions:
        mtf_alignment = "conflicting"
    else:
        mtf_alignment = "neutral"

    return {
        "symbol": symbol,
        "timeframes": results,
        "mtf_alignment": mtf_alignment,
        "timeframes_analyzed": timeframes,
    }


def print_signal(result: dict, detail: bool = False):
    """Pretty-print a signal result."""
    sig = result["signal"]
    ind = result["indicators"]
    sr = result["support_resistance"]

    # Signal icon
    if sig["signal_type"] == "bullish":
        icon = "🟢" if sig["strength"] == "strong" else "🟡"
    elif sig["signal_type"] == "bearish":
        icon = "🔴" if sig["strength"] == "strong" else "🟠"
    else:
        icon = "⚪"

    print(f"\n{icon} {sig['symbol']} — {sig['signal_type'].upper()} ({sig['strength']})")
    print(f"   Price: ${ind['price']}  ({ind['change_pct']:+.2f}%)  |  RSI: {ind['rsi']}  |  MACD: {'↑' if ind['macd_histogram'] > 0 else '↓'}  |  Vol: {ind['volume_ratio']}x avg")

    if sig["entry_zone"]:
        print(f"   Entry: ${sig['entry_zone']}  |  Stop: ${sig['stop_loss']}  |  Target: ${sig['target']}  |  R:R {sig['risk_reward']}")

    print(f"   Reasons:")
    for r in sig["reasons"]:
        print(f"     • {r}")

    if detail:
        print(f"\n   ── Indicators ──")
        print(f"   EMA 9/21: ${ind['ema_9']} / ${ind['ema_21']}")
        print(f"   SMA 50/200: ${ind['sma_50']} / ${ind['sma_200']}")
        print(f"   MACD: {ind['macd']:.4f} | Signal: {ind['macd_signal']:.4f} | Hist: {ind['macd_histogram']:.4f}")
        print(f"   ADX: {ind['adx']}  |  StochRSI: {ind['stoch_rsi']:.0f}")
        print(f"   BB: ${ind['bb_lower']} — ${ind['bb_middle']} — ${ind['bb_upper']} (pct: {ind['bb_pct']:.2f})")
        print(f"   ATR: ${ind['atr']} ({ind['atr_pct']:.1f}%)")
        print(f"   OBV slope: {ind['obv_slope']:.2f}%")
        print(f"\n   ── Support / Resistance ──")
        print(f"   S2: ${sr['support_2']}  |  S1: ${sr['support_1']}  |  Pivot: ${sr['pivot']}")
        print(f"   R1: ${sr['resistance_1']}  |  R2: ${sr['resistance_2']}")
        if sr["swing_highs"]:
            print(f"   Swing Highs: {sr['swing_highs']}")
        if sr["swing_lows"]:
            print(f"   Swing Lows: {sr['swing_lows']}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Technical Analysis Engine")
    parser.add_argument("symbols", nargs="*", help="Symbols to analyze")
    parser.add_argument("--watchlist", "-w", type=str, help="Use a predefined watchlist")
    parser.add_argument("--timeframe", "-t", type=str, default="1d", choices=["5m", "15m", "1h", "1d"], help="Timeframe for analysis")
    parser.add_argument("--mtf", action="store_true", help="Multi-timeframe analysis (1h + 1d)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--detail", "-d", action="store_true", help="Show full indicator detail")
    parser.add_argument("--signals-only", "-s", action="store_true", help="Only show non-neutral signals")
    args = parser.parse_args()

    # Determine symbols
    if args.watchlist:
        symbols = WATCHLISTS.get(args.watchlist, [])
        if not symbols:
            print(f"Unknown watchlist: {args.watchlist}")
            print(f"Available: {', '.join(WATCHLISTS.keys())}")
            sys.exit(1)
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = WATCHLISTS["default"]

    tf_label = INTRADAY_CONFIGS.get(args.timeframe, {}).get("label", args.timeframe)

    if not args.json:
        print("=" * 60)
        print(f"  TA ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        mode = "Multi-timeframe (1h + 1d)" if args.mtf else tf_label
        print(f"  Analyzing {len(symbols)} symbols | Timeframe: {mode}")
        print("=" * 60)

    results = []
    errors = []

    for sym in symbols:
        try:
            if args.mtf:
                mtf = analyze_multi_timeframe(sym)
                # Use daily for display, annotate with MTF alignment
                result = mtf["timeframes"].get("1d", mtf["timeframes"].get(list(mtf["timeframes"].keys())[0]))
                if "error" in result:
                    raise ValueError(result["error"])
                result["mtf_alignment"] = mtf["mtf_alignment"]
                results.append(result)

                if not args.json:
                    if args.signals_only and result["signal"]["signal_type"] == "neutral":
                        continue
                    print_signal(result, detail=args.detail)
                    align_icon = {"bullish_aligned": "🟢", "bearish_aligned": "🔴", "conflicting": "⚠️", "neutral": "⚪"}.get(mtf["mtf_alignment"], "❓")
                    print(f"   MTF: {align_icon} {mtf['mtf_alignment']} across {', '.join(mtf['timeframes_analyzed'])}")
                    for tf, tf_result in mtf["timeframes"].items():
                        if "error" not in tf_result:
                            sig = tf_result["signal"]
                            print(f"        {tf:4s}: {sig['signal_type']:8s} ({sig['strength']}) RSI: {tf_result['indicators']['rsi']}")
            else:
                result = analyze_symbol(sym, args.detail, timeframe=args.timeframe)
                results.append(result)

                if not args.json:
                    if args.signals_only and result["signal"]["signal_type"] == "neutral":
                        continue
                    print_signal(result, detail=args.detail)
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})
            if not args.json:
                print(f"\n⚠️  {sym}: {e}")

    if args.json:
        output = {"results": results, "errors": errors, "timestamp": datetime.now().isoformat()}
        print(json.dumps(output, indent=2))
    else:
        # Summary
        bullish = [r for r in results if r["signal"]["signal_type"] == "bullish"]
        bearish = [r for r in results if r["signal"]["signal_type"] == "bearish"]
        strong_bull = [r for r in bullish if r["signal"]["strength"] == "strong"]
        strong_bear = [r for r in bearish if r["signal"]["strength"] == "strong"]

        print(f"\n{'=' * 60}")
        print(f"  SUMMARY: {len(bullish)} bullish ({len(strong_bull)} strong) | {len(bearish)} bearish ({len(strong_bear)} strong) | {len(results) - len(bullish) - len(bearish)} neutral")
        if errors:
            print(f"  ⚠️  {len(errors)} errors: {', '.join(e['symbol'] for e in errors)}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
