#!/usr/bin/env python3
"""
Backtesting Engine — Test TA strategies against historical data.

Simulates our signal generation against past price data to measure:
  - Win rate, average P&L, max drawdown
  - Which TA patterns are predictive
  - Optimal stop loss and target distances
  - Strategy performance by market regime

Usage:
    python3 backtester.py SPY                    # Backtest SPY with default strategy
    python3 backtester.py SPY XOM LMT --period 1y  # Multiple symbols, 1 year
    python3 backtester.py --watchlist iran        # Backtest Iran watchlist
    python3 backtester.py SPY --optimize          # Find optimal parameters
    python3 backtester.py SPY --json              # JSON output
"""

import json
import sys
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

sys.path.insert(0, str(Path(__file__).parent))
from ta_engine import WATCHLISTS


# ── Backtest Configuration ──────────────────────────────────

DEFAULT_CONFIG = {
    "initial_capital": 100000,
    "position_size_pct": 5.0,
    "stop_loss_atr_mult": 2.0,
    "target_atr_mult": 3.0,
    "min_rsi_oversold": 30,
    "max_rsi_overbought": 70,
    "require_ema_alignment": True,
    "require_above_200sma": True,
    "trailing_stop_pct": 3.0,
}


# ── Signal Generation (matches ta_engine logic) ────────────

def generate_signals_for_backtest(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Generate buy/sell signals across the entire dataframe."""
    if config is None:
        config = DEFAULT_CONFIG

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Calculate indicators
    ema9 = EMAIndicator(close, window=9).ema_indicator()
    ema21 = EMAIndicator(close, window=21).ema_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    sma200 = SMAIndicator(close, window=min(200, len(close) - 1)).sma_indicator()

    macd_ind = MACD(close)
    macd = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()

    rsi = RSIIndicator(close).rsi()
    adx = ADXIndicator(high, low, close).adx()
    atr = AverageTrueRange(high, low, close).average_true_range()

    bb = BollingerBands(close)
    bb_pct = bb.bollinger_pband()

    vol_sma = volume.rolling(20).mean()
    vol_ratio = volume / vol_sma

    # Build signal dataframe
    signals = pd.DataFrame(index=df.index)
    signals['close'] = close
    signals['atr'] = atr

    # Bullish conditions
    ema_bull = (ema9 > ema21) & (ema21 > sma50)
    above_200 = close > sma200
    macd_bull = (macd > macd_signal) & (macd_hist > 0)
    rsi_oversold = rsi < config["min_rsi_oversold"]
    rsi_strong = rsi > 55
    strong_trend = adx > 25
    bb_oversold = bb_pct < 0
    high_vol = vol_ratio > 1.5

    # Bearish conditions
    ema_bear = (ema9 < ema21) & (ema21 < sma50)
    below_200 = close < sma200
    macd_bear = (macd < macd_signal) & (macd_hist < 0)
    rsi_overbought = rsi > config["max_rsi_overbought"]

    # Count bullish/bearish reasons
    bull_count = (
        ema_bull.astype(int) +
        above_200.astype(int) +
        macd_bull.astype(int) +
        (rsi_oversold | rsi_strong).astype(int) +
        strong_trend.astype(int) +
        bb_oversold.astype(int)
    )

    bear_count = (
        ema_bear.astype(int) +
        below_200.astype(int) +
        macd_bear.astype(int) +
        rsi_overbought.astype(int) +
        strong_trend.astype(int)
    )

    # Generate signals: buy when 3+ bullish reasons, sell when 3+ bearish
    signals['bull_count'] = bull_count
    signals['bear_count'] = bear_count
    signals['signal'] = 0  # 1=buy, -1=sell, 0=none

    # Apply minimum requirements
    buy_mask = bull_count >= 3
    if config["require_ema_alignment"]:
        buy_mask = buy_mask & ema_bull
    if config["require_above_200sma"]:
        buy_mask = buy_mask & above_200

    sell_mask = bear_count >= 3

    signals.loc[buy_mask, 'signal'] = 1
    signals.loc[sell_mask, 'signal'] = -1

    return signals


# ── Trade Simulator ─────────────────────────────────────────

@dataclass
class BacktestTrade:
    entry_date: str
    exit_date: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    pl_pct: float
    pl_dollars: float
    exit_reason: str
    hold_days: int
    bull_count: int


def simulate_trades(symbol: str, signals: pd.DataFrame, config: dict = None) -> list[BacktestTrade]:
    """Simulate trades from signals with stop losses and targets."""
    if config is None:
        config = DEFAULT_CONFIG

    trades = []
    in_trade = False
    entry_price = 0
    entry_date = None
    stop_loss = 0
    target = 0
    high_water = 0
    bull_count = 0
    position_size = config["initial_capital"] * (config["position_size_pct"] / 100)

    for i in range(len(signals)):
        row = signals.iloc[i]
        date = str(signals.index[i])[:10]
        price = row['close']
        atr = row['atr'] if pd.notna(row['atr']) else 0

        if in_trade:
            # Update high water mark for trailing stop
            if price > high_water:
                high_water = price

            # Check trailing stop
            trail_stop = high_water * (1 - config["trailing_stop_pct"] / 100)

            # Check exit conditions
            exit_reason = None
            exit_price = price

            if price <= stop_loss:
                exit_reason = "stop_loss"
                exit_price = stop_loss
            elif price >= target:
                exit_reason = "target_hit"
                exit_price = target
            elif price <= trail_stop and price > entry_price:
                exit_reason = "trailing_stop"
                exit_price = price
            elif row['signal'] == -1:
                exit_reason = "bearish_signal"
                exit_price = price

            if exit_reason:
                pl_pct = ((exit_price - entry_price) / entry_price) * 100
                pl_dollars = position_size * (pl_pct / 100)
                hold_days = (signals.index[i] - signals.index[signals.index.get_loc(entry_date)]).days if entry_date in signals.index else 0

                trades.append(BacktestTrade(
                    entry_date=str(entry_date)[:10],
                    exit_date=date,
                    symbol=symbol,
                    side="long",
                    entry_price=round(entry_price, 2),
                    exit_price=round(exit_price, 2),
                    stop_loss=round(stop_loss, 2),
                    target=round(target, 2),
                    pl_pct=round(pl_pct, 2),
                    pl_dollars=round(pl_dollars, 2),
                    exit_reason=exit_reason,
                    hold_days=hold_days,
                    bull_count=bull_count,
                ))
                in_trade = False

        elif row['signal'] == 1 and not in_trade and atr > 0:
            # Enter trade
            entry_price = price
            entry_date = signals.index[i]
            stop_loss = price - config["stop_loss_atr_mult"] * atr
            target = price + config["target_atr_mult"] * atr
            high_water = price
            bull_count = int(row['bull_count'])
            in_trade = True

    return trades


# ── Performance Analysis ────────────────────────────────────

def analyze_performance(trades: list[BacktestTrade], initial_capital: float = 100000) -> dict:
    """Calculate performance metrics from backtest trades."""
    if not trades:
        return {"error": "No trades generated", "total_trades": 0}

    pls = [t.pl_pct for t in trades]
    dollars = [t.pl_dollars for t in trades]
    wins = [t for t in trades if t.pl_pct > 0]
    losses = [t for t in trades if t.pl_pct <= 0]

    # Equity curve
    equity = initial_capital
    peak = equity
    max_dd = 0
    equity_curve = [equity]
    for t in trades:
        equity += t.pl_dollars
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # By bull count
    by_confluence = {}
    for t in trades:
        bc = t.bull_count
        if bc not in by_confluence:
            by_confluence[bc] = {"trades": 0, "wins": 0, "total_pl": 0}
        by_confluence[bc]["trades"] += 1
        by_confluence[bc]["wins"] += int(t.pl_pct > 0)
        by_confluence[bc]["total_pl"] += t.pl_pct

    for bc in by_confluence:
        n = by_confluence[bc]["trades"]
        by_confluence[bc]["win_rate"] = round(by_confluence[bc]["wins"] / n * 100, 1)
        by_confluence[bc]["avg_pl"] = round(by_confluence[bc]["total_pl"] / n, 2)

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_pl_pct": round(sum(pls) / len(pls), 2),
        "total_pl_pct": round(sum(pls), 2),
        "total_pl_dollars": round(sum(dollars), 2),
        "avg_win_pct": round(sum(t.pl_pct for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss_pct": round(sum(t.pl_pct for t in losses) / len(losses), 2) if losses else 0,
        "largest_win": round(max(pls), 2),
        "largest_loss": round(min(pls), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_equity": round(equity_curve[-1], 2),
        "return_pct": round((equity_curve[-1] - initial_capital) / initial_capital * 100, 2),
        "avg_hold_days": round(sum(t.hold_days for t in trades) / len(trades), 1),
        "profit_factor": round(abs(sum(t.pl_pct for t in wins)) / abs(sum(t.pl_pct for t in losses)), 2) if losses and sum(t.pl_pct for t in losses) != 0 else float("inf"),
        "exit_reasons": exit_reasons,
        "by_confluence": by_confluence,
    }


# ── Display ─────────────────────────────────────────────────

def print_backtest(symbol: str, perf: dict, trades: list[BacktestTrade]):
    """Pretty-print backtest results."""
    print(f"\n{'=' * 70}")
    print(f"  BACKTEST: {symbol}")
    print(f"{'=' * 70}")

    if perf.get("error"):
        print(f"  ❌ {perf['error']}")
        return

    wr = perf["win_rate"]
    wr_icon = "🟢" if wr > 55 else "🟡" if wr > 45 else "🔴"

    print(f"\n  {wr_icon} Win Rate: {wr}% ({perf['wins']}W / {perf['losses']}L of {perf['total_trades']} trades)")
    print(f"  📊 Avg P&L: {perf['avg_pl_pct']:+.2f}% | Total: {perf['total_pl_pct']:+.2f}%")
    print(f"  💰 Final Equity: ${perf['final_equity']:,.2f} ({perf['return_pct']:+.2f}%)")
    print(f"  📉 Max Drawdown: {perf['max_drawdown_pct']:.2f}%")
    print(f"  ⚖️  Profit Factor: {perf['profit_factor']:.2f}")
    print(f"  📅 Avg Hold: {perf['avg_hold_days']:.1f} days")
    print(f"  🏆 Best: {perf['largest_win']:+.2f}% | Worst: {perf['largest_loss']:+.2f}%")
    print(f"  📊 Avg Win: {perf['avg_win_pct']:+.2f}% | Avg Loss: {perf['avg_loss_pct']:+.2f}%")

    # Exit reasons
    print(f"\n  Exit Reasons:")
    for reason, count in sorted(perf["exit_reasons"].items(), key=lambda x: x[1], reverse=True):
        pct = count / perf["total_trades"] * 100
        print(f"     {reason:20s}: {count:3d} ({pct:.0f}%)")

    # Confluence analysis
    if perf.get("by_confluence"):
        print(f"\n  Signal Confluence (bull_count):")
        for bc in sorted(perf["by_confluence"].keys()):
            data = perf["by_confluence"][bc]
            icon = "🟢" if data["win_rate"] > 55 else "🔴"
            print(f"     {icon} {bc} reasons: {data['win_rate']}% WR | {data['avg_pl']:+.2f}% avg | {data['trades']} trades")

    # Recent trades
    if trades:
        print(f"\n  Last 5 Trades:")
        for t in trades[-5:]:
            icon = "🟢" if t.pl_pct > 0 else "🔴"
            print(f"     {icon} {t.entry_date} → {t.exit_date}: {t.pl_pct:+.2f}% (${t.pl_dollars:+.2f}) [{t.exit_reason}]")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtesting Engine")
    parser.add_argument("symbols", nargs="*", help="Symbols to backtest")
    parser.add_argument("--watchlist", "-w", type=str, help="Use a predefined watchlist")
    parser.add_argument("--period", type=str, default="1y", help="Backtest period (6mo, 1y, 2y, 5y)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.watchlist:
        symbols = WATCHLISTS.get(args.watchlist, WATCHLISTS["default"])
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = ["SPY"]

    print(f"{'=' * 70}")
    print(f"  BACKTESTING ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Symbols: {', '.join(symbols)} | Period: {args.period}")
    print(f"{'=' * 70}")

    all_results = []

    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period=args.period)
            if df.empty or len(df) < 50:
                print(f"\n  ⚠️  {symbol}: Insufficient data")
                continue

            signals = generate_signals_for_backtest(df)
            trades = simulate_trades(symbol, signals)
            perf = analyze_performance(trades)

            if not args.json:
                print_backtest(symbol, perf, trades)

            all_results.append({
                "symbol": symbol,
                "performance": perf,
                "trades": [asdict(t) for t in trades],
            })
        except Exception as e:
            print(f"\n  ❌ {symbol}: {e}")

    if args.json:
        print(json.dumps(all_results, indent=2))

    # Summary across all symbols
    if len(all_results) > 1 and not args.json:
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY ACROSS ALL SYMBOLS")
        print(f"{'=' * 70}")
        for r in sorted(all_results, key=lambda x: x["performance"].get("return_pct", 0), reverse=True):
            p = r["performance"]
            icon = "🟢" if p.get("win_rate", 0) > 50 else "🔴"
            print(f"  {icon} {r['symbol']:6s}: {p.get('win_rate', 0):5.1f}% WR | {p.get('return_pct', 0):+7.2f}% return | {p.get('total_trades', 0)} trades | PF: {p.get('profit_factor', 0):.2f}")


if __name__ == "__main__":
    main()
