#!/usr/bin/env python3
"""
Wolf data job: free market data for the Wolf's Morning Brief.

Runs on GitHub Actions every weekday before the open (see wolf-data.yml) and pulls from
Yahoo Finance through yfinance (free, for personal use):
  - two years of daily prices for your focus stocks and the market gauges
  - price history for each stock: 1-day / 1-month / YTD / 1-year change, 52-week range,
    50- and 200-day averages, RSI, ATR, support and resistance, the last earnings report
    and the stock's move on it, and the next earnings date
  - option chains: the next two monthly expirations 30-95 days out, strikes within 15%
    of the price, with bid, ask, last, implied volatility, open interest and volume

It writes three files the Wolf reads each morning:
  data/summary.md    the numbers as tables (also easy for you to read on GitHub)
  data/metrics.json  the same numbers as data
  data/options.json  the option chains
"""
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:
    NY = timezone(timedelta(hours=-5))

# ═══════════════════════════ Your focus list: edit freely ═══════════════════════════
SECTORS = {
    "Big tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "AMD", "ORCL", "TSLA", "PLTR", "CRM"],
    "Big pharma": ["LLY", "NVO", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "AZN", "GILD", "VRTX"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "OXY", "SLB", "MPC", "PSX", "CEG", "VST"],
}
GAUGES = {
    "NQ=F": "Nasdaq-100 futures", "ES=F": "S&P 500 futures", "^NDX": "Nasdaq-100", "^IXIC": "Nasdaq Composite",
    "QQQ": "QQQ", "XLK": "Tech ETF (XLK)", "XLV": "Health care ETF (XLV)", "XLE": "Energy ETF (XLE)",
    "^TNX": "10-year yield", "CL=F": "WTI crude", "^VIX": "VIX",
}
STRIKE_WINDOW = 0.15          # keep strikes within ±15% of the stock price
MIN_DTE, MAX_DTE = 30, 95     # keep expirations this many days out
OUT_DIR = "data"
# ═════════════════════════════════════════════════════════════════════════════════════

NOW = datetime.now(NY)
TODAY = NOW.date()
FIELDS = ["strike", "bid", "ask", "last", "iv", "oi", "volume"]


def log(msg):
    print(f"[{datetime.now(NY):%H:%M:%S}] {msg}", flush=True)


def num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def rnd(x, d=2):
    x = num(x)
    return None if x is None else round(x, d)


def whole(x):
    x = num(x)
    return None if x is None else int(x)


def pct(x, d=1):
    x = num(x)
    return "n/a" if x is None else f"{x:+.{d}f}%"


def usd(x):
    x = num(x)
    return "n/a" if x is None else f"${x:,.2f}"


def call_yahoo(fn, what):
    """Call Yahoo once more after a pause if it rate-limits; return None on failure."""
    for attempt in (1, 2):
        try:
            return fn()
        except Exception as e:
            text = str(e).lower()
            if attempt == 1 and ("rate" in text or "too many" in text):
                log(f"   {what}: rate-limited, waiting 30s")
                time.sleep(30)
                continue
            log(f"   {what}: unavailable ({type(e).__name__}: {str(e)[:120]})")
            return None


# ──────────────────────────────── prices and history ────────────────────────────────
def download(symbols):
    raw = call_yahoo(lambda: yf.download(symbols, period="2y", interval="1d", group_by="ticker",
                                         auto_adjust=True, progress=False, threads=True), "price history")
    frames = {}
    if raw is None or raw.empty:
        return frames
    for sym in symbols:
        try:
            df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=["Close"])
        except KeyError:
            continue
        if len(df) >= 2:
            frames[sym] = df
    return frames


def metrics(df):
    c, h, l, v = (df[k].astype(float) for k in ("Close", "High", "Low", "Volume"))
    last = float(c.iloc[-1])

    def chg(n):
        return rnd((last / float(c.iloc[-1 - n]) - 1) * 100) if len(c) > n else None

    def sma(n):
        return float(c.tail(n).mean()) if len(c) >= n else None

    delta = c.diff()
    gain = float(delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    loss = float((-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    true_range = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(true_range.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    prior_year = c[c.index.year < c.index[-1].year]
    s50, s200 = sma(50), sma(200)
    hi52, lo52 = float(h.tail(252).max()), float(l.tail(252).min())
    avg_vol = float(v.iloc[-21:-1].mean()) if len(v) > 21 else 0.0
    return {
        "asof": c.index[-1].date().isoformat(), "close": round(last, 2), "prev_close": rnd(c.iloc[-2]),
        "chg_1d": chg(1), "chg_5d": chg(5), "chg_1m": chg(21), "chg_3m": chg(63), "chg_1y": chg(252),
        "chg_ytd": rnd((last / float(prior_year.iloc[-1]) - 1) * 100) if len(prior_year) else None,
        "high_52w": round(hi52, 2), "low_52w": round(lo52, 2), "from_high_pct": rnd((last / hi52 - 1) * 100),
        "sma50": rnd(s50), "sma200": rnd(s200),
        "vs_sma50_pct": rnd((last / s50 - 1) * 100) if s50 else None,
        "vs_sma200_pct": rnd((last / s200 - 1) * 100) if s200 else None,
        "rsi14": round(rsi, 1), "atr14": round(atr, 2), "atr_pct": round(atr / last * 100, 2),
        "support_20d": round(float(l.tail(20).min()), 2), "resistance_20d": round(float(h.tail(20).max()), 2),
        "support_60d": round(float(l.tail(60).min()), 2), "resistance_60d": round(float(h.tail(60).max()), 2),
        "volume_vs_20d": rnd(float(v.iloc[-1]) / avg_vol) if avg_vol > 0 else None,
    }


# ────────────────────────────────────── earnings ──────────────────────────────────────
def earnings(ticker, df):
    """Last report (date, timing, EPS surprise, the stock's move) and the next report date."""
    out = {"last_earnings": None, "last_timing": None, "last_surprise_pct": None, "last_move_pct": None,
           "next_earnings": None, "next_in_days": None}
    ed = call_yahoo(lambda: ticker.get_earnings_dates(limit=12), f"{ticker.ticker} earnings dates")
    if ed is None or getattr(ed, "empty", True):
        return out
    ed = ed.sort_index()
    now = pd.Timestamp.now(tz="America/New_York")
    upcoming = [ts for ts in ed.index if ts.date() >= TODAY]
    if upcoming:
        out["next_earnings"] = upcoming[0].date().isoformat()
        out["next_in_days"] = (upcoming[0].date() - TODAY).days
    reported = ed["Reported EPS"].notna() if "Reported EPS" in ed.columns else pd.Series(True, index=ed.index)
    past = ed[(ed.index < now) & reported.values]
    if past.empty:
        return out
    ts, row = past.index[-1], past.iloc[-1]
    timing = "after close" if ts.hour >= 12 else ("before open" if ts.hour > 0 else "time n/a")
    out.update(last_earnings=ts.date().isoformat(), last_timing=timing,
               last_surprise_pct=rnd(row.get("Surprise(%)")))
    days, closes = list(df.index.date), df["Close"].astype(float).tolist()
    report = ts.date()
    before = [i for i, d in enumerate(days) if d < report]
    on_or_after = [i for i, d in enumerate(days) if d >= report]
    after = [i for i, d in enumerate(days) if d > report]
    if timing == "before open" and before and on_or_after:
        a, b = before[-1], on_or_after[0]          # prior close → close on report day
    elif timing == "after close" and after:
        a, b = after[0] - 1, after[0]              # report-day close → next close
    elif before and after:
        a, b = before[-1], after[0]                # unknown time: the two-day window around it
    else:
        return out
    if 0 <= a < b < len(closes):
        out["last_move_pct"] = rnd((closes[b] / closes[a] - 1) * 100)
    return out


# ──────────────────────────────────── option chains ────────────────────────────────────
def pick_expiries(expiries):
    window = []
    for text in expiries or ():
        try:
            d = date.fromisoformat(str(text))
        except ValueError:
            continue
        if MIN_DTE <= (d - TODAY).days <= MAX_DTE:
            window.append(d)
    monthly = [d for d in window if d.weekday() == 4 and 15 <= d.day <= 21]  # third Friday
    return sorted(monthly or window)[:2]


def chain_rows(frame, price):
    rows = []
    if frame is None or getattr(frame, "empty", True):
        return rows
    lo, hi = price * (1 - STRIKE_WINDOW), price * (1 + STRIKE_WINDOW)
    for _, r in frame.iterrows():
        k = num(r.get("strike"))
        if k is None or not lo <= k <= hi:
            continue
        rows.append([k, rnd(r.get("bid")), rnd(r.get("ask")), rnd(r.get("lastPrice")),
                     rnd(r.get("impliedVolatility"), 4), whole(r.get("openInterest")), whole(r.get("volume"))])
    return rows


def at_the_money_iv(calls, puts, price):
    """Average implied volatility of the call and put nearest the money (with a live bid)."""
    ivs = []
    for side in (calls, puts):
        live = [r for r in side if (r[1] or 0) > 0 and (r[4] or 0) > 0]
        if live:
            ivs.append(min(live, key=lambda r: abs(r[0] - price))[4])
    return round(sum(ivs) / len(ivs) * 100, 1) if ivs else None


def option_chains(ticker, price):
    expiries = call_yahoo(lambda: ticker.options, f"{ticker.ticker} expirations")
    out, atm_iv = [], None
    for d in pick_expiries(expiries):
        chain = call_yahoo(lambda d=d: ticker.option_chain(d.isoformat()), f"{ticker.ticker} {d} chain")
        if chain is None:
            continue
        calls, puts = chain_rows(chain.calls, price), chain_rows(chain.puts, price)
        if not calls and not puts:
            continue
        out.append({"expiry": d.isoformat(), "dte": (d - TODAY).days, "calls": calls, "puts": puts})
        if atm_iv is None:
            atm_iv = at_the_money_iv(calls, puts, price)
        time.sleep(0.3)
    return out, atm_iv


# ──────────────────────────────────────── output ────────────────────────────────────────
def gauge_value(sym, m):
    if sym == "^TNX":
        return f"{m['close']:.2f}%"
    if sym == "^VIX":
        return f"{m['close']:.2f}"
    return f"{m['close']:,.2f}"


def gauge_move(sym, m):
    if sym == "^TNX" and m.get("prev_close") is not None:
        return f"{(m['close'] - m['prev_close']) * 100:+.0f}bp"
    return pct(m["chg_1d"])


def summary_md(stocks, gauges, missing):
    asof = max((m["asof"] for m in stocks.values()), default="n/a")
    lines = [f"# Wolf data · {NOW:%A, %b} {NOW.day}, {NOW.year}",
             f"Generated {NOW:%Y-%m-%d %H:%M} ET. Stock prices are as of the {asof} close; futures are as of "
             "generation time. Source: Yahoo Finance via yfinance (free, personal use).", "",
             "## Gauges", "| Gauge | Symbol | Last | 1D | 5D | 1M |", "|---|---|---:|---:|---:|---:|"]
    for sym, name in GAUGES.items():
        m = gauges.get(sym)
        if m:
            lines.append(f"| {name} | {sym} | {gauge_value(sym, m)} | {gauge_move(sym, m)} | {pct(m['chg_5d'])} | "
                         f"{pct(m['chg_1m'])} |")
    for sector, tickers in SECTORS.items():
        lines += ["", f"## {sector}",
                  "| Ticker | Close | 1D | 1M | YTD | 1Y | 52-wk range (vs high) | vs 50d / 200d | RSI | ATR | "
                  "20-day support / resistance | Last earnings (move) | Next earnings | ATM IV |",
                  "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|"]
        for sym in tickers:
            m = stocks.get(sym)
            if not m:
                lines.append(f"| {sym} | n/a |  |  |  |  |  |  |  |  |  |  |  |  |")
                continue
            last = "n/a"
            if m["last_earnings"]:
                last = f"{m['last_earnings']} {m['last_timing']} ({pct(m['last_move_pct'])})"
            nxt = f"{m['next_earnings']} (in {m['next_in_days']}d)" if m["next_earnings"] else "n/a"
            iv = f"{m['atm_iv_pct']:.1f}%" if m.get("atm_iv_pct") is not None else "n/a"
            lines.append(
                f"| {sym} | {usd(m['close'])} | {pct(m['chg_1d'])} | {pct(m['chg_1m'])} | {pct(m['chg_ytd'])} | "
                f"{pct(m['chg_1y'])} | {usd(m['low_52w'])}-{usd(m['high_52w'])} ({pct(m['from_high_pct'])}) | "
                f"{pct(m['vs_sma50_pct'])} / {pct(m['vs_sma200_pct'])} | {m['rsi14']:.0f} | {m['atr_pct']:.1f}% | "
                f"{usd(m['support_20d'])} / {usd(m['resistance_20d'])} | {last} | {nxt} | {iv} |")
    if missing:
        lines += ["", f"No data today for: {', '.join(missing)}"]
    return "\n".join(lines) + "\n"


def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stocks = [s for group in SECTORS.values() for s in group]
    log(f"Pulling two years of prices for {len(stocks)} stocks and {len(GAUGES)} gauges...")
    frames = download(stocks + list(GAUGES))
    if not any(s in frames for s in stocks):
        sys.exit("No price data came back from Yahoo Finance; keeping the previous files.")

    data, chains, missing = {}, {}, []
    for sym in stocks:
        df = frames.get(sym)
        if df is None or len(df) < 30:
            missing.append(sym)
            continue
        m = metrics(df)
        ticker = yf.Ticker(sym)
        m.update(earnings(ticker, df))
        expirations, m["atm_iv_pct"] = option_chains(ticker, m["close"])
        if expirations:
            chains[sym] = {"price": m["close"], "expirations": expirations}
        data[sym] = m
        log(f"   {sym}: close {usd(m['close'])}, {len(expirations)} expirations")
        time.sleep(0.4)
    gauges = {s: metrics(frames[s]) for s in GAUGES if s in frames and len(frames[s]) >= 30}

    stamp = NOW.isoformat(timespec="minutes")
    asof = max((m["asof"] for m in data.values()), default=None)
    write(os.path.join(OUT_DIR, "summary.md"), summary_md(data, gauges, missing))
    write(os.path.join(OUT_DIR, "metrics.json"), json.dumps(
        {"generated": stamp, "asof": asof, "sectors": SECTORS, "stocks": data, "gauges": gauges,
         "missing": missing}, indent=1))
    write(os.path.join(OUT_DIR, "options.json"), json.dumps(
        {"generated": stamp, "asof": asof, "fields": FIELDS, "chains": chains}, separators=(",", ":")))
    log(f"Done: {len(data)} stocks, {len(chains)} with option chains, missing: {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
