import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Dynamic universe loaders ────────────────────────────────────────────
#
# Active markets: US stocks + Gold proxies.
# Thailand loader (load_th_universe) is kept for optional future use but
# is NOT called anywhere in the active scan flow.
#
#   US   -> Nasdaq FTP trader directory (nasdaqlisted + otherlisted)
#   Gold -> built-in proxy list (always succeeds)

import re as _re

# ── Shared helper ─────────────────────────────────────────────

def _dedup(symbols: list[str]) -> list[str]:
    """Deduplicate while preserving order; strip and uppercase each entry."""
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── US universe ─────────────────────────────────────────────────

# Nasdaq trader FTP -- tab/pipe-separated flat files updated nightly.
# Source: https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _parse_nasdaq_listed(text: str) -> list[str]:
    """
    Parse nasdaqlisted.txt (pipe-delimited).
    Columns: Symbol | Security Name | Market Category | Test Issue |
             Financial Status | Round Lot Size | ETF | NextShares
    Keep real stocks: not a test issue, not an ETF, symbol is 1-5 plain letters.
    """
    symbols: list[str] = []
    for line in text.strip().splitlines()[1:]:   # skip header row
        parts = line.split("|")
        if len(parts) < 7:
            continue
        sym, test_issue, etf = parts[0].strip().upper(), parts[3].strip().upper(), parts[6].strip().upper()
        if test_issue == "Y" or etf == "Y":
            continue
        if _re.fullmatch(r"[A-Z]{1,5}", sym):
            symbols.append(sym)
    return symbols


def _parse_other_listed(text: str) -> list[str]:
    """
    Parse otherlisted.txt (pipe-delimited).
    Columns: ACT Symbol | Security Name | Exchange | CQS Symbol |
             ETF | Round Lot Size | Test Issue | NASDAQ Symbol
    Keep real stocks: not a test issue, not an ETF, symbol is 1-5 plain letters.
    """
    symbols: list[str] = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 7:
            continue
        sym, etf, test_issue = parts[0].strip().upper(), parts[4].strip().upper(), parts[6].strip().upper()
        if test_issue == "Y" or etf == "Y":
            continue
        if _re.fullmatch(r"[A-Z]{1,5}", sym):
            symbols.append(sym)
    return symbols


def load_us_universe() -> list[str]:
    """
    Download the Nasdaq trader directory files and return clean US common-stock
    symbols (NASDAQ + NYSE/AMEX, no ETFs, no test issues, no warrants).
    Falls back to an empty list on any error so other markets still run.
    """
    symbols: list[str] = []

    for url, parser, label in [
        (_NASDAQ_LISTED_URL, _parse_nasdaq_listed, "NASDAQ"),
        (_OTHER_LISTED_URL,  _parse_other_listed,  "NYSE/AMEX"),
    ]:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            batch = parser(resp.text)
            symbols.extend(batch)
            print(f"  \u2714  {label}: {len(batch)} symbols fetched")
        except Exception as exc:
            print(f"  \u26a0  Could not load {label} symbols: {exc}")

    symbols = _dedup(symbols)
    if symbols:
        print(f"  \U0001f4ca US universe: {len(symbols)} unique common-stock symbols")
    else:
        print("  \u26a0  US universe empty -- all Nasdaq directory fetches failed")
    return symbols


# ── Thai universe ──────────────────────────────────────────────────

# The SET public API blocks automated requests (403 Forbidden), so Thai symbols
# are loaded from a local plain-text file instead.
#
# Configure in .env:
#   TH_SYMBOL_FILE=symbols/th.txt
#
# File format: one Yahoo Finance ticker per line, .BK suffix required.
# Example lines:
#   PTT.BK
#   ADVANC.BK
#   CPALL.BK
# Blank lines and lines starting with # are ignored.


def load_th_universe() -> list[str]:
    """
    Load Thai equity symbols from a local text file configured via TH_SYMBOL_FILE
    in .env.  Each line must be a Yahoo Finance ticker with the .BK suffix.

    NOTE: This function is kept for optional future use but is NOT called anywhere
    in the active scan flow. To re-enable Thailand, add load_th_universe() back
    to build_base_universe().
    """
    path = os.getenv("TH_SYMBOL_FILE", "").strip()
    if not path:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except (FileNotFoundError, OSError):
        return []

    valid = [sym for sym in raw if sym and sym.endswith(".BK")]
    return _dedup(valid)


# ── Gold / commodity universe ──────────────────────────────────

# Small built-in list of well-known gold and precious-metal proxies available
# on Yahoo Finance. Static -- no network call required.
_GOLD_SYMBOLS = ["GLD", "IAU", "GC=F", "GOLD", "SLV", "SI=F"]


def load_gold_universe() -> list[str]:
    """Return a built-in list of gold / precious-metal proxy tickers.
    Always succeeds -- no network call required."""
    symbols = _dedup(_GOLD_SYMBOLS)
    print(f"  \U0001f4ca Gold/Commodity universe: {len(symbols)} built-in symbols")
    return symbols


# ── Universe builder ──────────────────────────────────────────────

def build_base_universe() -> list[str]:
    """
    Assemble the active universe: US stocks + Gold proxies.
    Applies configurable size caps so large markets don't overwhelm the server.

    Cap resolution order (highest priority first):
      1. MAX_US_SYMBOLS / MAX_TOTAL_SYMBOLS  -- explicit overrides if not None
      2. UNIVERSE_SIZE_MODE preset           -- "small" / "medium" / "large" / "full"
    """
    print("\n  \U0001f50d Building symbol universe...")

    # Resolve effective caps
    mode_caps  = _UNIVERSE_MODE_CAPS.get(UNIVERSE_SIZE_MODE, _UNIVERSE_MODE_CAPS["medium"])
    us_cap     = MAX_US_SYMBOLS    if MAX_US_SYMBOLS    is not None else mode_caps["us"]
    total_cap  = MAX_TOTAL_SYMBOLS if MAX_TOTAL_SYMBOLS is not None else mode_caps["total"]

    # Load raw universes
    us_raw    = load_us_universe()
    gold_syms = load_gold_universe()

    print(f"  \U0001f4ca Raw US symbols: {len(us_raw)}")

    # Cap US universe before combining
    if us_cap is not None and len(us_raw) > us_cap:
        us_syms = us_raw[:us_cap]
        print(f"  \u2702  US universe capped: {len(us_raw)} \u2192 {len(us_syms)} "
              f"(mode={UNIVERSE_SIZE_MODE}, MAX_US_SYMBOLS={us_cap})")
    else:
        us_syms = us_raw
        print(f"  \u2714  US universe: {len(us_syms)} symbols (no cap applied)")

    print(f"  \U0001f4ca Gold symbols: {len(gold_syms)}")

    # Combine and apply total cap
    combined = _dedup(us_syms + gold_syms)
    print(f"  \U0001f4ca Combined universe (pre-cap): {len(combined)} unique symbols")

    if total_cap is not None and len(combined) > total_cap:
        combined = combined[:total_cap]
        print(f"  \u2702  Total universe capped: \u2192 {len(combined)} "
              f"(MAX_TOTAL_SYMBOLS={total_cap})")

    if combined:
        print(f"  \u2705 Final universe: {len(combined)} symbols ready for scanning")
    else:
        print("  \u274c All universe loaders returned empty -- cannot proceed")

    return combined

# ── Universe & prefilter config ───────────────────────────────────────────────

# Prefilter thresholds
MIN_PRICE = 10.0                   # minimum close price in USD
MIN_AVG_TRADED_VALUE = 50_000_000  # minimum 20-day avg (close * volume) in USD
REQUIRE_ABOVE_SMA50 = False        # set True to keep only stocks in uptrend

# Value prefilter thresholds (applied to stocks only; gold/commodity symbols exempt)
# Set any threshold to None to disable that individual check.
MAX_PE         = 50.0              # trailing P/E upper bound  (None = skip check)
MAX_PB         = 10.0              # price-to-book upper bound (None = skip check)
MIN_MARKET_CAP = 500_000_000       # minimum market cap in USD (None = skip check)

# Symbols that are non-equity instruments and must skip the value filter
_GOLD_SYMBOL_SUFFIXES = {"=F", "=X"}          # futures / forex (GC=F, EURUSD=X)
_GOLD_EXACT_SYMBOLS   = {"GLD", "IAU", "GOLD", "SLV"}  # known ETF proxies

# Universe size control
# UNIVERSE_SIZE_MODE picks a preset; individual caps override the preset if set.
# Options: "small" | "medium" | "large" | "full"
UNIVERSE_SIZE_MODE = "medium"      # default: scan a manageable slice of the market

_UNIVERSE_MODE_CAPS: dict[str, dict] = {
    "small":  {"us": 50,  "total": 56},
    "medium": {"us": 100, "total": 106},
    "large":  {"us": 200, "total": 206},
    "full":   {"us": None, "total": None},   # no cap applied
}

# Override individual caps by setting these to an int (or None to use the mode).
MAX_US_SYMBOLS    = None           # int to override, None to use UNIVERSE_SIZE_MODE
MAX_TOTAL_SYMBOLS = None           # int to override, None to use UNIVERSE_SIZE_MODE

# Value-first selection
VALUE_CANDIDATE_POOL = 500         # max raw US symbols to score (bounds API calls)
TOP_VALUE_COUNT = 50               # top N stocks by value score to carry into scan

# Backtest config
BACKTEST_YEARS = 2                 # how many years of history to backtest over
BACKTEST_HOLD_DAYS = 10            # max holding period in trading days


# ── Data cache ────────────────────────────────────────────────────────────────
# Simple in-memory dict keyed by (symbol, period_key).
# Avoids re-downloading the same symbol multiple times in one run.

_DATA_CACHE: dict[str, pd.DataFrame] = {}

# Run diagnostics for wishlist visibility in console + email
LAST_WISHLIST_CONTEXT: dict = {}
LAST_PREFILTER_DIAGNOSTICS: dict = {}


def classify_market(symbol: str) -> str:
    """Classify wishlist symbol for reporting purposes."""
    return "TH" if symbol.upper().endswith(".BK") else "US"


def _join_symbols(symbols: list[str], limit: int = 30) -> str:
    """Join symbols for compact console / email display."""
    if not symbols:
        return "—"
    if len(symbols) <= limit:
        return ", ".join(symbols)
    head = ", ".join(symbols[:limit])
    return f"{head}, … (+{len(symbols) - limit} more)"


def _cached_download(symbol: str, period_key: str, **kwargs) -> pd.DataFrame:
    """Download via yfinance and cache result for the lifetime of the process."""
    cache_key = f"{symbol}::{period_key}"
    if cache_key not in _DATA_CACHE:
        df = yf.download(symbol, progress=False, auto_adjust=True, **kwargs)
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        _DATA_CACHE[cache_key] = df
    return _DATA_CACHE[cache_key]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(symbol: str) -> pd.DataFrame | None:
    end   = datetime.today()
    start = end - timedelta(days=365)
    df = _cached_download(symbol, "1y", start=start, end=end, interval="1d")
    if df.empty or len(df) < 60:
        return None
    return df.copy()


def load_data_long(symbol: str) -> pd.DataFrame | None:
    """Load 2 years of daily data — needed for SMA200 in mid/long-term signals."""
    end   = datetime.today()
    start = end - timedelta(days=730)
    df = _cached_download(symbol, "2y", start=start, end=end, interval="1d")
    if df.empty or len(df) < 200:
        return None
    return df.copy()


# ── Watchlist builder (prefilter stage) ──────────────────────────────────────

def build_watchlist_with_diagnostics(symbols: list[str]) -> tuple[list[str], dict]:
    """
    Download recent data for each symbol and apply quick prefilters.
    Returns (passed_symbols, diagnostics).

    Diagnostics makes it visible which wishlist symbols disappeared before scan.
    """
    passed: list[str] = []
    failed: list[str] = []
    skipped_no_data: list[str] = []
    fail_reason_by_symbol: dict[str, str] = {}

    print(f"\n  Running prefilters on {len(symbols)} symbols...")

    for symbol in symbols:
        df = _cached_download(symbol, "60d", period="60d", interval="1d")
        if df.empty or len(df) < 21:
            skipped_no_data.append(symbol)
            fail_reason_by_symbol[symbol] = "insufficient 60d data"
            print(f"  ⚠  {symbol}: insufficient 60d data for liquidity prefilter")
            continue

        close = float(df["Close"].iloc[-1])
        volume = df["Volume"].astype(float)
        traded_value_avg = float((df["Close"].astype(float) * volume).rolling(20).mean().iloc[-1])

        if close < MIN_PRICE:
            failed.append(symbol)
            fail_reason_by_symbol[symbol] = f"close ${close:.2f} < MIN_PRICE ${MIN_PRICE:.2f}"
            continue

        if traded_value_avg < MIN_AVG_TRADED_VALUE:
            failed.append(symbol)
            fail_reason_by_symbol[symbol] = (
                f"avg traded value ${traded_value_avg:,.0f} < MIN_AVG_TRADED_VALUE ${MIN_AVG_TRADED_VALUE:,.0f}"
            )
            continue

        if REQUIRE_ABOVE_SMA50:
            sma50 = df["Close"].astype(float).rolling(50).mean().iloc[-1]
            if pd.isna(sma50) or close < float(sma50):
                failed.append(symbol)
                fail_reason_by_symbol[symbol] = f"close ${close:.2f} below SMA50 ${float(sma50):.2f}"
                continue

        passed.append(symbol)

    diagnostics = {
        "input_symbols": list(symbols),
        "passed": passed,
        "failed": failed,
        "skipped_no_data": skipped_no_data,
        "fail_reason_by_symbol": fail_reason_by_symbol,
        "passed_us": [s for s in passed if classify_market(s) == "US"],
        "passed_th": [s for s in passed if classify_market(s) == "TH"],
        "failed_us": [s for s in failed if classify_market(s) == "US"],
        "failed_th": [s for s in failed if classify_market(s) == "TH"],
        "no_data_us": [s for s in skipped_no_data if classify_market(s) == "US"],
        "no_data_th": [s for s in skipped_no_data if classify_market(s) == "TH"],
    }

    global LAST_PREFILTER_DIAGNOSTICS
    LAST_PREFILTER_DIAGNOSTICS = diagnostics
    return passed, diagnostics


def build_watchlist(symbols: list[str]) -> list[str]:
    """Backward-compatible wrapper returning only passed symbols."""
    passed, _ = build_watchlist_with_diagnostics(symbols)
    return passed


# ── Value prefilter ──────────────────────────────────────────────────────────

def _is_gold_symbol(symbol: str) -> bool:
    """Return True for gold/commodity proxies that bypass value filtering."""
    if symbol in _GOLD_EXACT_SYMBOLS:
        return True
    for suffix in _GOLD_SYMBOL_SUFFIXES:
        if symbol.endswith(suffix):
            return True
    return False


# ── Value scoring ───────────────────────────────────────────────────

def compute_value_score(symbol: str) -> float | None:
    """
    Return a composite value score for a single equity symbol.
    Higher score = better value.

    Components (all optional; missing metrics are skipped gracefully):
      PE component  : 1 / trailingPE   (lower PE  -> higher score)
      PB component  : 1 / priceToBook  (lower PB  -> higher score)
      MCap bonus    : log10(marketCap) / 10  (large cap -> small bonus)

    Returns None if yfinance returns no usable price/fundamental data.
    Gold/commodity proxies always return None (they bypass value scoring).
    """
    if _is_gold_symbol(symbol):
        return None

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return None

    has_price = (
        info.get("regularMarketPrice") is not None
        or info.get("currentPrice") is not None
        or info.get("previousClose") is not None
    )
    if not info or not has_price:
        return None

    import math
    score = 0.0

    pe   = info.get("trailingPE")
    pb   = info.get("priceToBook")
    mcap = info.get("marketCap")

    if isinstance(pe, (int, float)) and not pd.isna(pe) and pe > 0:
        score += 1.0 / pe

    if isinstance(pb, (int, float)) and not pd.isna(pb) and pb > 0:
        score += 1.0 / pb

    if isinstance(mcap, (int, float)) and not pd.isna(mcap) and mcap > 0:
        score += math.log10(mcap) / 10.0

    return score


def select_top_value_symbols(
    us_symbols: list[str],
    gold_symbols: list[str],
) -> list[str]:
    """
    Score up to VALUE_CANDIDATE_POOL US symbols by fundamental value,
    keep the top TOP_VALUE_COUNT, then append gold proxies.

    Steps:
      1. Cap raw US list to VALUE_CANDIDATE_POOL to bound API calls.
      2. Score each candidate with compute_value_score().
      3. Sort descending by score; keep top TOP_VALUE_COUNT.
      4. Append gold symbols (always included, no scoring required).
    """
    raw_count  = len(us_symbols)
    candidates = us_symbols[:VALUE_CANDIDATE_POOL]

    print(f"\n  💰 Value scoring: {raw_count} raw US symbols  "
          f"→  scoring {len(candidates)} candidates "
          f"(pool cap={VALUE_CANDIDATE_POOL})...")

    scored: list[tuple[str, float]] = []
    no_data = 0

    for i, symbol in enumerate(candidates, 1):
        if i % 50 == 0:
            print(f"     ... scored {i}/{len(candidates)}")
        vs = compute_value_score(symbol)
        if vs is None:
            no_data += 1
        else:
            scored.append((symbol, vs))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_us = [sym for sym, _ in scored[:TOP_VALUE_COUNT]]

    print(f"  ✔  Scored: {len(scored)}  |  no-data: {no_data}  |  "
          f"top {len(top_us)} selected by value score")

    combined = _dedup(top_us + gold_symbols)
    print(f"  ✅ Value-selected universe: {len(combined)} symbols "
          f"({len(top_us)} value stocks + {len(gold_symbols)} gold proxies)")
    return combined


def build_value_watchlist(symbols: list[str]) -> list[str]:
    """
    Apply a fundamentals-based value prefilter to equity symbols.

    Gold/commodity proxies are passed through unchanged.
    For all other symbols, yfinance .info is queried for:
      - trailingPE   (trailing P/E ratio)
      - priceToBook  (price-to-book ratio)
      - marketCap    (total market cap in USD)

    If a symbol has no usable yfinance data it is skipped with a warning.
    If a specific metric is absent, that individual check is skipped gracefully.
    Thresholds: MAX_PE, MAX_PB, MIN_MARKET_CAP (set any to None to disable).
    """
    print(f"\n  💰 Value prefilter: checking {len(symbols)} symbols...")

    passed: list[str] = []
    skipped_no_data = 0
    skipped_failed  = 0

    for symbol in symbols:
        # Gold / commodity proxies are exempt from value filtering
        if _is_gold_symbol(symbol):
            passed.append(symbol)
            continue

        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info or {}
        except Exception as exc:
            print(f"  ⚠  {symbol}: could not fetch fundamentals ({exc}) -- skipped")
            skipped_no_data += 1
            continue

        # Empty or price-less info means the ticker is unknown to yfinance
        has_price = (
            info.get("regularMarketPrice") is not None
            or info.get("currentPrice") is not None
            or info.get("previousClose") is not None
        )
        if not info or not has_price:
            print(f"  ⚠  {symbol}: no fundamental data available -- skipped")
            skipped_no_data += 1
            continue

        pe   = info.get("trailingPE")
        pb   = info.get("priceToBook")
        mcap = info.get("marketCap")

        fail_reasons: list[str] = []

        # Trailing P/E check
        if MAX_PE is not None:
            if isinstance(pe, (int, float)) and not pd.isna(pe):
                if pe <= 0 or pe > MAX_PE:
                    fail_reasons.append(f"PE={pe:.1f} (max {MAX_PE})")
            # absent -- skip gracefully

        # Price-to-book check
        if MAX_PB is not None:
            if isinstance(pb, (int, float)) and not pd.isna(pb):
                if pb <= 0 or pb > MAX_PB:
                    fail_reasons.append(f"PB={pb:.1f} (max {MAX_PB})")
            # absent -- skip gracefully

        # Market cap check
        if MIN_MARKET_CAP is not None:
            if isinstance(mcap, (int, float)) and not pd.isna(mcap):
                if mcap < MIN_MARKET_CAP:
                    fail_reasons.append(
                        f"MCap={mcap/1e9:.2f}B (min {MIN_MARKET_CAP/1e9:.2f}B)"
                    )
            # absent -- skip gracefully

        if fail_reasons:
            print(f"  ❎  {symbol}: value filter failed -- {chr(44).join(fail_reasons)}")
            skipped_failed += 1
        else:
            passed.append(symbol)

    print(
        f"  ✅ Value prefilter done: {len(passed)}/{len(symbols)} passed"
        f"  (no-data: {skipped_no_data}, filtered out: {skipped_failed})"
    )
    return passed


# ── Indicators ────────────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()

    # RSI 14 — handle all zero-division edge cases:
    # both == 0 → flat → 50; loss only == 0 → pure uptrend → 100; gain only == 0 → pure downtrend → 0
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = np.where(
        (gain == 0) & (loss == 0), 50.0,
        np.where(loss == 0, 100.0,
        np.where(gain == 0, 0.0,
        100 - (100 / (1 + gain / loss))))
    )
    df["rsi"] = rsi

    # ATR 14
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Volume average 20
    df["vol_avg20"] = df["Volume"].rolling(20).mean()

    # 20-day high/low (excluding current bar)
    df["high20"] = df["Close"].shift(1).rolling(20).max()
    df["low20"]  = df["Close"].shift(1).rolling(20).min()

    return df


# ── Short-term signal (existing logic — unchanged) ────────────────────────────

def generate_signal(df: pd.DataFrame) -> tuple[str, float, dict, bool, bool, bool, bool, float]:
    row = df.iloc[-1]

    close    = float(row["Close"])
    sma20    = float(row["sma20"])
    sma50    = float(row["sma50"])
    rsi      = float(row["rsi"])      # raw — used for all logic and reason generation
    volume   = float(row["Volume"])
    vol_avg  = float(row["vol_avg20"])
    high20   = float(row["high20"])
    low20    = float(row["low20"])

    # All decision logic uses raw (unrounded) values
    trend_up     = sma20 > sma50
    breakout     = close > high20
    breakdown    = close < low20
    volume_ratio = volume / vol_avg if vol_avg > 0 else 0.0
    volume_spike = volume_ratio > 1.5

    if trend_up and breakout and volume_spike:
        signal = "BUY"
    elif not trend_up and breakdown:
        signal = "SELL"
    else:
        signal = "HOLD"

    score = calculate_score(signal, trend_up, breakout, breakdown, volume_spike, rsi)

    # Round only for display/output
    stats = {
        "close":        round(close, 2),
        "rsi":          round(rsi, 2),
        "volume_ratio": round(volume_ratio, 2),
        "atr":          round(float(row["atr"]), 2),
        "sma20":        round(sma20, 2),
        "sma50":        round(sma50, 2),
    }

    # rsi returned raw so build_reason() uses unrounded value
    return signal, score, stats, trend_up, breakout, breakdown, volume_spike, rsi


# ── Mid-term signal: SMA50 vs SMA200, no strict breakout ─────────────────────

def generate_mid_term_signal(df: pd.DataFrame) -> tuple[str, str]:
    """
    Mid-term signal based on the SMA50/SMA200 golden/death cross.
    No 20-day breakout requirement — trend alignment is sufficient.

    Returns (signal, reason) where signal is BUY / SELL / HOLD.
    """
    row = df.iloc[-1]

    close  = float(row["Close"])
    sma50  = float(row["sma50"])
    rsi    = float(row["rsi"])

    # SMA200 may be NaN if data is too short; treat as HOLD in that case
    sma200_val = row.get("sma200", np.nan)
    if pd.isna(sma200_val):
        return "HOLD", "insufficient data for SMA200"
    sma200 = float(sma200_val)

    above_sma200   = close > sma200
    sma50_above200 = sma50 > sma200
    golden_cross   = above_sma200 and sma50_above200
    death_cross    = (not above_sma200) and (not sma50_above200)

    if golden_cross and rsi > 45:
        signal = "BUY"
        parts = [f"SMA50 ${sma50:.2f} > SMA200 ${sma200:.2f}", "golden cross"]
        if rsi >= 50:
            parts.append(f"RSI {rsi:.1f} bullish")
        reason = " + ".join(parts)
    elif death_cross and rsi < 55:
        signal = "SELL"
        parts = [f"SMA50 ${sma50:.2f} < SMA200 ${sma200:.2f}", "death cross"]
        if rsi <= 50:
            parts.append(f"RSI {rsi:.1f} bearish")
        reason = " + ".join(parts)
    else:
        signal = "HOLD"
        if above_sma200 and not sma50_above200:
            reason = f"price above SMA200 ${sma200:.2f} but SMA50 still lagging"
        elif sma50_above200 and not above_sma200:
            reason = f"SMA50 above SMA200 but price ${close:.2f} pulled back below SMA200"
        else:
            reason = f"mixed trend — SMA50 ${sma50:.2f}, SMA200 ${sma200:.2f}"

    return signal, reason


# ── Long-term signal: broad trend structure via SMA200 ───────────────────────

def generate_long_term_signal(df: pd.DataFrame) -> tuple[str, str]:
    """
    Long-term signal based on price position relative to SMA200
    and the slope/direction of SMA200 itself.

    Returns (signal, reason) where signal is BUY / SELL / HOLD.
    """
    row = df.iloc[-1]

    close = float(row["Close"])
    rsi   = float(row["rsi"])

    sma200_val = row.get("sma200", np.nan)
    if pd.isna(sma200_val):
        return "HOLD", "insufficient data for SMA200"
    sma200 = float(sma200_val)

    # SMA200 slope: compare current value to 20 bars ago
    if len(df) >= 20:
        sma200_20ago_val = df["sma200"].iloc[-20]
        sma200_rising = (
            False if pd.isna(sma200_20ago_val)
            else float(sma200_20ago_val) < sma200
        )
    else:
        sma200_rising = False

    pct_above = (close - sma200) / sma200 * 100

    if close > sma200 and sma200_rising:
        signal = "BUY"
        reason = (
            f"price ${close:.2f} above rising SMA200 ${sma200:.2f}"
            f" (+{pct_above:.1f}%)"
        )
        if rsi > 60:
            reason += f", RSI {rsi:.1f} confirms"
    elif close < sma200 and not sma200_rising:
        signal = "SELL"
        reason = (
            f"price ${close:.2f} below falling SMA200 ${sma200:.2f}"
            f" ({pct_above:.1f}%)"
        )
        if rsi < 45:
            reason += f", RSI {rsi:.1f} confirms"
    elif close > sma200:
        signal = "HOLD"
        reason = (
            f"price above SMA200 ${sma200:.2f} (+{pct_above:.1f}%)"
            " but SMA200 not yet rising"
        )
    else:
        signal = "HOLD"
        reason = (
            f"price below SMA200 ${sma200:.2f} ({pct_above:.1f}%)"
            " but SMA200 not decisively falling"
        )

    return signal, reason


# ── Scoring ───────────────────────────────────────────────────────────────────

def calculate_score(signal: str, trend_up: bool, breakout: bool, breakdown: bool, volume_spike: bool, rsi: float) -> float:
    """
    Score is signal-aware so SELL setups can rank strongly alongside BUY setups.
    BUY side rewards uptrend conditions; SELL side rewards downtrend conditions.
    Both scales are symmetric so ranking across mixed signals is meaningful.
    """
    score = 0.0

    if signal == "BUY":
        if trend_up:            score += 1.0
        if breakout:            score += 1.0
        if volume_spike:        score += 1.0
        if 50 <= rsi <= 70:     score += 0.5   # healthy momentum
        if rsi > 80:            score -= 0.5   # overbought risk

    elif signal == "SELL":
        if not trend_up:        score += 1.0   # trend down confirms SELL
        if breakdown:           score += 1.0
        if volume_spike:        score += 0.5   # volume confirms selling pressure
        if rsi < 30:            score += 0.5   # oversold confirms weakness
        if rsi < 50:            score += 0.5   # bearish RSI zone
        if rsi > 70:            score -= 0.5   # contradicts SELL

    else:  # HOLD — score reflects proximity to a signal
        if trend_up:            score += 0.5
        if breakout:            score += 0.5
        if breakdown:           score -= 0.5
        if 50 <= rsi <= 70:     score += 0.25

    return score


# ── Reason builder ────────────────────────────────────────────────────────────

def build_reason(signal: str, trend_up: bool, breakout: bool, breakdown: bool, volume_spike: bool, rsi: float) -> str:
    # rsi is raw (unrounded) — all thresholds applied to actual value
    if signal == "BUY":
        parts = ["trend up", "breakout"]
        if volume_spike:        parts.append("volume spike")
        if rsi > 80:            parts.append("RSI overbought")
        elif 50 <= rsi <= 70:   parts.append("RSI healthy")
        return " + ".join(parts)

    if signal == "SELL":
        parts = ["trend down", "breakdown"]
        if volume_spike:        parts.append("volume spike")
        if rsi < 30:            parts.append("RSI oversold")
        elif rsi < 50:          parts.append("RSI bearish")
        elif rsi > 70:          parts.append("RSI overbought")
        return " + ".join(parts)

    return "no clear setup"


# ── Combined summary sentence ─────────────────────────────────────────────────

def build_combined_summary(
    short: str, mid: str, long_: str,
    symbol: str, close: float,
) -> str:
    """
    Generate one plain-English sentence summarising the alignment
    across all three timeframes.
    """
    signals = [short, mid, long_]
    buy_count  = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count == 3:
        return f"{symbol} is bullish across all timeframes — short, mid, and long-term trends aligned up."
    if sell_count == 3:
        return f"{symbol} is bearish across all timeframes — short, mid, and long-term trends aligned down."
    if buy_count == 2 and sell_count == 0:
        return f"{symbol} shows bullish momentum on 2 of 3 timeframes; near-term outlook positive."
    if sell_count == 2 and buy_count == 0:
        return f"{symbol} is under pressure on 2 of 3 timeframes; near-term outlook cautious."
    if short == "BUY" and long_ == "SELL":
        return f"{symbol} has a short-term bounce against a longer-term downtrend — countertrend setup, higher risk."
    if short == "SELL" and long_ == "BUY":
        return f"{symbol} is pulling back within a longer-term uptrend — potential re-entry zone for long-term bulls."
    if short == "BUY" and mid == "BUY":
        return f"{symbol} has short and mid-term bullish alignment; long-term trend still developing."
    if short == "SELL" and mid == "SELL":
        return f"{symbol} has short and mid-term bearish alignment; long-term trend still developing."
    return f"{symbol} shows mixed signals across timeframes — no strong directional consensus at ${close:.2f}."


# ── Explanation builder ───────────────────────────────────────────────────────

def build_explanation(
    signal: str,
    close: float,
    sma20: float,
    sma50: float,
    high20: float,
    low20: float,
    volume_ratio: float,
    rsi: float,
    atr: float,
    trend_up: bool,
    breakout: bool,
    breakdown: bool,
    volume_spike: bool,
) -> str:
    """
    Return a human-readable, value-rich explanation of why a signal was generated.
    Covers: triggered conditions, supporting metrics, and warning/weakness notes.
    All values passed in are raw (unrounded); rounding is done here for display only.
    """
    lines: list[str] = []
    atr_pct = (atr / close * 100) if close > 0 else 0.0

    # ── BUY ──────────────────────────────────────────────────────────────────
    if signal == "BUY":
        lines.append("TRIGGERED CONDITIONS:")
        lines.append(
            f"  • Uptrend confirmed: SMA20 ${sma20:.2f} > SMA50 ${sma50:.2f}"
            f" (gap {((sma20 - sma50) / sma50 * 100):.1f}%)"
        )
        lines.append(
            f"  • 20-day breakout: close ${close:.2f} > prior high ${high20:.2f}"
            f" (+{((close - high20) / high20 * 100):.1f}% above)"
        )
        lines.append(
            f"  • Volume spike: {volume_ratio:.2f}x avg"
            f" (threshold 1.5x) — buying pressure confirmed"
        )

        lines.append("SUPPORTING METRICS:")
        lines.append(f"  • RSI(14): {rsi:.1f}" + (
            " — momentum healthy, not overbought" if 50 <= rsi <= 70
            else " — neutral zone" if rsi < 50
            else " — elevated, watch for exhaustion"
        ))
        lines.append(f"  • ATR(14): ${atr:.2f} ({atr_pct:.1f}% of price) — daily range context")
        lines.append(f"  • 20-day range: ${low20:.2f} – ${high20:.2f}")

        warnings: list[str] = []
        if rsi > 80:
            warnings.append(f"RSI {rsi:.1f} is overbought — pullback risk elevated")
        if volume_ratio < 2.0:
            warnings.append(f"Volume {volume_ratio:.2f}x is marginal — breakout conviction moderate")
        if atr_pct > 4.0:
            warnings.append(f"ATR {atr_pct:.1f}% is high — expect wider intraday swings")
        if (close - high20) / high20 * 100 > 3.0:
            warnings.append("Price extended well above breakout level — consider waiting for a retest")
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ⚠ {w}")

    # ── SELL ─────────────────────────────────────────────────────────────────
    elif signal == "SELL":
        lines.append("TRIGGERED CONDITIONS:")
        lines.append(
            f"  • Downtrend confirmed: SMA20 ${sma20:.2f} < SMA50 ${sma50:.2f}"
            f" (gap {((sma50 - sma20) / sma50 * 100):.1f}%)"
        )
        lines.append(
            f"  • 20-day breakdown: close ${close:.2f} < prior low ${low20:.2f}"
            f" ({((low20 - close) / low20 * 100):.1f}% below)"
        )
        if volume_spike:
            lines.append(
                f"  • Volume spike: {volume_ratio:.2f}x avg — selling pressure confirmed"
            )

        lines.append("SUPPORTING METRICS:")
        lines.append(f"  • RSI(14): {rsi:.1f}" + (
            " — oversold, momentum fully bearish" if rsi < 30
            else " — bearish zone" if rsi < 50
            else " — above 50, trend weakness less clear" if rsi < 70
            else " — overbought despite downtrend, contradictory"
        ))
        lines.append(f"  • ATR(14): ${atr:.2f} ({atr_pct:.1f}% of price) — daily range context")
        lines.append(f"  • 20-day range: ${low20:.2f} – ${high20:.2f}")

        warnings: list[str] = []
        if rsi > 60:
            warnings.append(f"RSI {rsi:.1f} contradicts bearish signal — conviction reduced")
        if not volume_spike:
            warnings.append(f"Volume {volume_ratio:.2f}x is below 1.5x — breakdown lacks confirmation")
        if atr_pct > 4.0:
            warnings.append(f"ATR {atr_pct:.1f}% is high — volatile conditions, wider stops advised")
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ⚠ {w}")

    # ── HOLD ─────────────────────────────────────────────────────────────────
    else:
        lines.append("NO SIGNAL TRIGGERED:")
        if trend_up:
            lines.append(
                f"  • Trend is UP (SMA20 ${sma20:.2f} > SMA50 ${sma50:.2f})"
                " but no breakout yet"
            )
        else:
            lines.append(
                f"  • Trend is DOWN (SMA20 ${sma20:.2f} < SMA50 ${sma50:.2f})"
                " but no breakdown yet"
            )
        if not breakout and not breakdown:
            lines.append(
                f"  • Price ${close:.2f} is inside 20-day range"
                f" (${low20:.2f} – ${high20:.2f})"
            )
        lines.append(f"  • Volume ratio: {volume_ratio:.2f}x (need >1.5x for confirmation)")
        lines.append("SUPPORTING METRICS:")
        lines.append(f"  • RSI(14): {rsi:.1f}" + (
            " — bullish momentum" if rsi > 60
            else " — bearish momentum" if rsi < 40
            else " — neutral"
        ))
        lines.append(f"  • ATR(14): ${atr:.2f} ({atr_pct:.1f}% of price)")
        lines.append(f"  • 20-day range: ${low20:.2f} – ${high20:.2f}")

        dist_to_high = (high20 - close) / close * 100
        dist_to_low  = (close - low20)  / close * 100
        if dist_to_high < dist_to_low:
            lines.append(
                f"  • Closer to breakout level (${high20:.2f},"
                f" {dist_to_high:.1f}% away) than breakdown"
            )
        else:
            lines.append(
                f"  • Closer to breakdown level (${low20:.2f},"
                f" {dist_to_low:.1f}% away) than breakout"
            )

    return "\n".join(lines)


# ── Market scanner ────────────────────────────────────────────────────────────

def scan_market(symbols: list[str]) -> list[dict]:
    results = []

    print(f"\n  Running full scan on {len(symbols)} symbols...")

    for symbol in symbols:
        print(f"  Scanning {symbol}...")

        # Load 2-year data once — covers short, mid, and long-term calculations.
        # Short-term logic only uses the last ~60 bars so the longer window doesn't affect it.
        df = load_data_long(symbol)
        if df is None:
            # Fall back to 1-year data if 2-year history is unavailable
            df = load_data(symbol)
        if df is None:
            print(f"  ⚠ Skipping {symbol}: insufficient data")
            continue

        df = calculate_indicators(df)
        required_cols = ["sma20", "sma50", "rsi", "atr", "vol_avg20", "high20", "low20"]
        if df.iloc[-1][required_cols].isnull().any():
            print(f"  ⚠ Skipping {symbol}: NaN in indicators")
            continue

        # ── Short-term signal (existing logic — uses only recent bars) ───────
        signal, score, stats, trend_up, breakout, breakdown, volume_spike, rsi_raw = generate_signal(df)
        reason = build_reason(signal, trend_up, breakout, breakdown, volume_spike, rsi_raw)

        row_last = df.iloc[-1]
        volume_ratio_raw = (
            float(row_last["Volume"]) / float(row_last["vol_avg20"])
            if float(row_last["vol_avg20"]) > 0 else 0.0
        )

        explanation = build_explanation(
            signal       = signal,
            close        = float(row_last["Close"]),
            sma20        = float(row_last["sma20"]),
            sma50        = float(row_last["sma50"]),
            high20       = float(row_last["high20"]),
            low20        = float(row_last["low20"]),
            volume_ratio = volume_ratio_raw,
            rsi          = rsi_raw,
            atr          = float(row_last["atr"]),
            trend_up     = trend_up,
            breakout     = breakout,
            breakdown    = breakdown,
            volume_spike = volume_spike,
        )

        # ── Mid and long-term signals — same df, no extra download ───────────
        mid_signal,  mid_reason  = generate_mid_term_signal(df)
        long_signal, long_reason = generate_long_term_signal(df)

        combined_summary = build_combined_summary(
            short   = signal,
            mid     = mid_signal,
            long_   = long_signal,
            symbol  = symbol,
            close   = stats["close"],
        )

        results.append({
            "symbol":               symbol,
            "market":               classify_market(symbol),
            # Primary (short-term) signal — used for ranking and email sections
            "signal":               signal,
            "score":                score,
            "close":                stats["close"],
            "rsi":                  stats["rsi"],
            "volume_ratio":         stats["volume_ratio"],
            "reason":               reason,
            "explanation":          explanation,
            # Multi-timeframe signals
            "short_term_signal":    signal,
            "short_term_reason":    reason,
            "mid_term_signal":      mid_signal,
            "mid_term_reason":      mid_reason,
            "long_term_signal":     long_signal,
            "long_term_reason":     long_reason,
            "combined_summary":     combined_summary,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results  # full sorted list — callers apply their own slice if needed


# ── Email formatting ──────────────────────────────────────────────────────────

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pill(label: str, value: str, label_color: str = "#6b7280") -> str:
    """Inline metric pill: small label above, bold value below."""
    return (
        '<div style="display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:8px;padding:8px 14px;margin:4px 6px 4px 0;text-align:center;'
        'min-width:64px;vertical-align:top;">'
        f'<div style="font-size:10px;color:{label_color};text-transform:uppercase;'
        f'letter-spacing:.06em;margin-bottom:3px;font-weight:600;">{label}</div>'
        f'<div style="font-size:14px;font-weight:700;color:#1e293b;">{value}</div>'
        '</div>'
    )


def _timeframe_badge(label: str, signal: str, reason: str) -> str:
    """
    Render a single timeframe row: labelled badge (SHORT/MID/LONG) +
    signal pill + compact reason text.
    """
    colors = {
        "BUY":  ("#16a34a", "#dcfce7", "#bbf7d0"),
        "SELL": ("#dc2626", "#fee2e2", "#fecaca"),
        "HOLD": ("#64748b", "#f1f5f9", "#e2e8f0"),
    }
    fg, bg, border = colors.get(signal, colors["HOLD"])
    arrow = {"BUY": "▲", "SELL": "▼", "HOLD": "—"}.get(signal, "—")

    return (
        '<div style="display:flex;align-items:flex-start;gap:10px;'
        'margin-bottom:8px;flex-wrap:wrap;">'
        # timeframe label
        f'<span style="font-size:10px;font-weight:700;color:#94a3b8;'
        f'text-transform:uppercase;letter-spacing:.08em;min-width:40px;'
        f'padding-top:3px;">{label}</span>'
        # signal badge
        f'<span style="background:{bg};color:{fg};border:1px solid {border};'
        f'font-size:11px;font-weight:700;padding:2px 10px;border-radius:20px;'
        f'white-space:nowrap;">{arrow} {signal}</span>'
        # reason
        f'<span style="font-size:12px;color:#475569;flex:1;padding-top:2px;'
        f'line-height:1.4;">{reason}</span>'
        '</div>'
    )


def _explanation_rows(explanation: str, accent: str) -> str:
    """
    Convert the plain-text explanation into stacked HTML rows.
    Section headers become coloured label rows.
    Bullet lines (•) and warning lines (⚠) each get their own styled row.
    No tables used — pure div stacking.
    """
    section_styles = {
        "TRIGGERED CONDITIONS:": (accent,    "#fff"),
        "SUPPORTING METRICS:":   ("#1d4ed8", "#fff"),
        "WARNINGS:":             ("#92400e", "#fef3c7"),
        "NO SIGNAL TRIGGERED:":  ("#475569", "#f1f5f9"),
    }
    rows: list[str] = []
    for raw in explanation.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line in section_styles:
            fg, bg = section_styles[line]
            rows.append(
                f'<div style="background:{bg};color:{fg};font-size:10px;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em;padding:6px 12px;'
                f'border-radius:6px;margin:10px 0 4px;">{line}</div>'
            )
        elif line.startswith("⚠"):
            rows.append(
                f'<div style="color:#92400e;background:#fef3c7;font-size:13px;'
                f'padding:6px 12px;border-radius:6px;margin:3px 0;">{line}</div>'
            )
        elif line.startswith("•"):
            rows.append(
                f'<div style="color:#334155;font-size:13px;padding:3px 4px;'
                f'margin:2px 0;line-height:1.5;">{line}</div>'
            )
        else:
            rows.append(
                f'<div style="color:#64748b;font-size:12px;padding:2px 4px;">{line}</div>'
            )
    return "\n".join(rows)


def _signal_card(r: dict) -> str:
    """
    Render one white card for a single BUY or SELL signal.
    Layout (top to bottom):
      ① Header bar        — symbol | primary signal badge | score badge
      ② Combined summary  — one-sentence cross-timeframe overview
      ③ Timeframe badges  — SHORT / MID / LONG rows with signal + reason
      ④ Metric pills      — Close · RSI · Vol Ratio
      ⑤ Divider
      ⑥ Short-term detail — sectioned explanation rows
    """
    is_buy = r["signal"] == "BUY"

    accent     = "#16a34a" if is_buy else "#dc2626"
    accent_lt  = "#dcfce7" if is_buy else "#fee2e2"
    accent_mid = "#bbf7d0" if is_buy else "#fecaca"
    arrow      = "&#9650;" if is_buy else "&#9660;"

    # ① Header
    header = (
        '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">'
        f'<span style="font-size:22px;font-weight:800;color:#0f172a;letter-spacing:-.02em;">'
        f'{r["symbol"]}</span>'
        f'<span style="background:{accent};color:#fff;font-size:11px;font-weight:700;'
        f'padding:3px 11px;border-radius:20px;letter-spacing:.06em;">'
        f'{arrow}&nbsp;{r["signal"]}</span>'
        f'<span style="background:{accent_lt};color:{accent};font-size:11px;font-weight:700;'
        f'padding:3px 10px;border-radius:20px;border:1px solid {accent_mid};">'
        f'Score&nbsp;{r["score"]:+.1f}</span>'
        '</div>'
    )

    # ② Combined summary sentence
    summary_row = (
        f'<div style="font-size:13px;color:#334155;font-style:italic;'
        f'margin-bottom:14px;line-height:1.5;padding:8px 12px;'
        f'background:#f8fafc;border-radius:8px;border-left:3px solid {accent};">'
        f'{r.get("combined_summary", "")}</div>'
    )

    # ③ Timeframe badges block
    tf_block = (
        '<div style="background:#f8fafc;border-radius:10px;padding:12px 14px;margin-bottom:14px;">'
        '<div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
        'letter-spacing:.08em;margin-bottom:10px;">Timeframe Analysis</div>'
        + _timeframe_badge("Short", r.get("short_term_signal", r["signal"]), r.get("short_term_reason", r["reason"]))
        + _timeframe_badge("Mid",   r.get("mid_term_signal",   "HOLD"),       r.get("mid_term_reason",   "—"))
        + _timeframe_badge("Long",  r.get("long_term_signal",  "HOLD"),       r.get("long_term_reason",  "—"))
        + '</div>'
    )

    # ④ Metric pills
    pills = (
        _pill("Close",     f'${r["close"]:,.2f}') +
        _pill("RSI",       f'{r["rsi"]:.1f}') +
        _pill("Vol Ratio", f'{r["volume_ratio"]:.2f}x')
    )
    pills_row = f'<div style="margin-bottom:16px;">{pills}</div>'

    # ⑤ Divider
    divider = f'<hr style="border:none;border-top:1px solid {accent_mid};margin:0 0 12px;">'

    # ⑥ Short-term detail
    detail = _explanation_rows(r["explanation"], accent)
    detail_block = f'<div style="line-height:1.6;">{detail}</div>'

    card_inner = header + summary_row + tf_block + pills_row + divider + detail_block

    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;'
        f'border-top:4px solid {accent};border-radius:12px;'
        'padding:20px 22px;margin-bottom:20px;'
        'box-shadow:0 2px 8px rgba(0,0,0,.06);">'
        + card_inner +
        '</div>'
    )


def _stat_box(label: str, value: str | int, bg: str, label_color: str, value_color: str) -> str:
    """One summary stat box in the header row."""
    return (
        f'<div style="background:{bg};border-radius:10px;padding:14px 20px;'
        f'min-width:72px;text-align:center;flex:1 1 72px;">'
        f'<div style="font-size:10px;color:{label_color};text-transform:uppercase;'
        f'letter-spacing:.08em;font-weight:700;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:26px;font-weight:800;color:{value_color};">{value}</div>'
        '</div>'
    )


def _section_title(label: str, color: str) -> str:
    """Bold section heading separating BUY from SELL card groups."""
    return (
        f'<div style="margin:28px 0 14px;">'
        f'<span style="font-size:14px;font-weight:700;color:{color};'
        f'text-transform:uppercase;letter-spacing:.06em;'
        f'border-bottom:3px solid {color};padding-bottom:4px;">'
        f'{label}</span></div>'
    )


def _summary_card(title: str, lines: list[str]) -> str:
    rows = "".join(
        f'<div style="font-size:12px;color:#475569;line-height:1.6;margin:2px 0;">{line}</div>'
        for line in lines if line
    )
    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
        'padding:16px 18px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);">'
        f'<div style="font-size:13px;font-weight:800;color:#0f172a;text-transform:uppercase;'
        f'letter-spacing:.06em;margin-bottom:10px;">{title}</div>'
        f'{rows}'
        '</div>'
    )


def _compact_row(r: dict) -> str:
    sig_colors = {
        "BUY": ("#16a34a", "#dcfce7"),
        "SELL": ("#dc2626", "#fee2e2"),
        "HOLD": ("#64748b", "#f1f5f9"),
    }
    sfg, sbg = sig_colors.get(r.get("short_term_signal", "HOLD"), sig_colors["HOLD"])
    mfg, mbg = sig_colors.get(r.get("mid_term_signal", "HOLD"), sig_colors["HOLD"])
    lfg, lbg = sig_colors.get(r.get("long_term_signal", "HOLD"), sig_colors["HOLD"])
    return (
        '<div style="display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap;'
        'padding:10px 0;border-top:1px solid #e2e8f0;">'
        f'<div style="min-width:70px;font-size:13px;font-weight:800;color:#0f172a;">{r.get("symbol", "")}</div>'
        f'<div style="min-width:30px;font-size:11px;font-weight:700;color:#64748b;">{r.get("market", "")}</div>'
        f'<span style="background:{sbg};color:{sfg};padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;">S {r.get("short_term_signal", "HOLD")}</span>'
        f'<span style="background:{mbg};color:{mfg};padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;">M {r.get("mid_term_signal", "HOLD")}</span>'
        f'<span style="background:{lbg};color:{lfg};padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;">L {r.get("long_term_signal", "HOLD")}</span>'
        f'<span style="font-size:11px;color:#475569;">score {r.get("score", 0):+.1f}</span>'
        f'<span style="font-size:11px;color:#475569;">close ${r.get("close", 0):.2f}</span>'
        f'<div style="flex:1 1 100%;font-size:12px;color:#475569;line-height:1.5;">{r.get("reason", "")}</div>'
        '</div>'
    )


def _wishlist_summary_section(wishlist_context: dict, prefilter_diag: dict, total_scanned: int) -> str:
    if not wishlist_context and not prefilter_diag:
        return ""

    us_loaded = wishlist_context.get("us_symbols", [])
    th_loaded = wishlist_context.get("th_symbols", [])
    invalid_th = wishlist_context.get("invalid_th_symbols", [])

    passed_us = prefilter_diag.get("passed_us", [])
    passed_th = prefilter_diag.get("passed_th", [])
    failed_us = prefilter_diag.get("failed_us", [])
    failed_th = prefilter_diag.get("failed_th", [])
    no_data_us = prefilter_diag.get("no_data_us", [])
    no_data_th = prefilter_diag.get("no_data_th", [])

    lines = [
        f'US wishlist loaded: {len(us_loaded)}',
        f'TH wishlist loaded: {len(th_loaded)}',
        f'US passed liquidity filter: {len(passed_us)}',
        f'TH passed liquidity filter: {len(passed_th)}',
        f'Total scanned: {total_scanned}',
        f'US passed: {_join_symbols(passed_us, 20)}',
        f'TH passed: {_join_symbols(passed_th, 20)}',
        f'US filtered out: {_join_symbols(failed_us, 20)}',
        f'TH filtered out: {_join_symbols(failed_th, 20)}',
    ]

    if no_data_us or no_data_th:
        lines.append(f'No data / insufficient history: {_join_symbols(no_data_us + no_data_th, 20)}')
    if invalid_th:
        lines.append(f'Invalid TH symbols (missing .BK): {_join_symbols(invalid_th, 20)}')

    return _summary_card("Wishlist Coverage Summary", lines)


def _filtered_out_section(prefilter_diag: dict) -> str:
    if not prefilter_diag:
        return ""

    failed = prefilter_diag.get("failed", [])
    skipped = prefilter_diag.get("skipped_no_data", [])
    reasons = prefilter_diag.get("fail_reason_by_symbol", {})
    rows: list[str] = []

    for symbol in failed + skipped:
        rows.append(
            '<div style="padding:8px 0;border-top:1px solid #e2e8f0;">'
            f'<div style="font-size:12px;font-weight:700;color:#0f172a;">{symbol} '
            f'<span style="color:#64748b;font-weight:600;">({classify_market(symbol)})</span></div>'
            f'<div style="font-size:12px;color:#475569;line-height:1.5;">{reasons.get(symbol, "filtered out before scan")}</div>'
            '</div>'
        )

    if not rows:
        rows.append('<div style="font-size:12px;color:#475569;">No wishlist symbols were filtered out before scan.</div>')

    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
        'padding:16px 18px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);">'
        '<div style="font-size:13px;font-weight:800;color:#0f172a;text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:10px;">Wishlist Filtered Out Before Scan</div>'
        + "".join(rows) +
        '</div>'
    )


def _wishlist_scan_summary_section(results: list[dict]) -> str:
    if not results:
        return ""

    priority = {"BUY": 0, "SELL": 1, "HOLD": 2}
    ordered = sorted(
        results,
        key=lambda r: (
            0 if r.get("market") == "US" else 1,
            priority.get(r.get("short_term_signal", r.get("signal", "HOLD")), 2),
            -float(r.get("score", 0.0)),
            str(r.get("symbol", "")),
        ),
    )

    us_count = len([r for r in ordered if r.get("market") == "US"])
    th_count = len([r for r in ordered if r.get("market") == "TH"])
    hold_count = len([r for r in ordered if r.get("short_term_signal", r.get("signal", "HOLD")) == "HOLD"])
    rows = "".join(_compact_row(r) for r in ordered)
    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
        'padding:16px 18px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);">'
        '<div style="font-size:13px;font-weight:800;color:#0f172a;text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:10px;">All USA / Thai Wishlist Results (including HOLD)</div>'
        f'<div style="font-size:12px;color:#64748b;margin-bottom:8px;">{len(ordered)} scanned symbols rendered below · US {us_count} · TH {th_count} · HOLD {hold_count}</div>'
        '<div style="font-size:12px;color:#475569;margin-bottom:10px;line-height:1.6;">'
        'This section always shows every scanned wishlist symbol, even when the short-term signal is HOLD.'
        '</div>'
        + rows +
        '</div>'
    )


def format_email_html(
    results: list[dict],
    total_scanned: int,
    wishlist_context: dict | None = None,
    prefilter_diag: dict | None = None,
) -> str:
    """Build the full email HTML body with wishlist visibility diagnostics."""
    wishlist_context = wishlist_context or {}
    prefilter_diag = prefilter_diag or {}

    top_buys = sorted([r for r in results if r["signal"] == "BUY"], key=lambda r: r["score"], reverse=True)[:5]
    top_sells = sorted([r for r in results if r["signal"] == "SELL"], key=lambda r: r["score"], reverse=True)[:5]
    top_holds = sorted([r for r in results if r["signal"] == "HOLD"], key=lambda r: r["score"], reverse=True)[:5]

    if not top_buys and not top_sells and results:
        print("  ⚠ No BUY/SELL cards after ranking → HOLD fallback enabled")

    print("\n=== EMAIL DEBUG ===")
    print(f"Top BUY cards: {len(top_buys)}")
    print(f"Top SELL cards: {len(top_sells)}")
    print(f"Top HOLD cards: {len(top_holds)}")
    print(f"Wishlist summary rows: {len(results)}")

    buy_count = len([r for r in results if r["signal"] == "BUY"])
    sell_count = len([r for r in results if r["signal"] == "SELL"])
    hold_count = len([r for r in results if r["signal"] == "HOLD"])
    us_results = len([r for r in results if r.get("market") == "US"])
    th_results = len([r for r in results if r.get("market") == "TH"])
    date_str = datetime.today().strftime("%Y-%m-%d")

    page_open = (
        '<div style="background:#f1f5f9;padding:32px 16px;font-family:'
        'Arial,Helvetica,sans-serif;min-height:100%;">'
        '<div style="max-width:760px;margin:0 auto;">'
    )
    page_close = '</div></div>'

    header_card = (
        '<div style="background:#0f172a;border-radius:14px;padding:28px 28px 22px;'
        'margin-bottom:20px;">'
        '<div style="font-size:26px;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:4px;">&#128200; Stock Scanner</div>'
        '<div style="font-size:13px;color:#94a3b8;margin-bottom:2px;">'
        f'{date_str} &middot; Breakout + Trend Strategy</div>'
        '<div style="font-size:12px;color:#64748b;">'
        'Short · Mid · Long-term signals per symbol + wishlist coverage</div>'
        '</div>'
    )

    stat_row = (
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">'
        + _stat_box("Scanned", total_scanned, "#fff", "#64748b", "#0f172a")
        + _stat_box("BUY", buy_count, "#dcfce7", "#166534", "#16a34a")
        + _stat_box("SELL", sell_count, "#fee2e2", "#991b1b", "#dc2626")
        + _stat_box("HOLD", hold_count, "#fef9c3", "#854d0e", "#a16207")
        + _stat_box("US", us_results, "#eff6ff", "#1d4ed8", "#1d4ed8")
        + _stat_box("TH", th_results, "#f5f3ff", "#7c3aed", "#7c3aed")
        + '</div>'
    )

    coverage = _wishlist_summary_section(wishlist_context, prefilter_diag, total_scanned)

    if not results:
        empty = (
            '<div style="background:#fff;border-radius:12px;padding:40px 28px;'
            'text-align:center;color:#94a3b8;font-size:15px;margin-bottom:16px;">'
            'No scanned results were generated. Review the wishlist coverage summary below.</div>'
        )
        filtered = _filtered_out_section(prefilter_diag)
        footer = (
            f'<div style="text-align:center;font-size:11px;color:#94a3b8;'
            f'padding:24px 0 8px;">Generated by Stock Scanner &middot; {date_str}</div>'
        )
        return page_open + header_card + stat_row + coverage + empty + filtered + footer + page_close

    body = ""
    body += _wishlist_scan_summary_section(results)

    if top_buys:
        body += _section_title(f"&#9650; Top BUY Signals &nbsp;({len(top_buys)})", "#16a34a")
        body += "".join(_signal_card(r) for r in top_buys)
    if top_sells:
        body += _section_title(f"&#9660; Top SELL Signals &nbsp;({len(top_sells)})", "#dc2626")
        body += "".join(_signal_card(r) for r in top_sells)
    if not top_buys and not top_sells and top_holds:
        body += _section_title(f"&#8212; Top HOLD Signals &nbsp;({len(top_holds)})", "#64748b")
        body += "".join(_signal_card(r) for r in top_holds)

    body += _filtered_out_section(prefilter_diag)

    footer = (
        f'<div style="text-align:center;font-size:11px;color:#94a3b8;'
        f'padding:24px 0 8px;">Generated by Stock Scanner &middot; {date_str}</div>'
    )

    return page_open + header_card + stat_row + coverage + body + footer + page_close


# ── Email sender ──────────────────────────────────────────────────────────────

def _parse_email_list(env_value: str) -> list[str]:
    """Parse a comma-separated email string into a cleaned list, ignoring blanks."""
    return [addr.strip() for addr in env_value.split(",") if addr.strip()]


def send_email(subject: str, html_body: str) -> None:
    api_key    = os.getenv("RESEND_API_KEY", "")
    email_from = os.getenv("EMAIL_FROM", "")
    to_raw     = os.getenv("EMAIL_TO", "")
    bcc_raw    = os.getenv("EMAIL_BCC", "")

    to_list  = _parse_email_list(to_raw)
    bcc_list = _parse_email_list(bcc_raw)

    print("\n=== EMAIL CONFIG DEBUG ===")
    print("API KEY:", "OK" if api_key else "MISSING")
    print("FROM:", email_from or "MISSING")
    print("TO:", to_list if to_list else "MISSING")

    if not api_key or not email_from or not to_list:
        print("  ⚠ Email skipped: RESEND_API_KEY, EMAIL_FROM or EMAIL_TO not set in .env")
        return

    payload = {
        "from":    email_from,
        "to":      to_list,
        "subject": subject,
        "html":    html_body,
    }

    if bcc_list:
        payload["bcc"] = bcc_list

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=10,
        )
        if response.status_code in (200, 201):
            recipients = ", ".join(to_list)
            bcc_note   = f" (BCC: {', '.join(bcc_list)})" if bcc_list else ""
            print(f"  ✔ Email sent → {recipients}{bcc_note}")
        else:
            print(f"  ✖ Email failed: HTTP {response.status_code} — {response.text}")
    except requests.RequestException as e:
        print(f"  ✖ Email error: {e}")
    finally:
        print("  📨 Email process completed")


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(symbol: str) -> list[dict]:
    """
    Download BACKTEST_YEARS of daily history, replay the BUY signal logic bar
    by bar, and simulate trades.

    Entry  : next day's Open after a BUY signal fires
    Exit   : after BACKTEST_HOLD_DAYS trading days
             OR earlier if Close drops below SMA20

    Returns a list of trade dicts (one per completed trade).
    """
    end   = datetime.today()
    start = end - timedelta(days=int(BACKTEST_YEARS * 365.25))

    raw = _cached_download(symbol, f"bt_{BACKTEST_YEARS}y", start=start, end=end, interval="1d")
    if raw.empty or len(raw) < 60:
        return []

    df = calculate_indicators(raw).dropna(
        subset=["sma20", "sma50", "vol_avg20", "high20", "low20"]
    )
    df = df.reset_index()

    trades   = []
    in_trade = False
    entry_date = entry_price = hold_count = None

    for i in range(len(df) - 1):
        row = df.iloc[i]

        if in_trade:
            hold_count += 1
            close      = float(row["Close"])
            sma20      = float(row["sma20"])

            stop_hit   = close < sma20
            time_limit = hold_count >= BACKTEST_HOLD_DAYS

            if stop_hit or time_limit:
                exit_price = float(row["Close"])
                exit_date  = str(row["Date"])[:10]
                ret_pct    = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "symbol":      symbol,
                    "entry_date":  entry_date,
                    "entry_price": round(entry_price, 2),
                    "exit_date":   exit_date,
                    "exit_price":  round(exit_price, 2),
                    "return_pct":  round(ret_pct, 2),
                    "exit_reason": "SMA20 stop" if stop_hit else "time limit",
                })
                in_trade = False
            continue

        sma20        = float(row["sma20"])
        sma50        = float(row["sma50"])
        close        = float(row["Close"])
        volume       = float(row["Volume"])
        vol_avg      = float(row["vol_avg20"])
        high20       = float(row["high20"])

        trend_up     = sma20 > sma50
        breakout     = close > high20
        volume_ratio = volume / vol_avg if vol_avg > 0 else 0.0
        volume_spike = volume_ratio > 1.5

        if trend_up and breakout and volume_spike:
            next_row    = df.iloc[i + 1]
            entry_price = float(next_row["Open"])
            entry_date  = str(next_row["Date"])[:10]
            hold_count  = 0
            in_trade    = True

    return trades


def run_backtest_for_universe(symbols: list[str]) -> None:
    """
    Run run_backtest() for each symbol, aggregate results, and print a
    formatted summary with metrics and the top 5 best trades.
    """
    print("\n" + "=" * 55)
    print("  BACKTEST — Short-Term Breakout Strategy (LONG-ONLY)")
    print("  Uses short-term BUY signals only (SMA20 > SMA50 + breakout + volume).")
    print("  No mid-term or long-term filter applied. No short positions.")
    print(f"  Period: {BACKTEST_YEARS}y  |  Hold: up to {BACKTEST_HOLD_DAYS} days or SMA20 stop")
    print("=" * 55)

    all_trades: list[dict] = []

    for symbol in symbols:
        print(f"  Backtesting {symbol}...")
        trades = run_backtest(symbol)
        all_trades.extend(trades)

    if not all_trades:
        print("\n  No trades generated.")
        return

    returns = [t["return_pct"] for t in all_trades]

    total_trades = len(all_trades)
    wins         = sum(1 for r in returns if r > 0)
    win_rate     = wins / total_trades * 100
    avg_return   = sum(returns) / total_trades

    compound = 1.0
    for r in returns:
        compound *= (1 + r / 100)
    total_return = (compound - 1) * 100

    equity  = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100))
    peak   = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    print(f"""
=== BACKTEST RESULT — SHORT-TERM LONG-ONLY ===
  Symbols tested : {len(symbols)}
  Total trades   : {total_trades}
  Win rate       : {win_rate:.1f}%
  Avg return     : {avg_return:+.2f}%
  Total return   : {total_return:+.2f}%
  Max drawdown   : -{max_dd:.2f}%
""")

    top5 = sorted(all_trades, key=lambda t: t["return_pct"], reverse=True)[:5]
    print("  Top 5 best trades:")
    print(f"  {'Symbol':<8} {'Entry date':<12} {'Entry $':>8} {'Exit date':<12} {'Exit $':>8} {'Return':>8}  Exit reason")
    print("  " + "-" * 72)
    for t in top5:
        print(f"  {t['symbol']:<8} {t['entry_date']:<12} {t['entry_price']:>8.2f} "
              f"{t['exit_date']:<12} {t['exit_price']:>8.2f} {t['return_pct']:>+7.2f}%  {t['exit_reason']}")
    print()


# ── Wishlist-based universe loader ───────────────────────────────────────────
#
# Configure in .env:
#   US_WISHLIST=AAPL,MSFT,NVDA,TSLA,GOOGL
#   TH_WISHLIST=PTT.BK,ADVANC.BK,CPALL.BK
#   ENABLE_GOLD=true          # include built-in gold proxies (default: false)
#
# Rules:
#   - Symbols are comma-separated; extra whitespace is stripped automatically.
#   - All symbols are uppercased and deduplicated.
#   - TH symbols must use Yahoo Finance format with .BK suffix; a warning is
#     printed once per run for any TH symbol missing that suffix.
#   - If both wishlists are empty and ENABLE_GOLD is false, the function
#     returns [] and the caller exits gracefully.

def build_env_watchlist() -> list[str]:
    """
    Build a symbol list from .env wishlist variables and save diagnostics
    for later console + email reporting.
    """
    def _parse_env_list(raw: str) -> list[str]:
        return _dedup([s.strip().upper() for s in raw.split(",") if s.strip()])

    us_raw = os.getenv("US_WISHLIST", "").strip()
    us_syms = _parse_env_list(us_raw) if us_raw else []
    print(f"  📋 Loaded US wishlist: {len(us_syms)} symbols")
    if us_syms:
        print(f"     {_join_symbols(us_syms, 50)}")

    th_raw = os.getenv("TH_WISHLIST", "").strip()
    th_parsed = _parse_env_list(th_raw) if th_raw else []
    th_syms: list[str] = []
    invalid_th_symbols: list[str] = []
    for sym in th_parsed:
        if not sym.endswith(".BK"):
            invalid_th_symbols.append(sym)
            print(f"  ⚠  TH symbol '{sym}' is missing .BK suffix "
                  "— it may fail on Yahoo Finance. Add .BK to your TH_WISHLIST entry.")
        th_syms.append(sym)
    print(f"  📋 Loaded TH wishlist: {len(th_syms)} symbols")
    if th_syms:
        print(f"     {_join_symbols(th_syms, 50)}")
    if invalid_th_symbols:
        print(f"  ⚠  Invalid TH symbols: {_join_symbols(invalid_th_symbols, 50)}")

    enable_gold_raw = os.getenv("ENABLE_GOLD", "false").strip().lower()
    enable_gold = enable_gold_raw in {"true", "1", "yes"}
    if enable_gold:
        gold_syms = load_gold_universe()
        print(f"  📋 Gold enabled: {len(gold_syms)} built-in symbols added")
    else:
        gold_syms = []
        print("  📋 Gold skipped (set ENABLE_GOLD=true in .env to include)")

    combined = _dedup(us_syms + th_syms + gold_syms)

    global LAST_WISHLIST_CONTEXT
    LAST_WISHLIST_CONTEXT = {
        "us_symbols": us_syms,
        "th_symbols": th_syms,
        "gold_symbols": gold_syms,
        "invalid_th_symbols": invalid_th_symbols,
        "combined": combined,
    }

    print(f"  ✅ Combined wishlist: {len(combined)} unique symbols")
    return combined


# ── Entry points ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  📈 STOCK SCANNER — Breakout + Trend Strategy")
    print("=" * 55)
    print(f"  📅 Date: {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 55)

    print("\n  📝 Loading symbols from .env wishlist...")
    wishlist = build_env_watchlist()
    wishlist_context = LAST_WISHLIST_CONTEXT.copy()

    if not wishlist:
        print("  ❌ No symbols in wishlist. Exiting.")
        print("     Set US_WISHLIST, TH_WISHLIST, or ENABLE_GOLD=true in .env")
        return

    passed_watchlist, prefilter_diag = build_watchlist_with_diagnostics(wishlist)

    print(f"  📊 Wishlist: {len(wishlist)}  →  {len(passed_watchlist)} passed liquidity filter")
    print(f"  📌 Final watchlist ({len(passed_watchlist)} symbols): {_join_symbols(passed_watchlist, 80)}")

    print("\n=== WISHLIST DEBUG ===")
    print(f"Loaded US wishlist ({len(wishlist_context.get('us_symbols', []))}): {_join_symbols(wishlist_context.get('us_symbols', []), 80)}")
    print(f"Loaded TH wishlist ({len(wishlist_context.get('th_symbols', []))}): {_join_symbols(wishlist_context.get('th_symbols', []), 80)}")
    print(f"Passed US liquidity filter ({len(prefilter_diag.get('passed_us', []))}): {_join_symbols(prefilter_diag.get('passed_us', []), 80)}")
    print(f"Passed TH liquidity filter ({len(prefilter_diag.get('passed_th', []))}): {_join_symbols(prefilter_diag.get('passed_th', []), 80)}")
    print(f"Failed US liquidity filter ({len(prefilter_diag.get('failed_us', []))}): {_join_symbols(prefilter_diag.get('failed_us', []), 80)}")
    print(f"Failed TH liquidity filter ({len(prefilter_diag.get('failed_th', []))}): {_join_symbols(prefilter_diag.get('failed_th', []), 80)}")
    print(f"No-data US ({len(prefilter_diag.get('no_data_us', []))}): {_join_symbols(prefilter_diag.get('no_data_us', []), 80)}")
    print(f"No-data TH ({len(prefilter_diag.get('no_data_th', []))}): {_join_symbols(prefilter_diag.get('no_data_th', []), 80)}")
    if wishlist_context.get('invalid_th_symbols'):
        print(f"Invalid TH symbols ({len(wishlist_context.get('invalid_th_symbols', []))}): {_join_symbols(wishlist_context.get('invalid_th_symbols', []), 80)}")

    if not passed_watchlist:
        print("\n  ❌ No symbols passed liquidity prefilter. Exiting.")
        date_str = datetime.today().strftime("%Y-%m-%d")
        subject = f"Stock Scanner Report — {date_str}"
        html_body = format_email_html(
            [],
            total_scanned=0,
            wishlist_context=wishlist_context,
            prefilter_diag=prefilter_diag,
        )
        send_email(subject, html_body)
        return

    results = scan_market(passed_watchlist)

    print("\n=== SCAN DEBUG ===")
    print(f"Total results: {len(results)}")
    print(f"BUY count: {len([r for r in results if r['signal'] == 'BUY'])}")
    print(f"SELL count: {len([r for r in results if r['signal'] == 'SELL'])}")
    print(f"HOLD count: {len([r for r in results if r['signal'] == 'HOLD'])}")
    print(f"Final scanned US count: {len([r for r in results if r.get('market') == 'US'])}")
    print(f"Final scanned TH count: {len([r for r in results if r.get('market') == 'TH'])}")

    display_results = results[:10]

    print("\n📊 RESULTS — Top 10 by score (full list sent via email)\n")
    for r in display_results:
        signal_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r["signal"], "⚪")
        print(f"{signal_icon}  {r['symbol']:10s} {r.get('market', ''):2s}  "
              f"S:{r['short_term_signal']:4s}  M:{r['mid_term_signal']:4s}  L:{r['long_term_signal']:4s}  "
              f"score={r['score']:+.1f}  close={r['close']:>8.2f}  RSI={r['rsi']:5.1f}")
        print(f"         {r['combined_summary']}")
        print()

    print("-" * 55)
    print(f"  📊 Total results: {len(results)}  |  Showing top {len(display_results)} in console")
    print("  Full JSON output:\n")
    print(json.dumps(results, indent=2))

    print("\n  📧 Sending email notification...")
    date_str = datetime.today().strftime("%Y-%m-%d")
    subject = f"Stock Scanner Report — {date_str}"
    print(f"  📧 Email: passing {len(results)} results to formatter")
    html_body = format_email_html(
        results,
        total_scanned=len(passed_watchlist),
        wishlist_context=wishlist_context,
        prefilter_diag=prefilter_diag,
    )
    send_email(subject, html_body)

    # Backtest is available separately: python stock_scanner.py backtest


def main_backtest_only():
    """Run backtest against all loaded universe symbols without the live scan."""
    print("=" * 55)
    print("  BACKTEST MODE")
    print("=" * 55)

    base_universe = build_base_universe()

    if not base_universe:
        print("  ❌ No symbols loaded. Exiting.")
        print("     US and Gold loaders both failed. Check network access and try again.")
        return

    run_backtest_for_universe(base_universe)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        main_backtest_only()
    else:
        main()
