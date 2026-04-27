import os
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

MIN_HISTORY_DAYS = 260
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "true").strip().lower() in {"1", "true", "yes"}

# Reference valuation rules carried over from the main scanner.
MAX_PE = 50.0
MAX_PB = 10.0
MIN_MARKET_CAP = 500_000_000


# ── Helpers ────────────────────────────────────────────────────────────────

def _dedup(symbols: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _parse_env_list(raw: str) -> list[str]:
    return _dedup([s.strip().upper() for s in raw.split(",") if s.strip()])


def get_market(symbol: str) -> str:
    return "TH" if symbol.endswith(".BK") else "US"


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    return f"{float(value):,.{digits}f}"


def fmt_market_cap(value: float | int | None) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    v = float(value)
    if v >= 1_000_000_000_000:
        return f"{v/1_000_000_000_000:.2f}T"
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    return f"{v:,.0f}"


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ── Wishlist loader ────────────────────────────────────────────────────────

def build_wishlist() -> tuple[list[str], list[str], list[str]]:
    us = _parse_env_list(os.getenv("US_WISHLIST", ""))
    th_raw = _parse_env_list(os.getenv("TH_WISHLIST", ""))

    invalid_th: list[str] = []
    th: list[str] = []
    for s in th_raw:
        if not s.endswith(".BK"):
            invalid_th.append(s)
        th.append(s)

    wishlist = _dedup(us + th)

    print("\n=== WISHLIST ===")
    print(f"US ({len(us)}): {', '.join(us) if us else '-'}")
    print(f"TH ({len(th)}): {', '.join(th) if th else '-'}")
    if invalid_th:
        print(f"Invalid TH symbols (missing .BK): {', '.join(invalid_th)}")

    return wishlist, us, invalid_th


# ── Market data ────────────────────────────────────────────────────────────

def load_data(symbol: str) -> pd.DataFrame | None:
    end = datetime.today()
    start = end - timedelta(days=730)
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if len(df) < MIN_HISTORY_DAYS:
        return None
    return df.copy()


def load_fundamentals(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol).info or {}
        return info if isinstance(info, dict) else {}
    except Exception as exc:
        print(f"  - fundamentals unavailable for {symbol}: {exc}")
        return {}


# ── Indicators ─────────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()

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

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["high20"] = df["Close"].shift(1).rolling(20).max()
    df["low20"] = df["Close"].shift(1).rolling(20).min()
    return df


# ── Signals ────────────────────────────────────────────────────────────────

def calculate_score(signal: str, trend_up: bool, breakout: bool, breakdown: bool, volume_spike: bool, rsi: float) -> float:
    score = 0.0
    if signal == "BUY":
        if trend_up:
            score += 1.0
        if breakout:
            score += 1.0
        if volume_spike:
            score += 1.0
        if 50 <= rsi <= 70:
            score += 0.5
        if rsi > 80:
            score -= 0.5
    elif signal == "SELL":
        if not trend_up:
            score += 1.0
        if breakdown:
            score += 1.0
        if volume_spike:
            score += 0.5
        if rsi < 30:
            score += 0.5
        if rsi < 50:
            score += 0.5
        if rsi > 70:
            score -= 0.5
    else:
        if trend_up:
            score += 0.5
        if breakout:
            score += 0.5
        if breakdown:
            score -= 0.5
        if 50 <= rsi <= 70:
            score += 0.25
    return score


def build_reason(signal: str, trend_up: bool, breakout: bool, breakdown: bool, volume_spike: bool, rsi: float) -> str:
    if signal == "BUY":
        parts = ["trend up", "breakout"]
        if volume_spike:
            parts.append("volume spike")
        if rsi > 80:
            parts.append("RSI overbought")
        elif 50 <= rsi <= 70:
            parts.append("RSI healthy")
        return " + ".join(parts)
    if signal == "SELL":
        parts = ["trend down", "breakdown"]
        if volume_spike:
            parts.append("volume spike")
        if rsi < 30:
            parts.append("RSI oversold")
        elif rsi < 50:
            parts.append("RSI bearish")
        elif rsi > 70:
            parts.append("RSI overbought")
        return " + ".join(parts)
    return "no clear setup"


def build_short_explanation(
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
    atr_pct = (atr / close * 100) if close > 0 else 0.0

    if signal == "BUY":
        parts = [
            f"BUY rule hit: SMA20 {sma20:.2f} > SMA50 {sma50:.2f}",
            f"close {close:.2f} > prior 20d high {high20:.2f}",
            f"volume ratio {volume_ratio:.2f}x {'>' if volume_spike else '<='} 1.5x",
            f"RSI {rsi:.1f}",
            f"ATR {atr:.2f} ({atr_pct:.1f}% of price)",
        ]
        return " | ".join(parts)

    if signal == "SELL":
        parts = [
            f"SELL rule hit: SMA20 {sma20:.2f} < SMA50 {sma50:.2f}",
            f"close {close:.2f} < prior 20d low {low20:.2f}",
            f"volume ratio {volume_ratio:.2f}x",
            f"RSI {rsi:.1f}",
            f"ATR {atr:.2f} ({atr_pct:.1f}% of price)",
        ]
        return " | ".join(parts)

    hold_parts = [
        "HOLD because no full trigger",
        f"trend_up={'Y' if trend_up else 'N'}",
        f"breakout={'Y' if breakout else 'N'}",
        f"breakdown={'Y' if breakdown else 'N'}",
        f"volume_spike={'Y' if volume_spike else 'N'} ({volume_ratio:.2f}x)",
        f"RSI {rsi:.1f}",
        f"close {close:.2f} within 20d range {low20:.2f}-{high20:.2f}",
    ]
    return " | ".join(hold_parts)


def generate_short_term_signal(df: pd.DataFrame) -> tuple[str, float, dict, str, str]:
    row = df.iloc[-1]
    close = float(row["Close"])
    sma20 = float(row["sma20"])
    sma50 = float(row["sma50"])
    rsi = float(row["rsi"])
    volume = float(row["Volume"])
    vol_avg = float(row["vol_avg20"])
    high20 = float(row["high20"])
    low20 = float(row["low20"])
    atr = float(row["atr"])

    trend_up = sma20 > sma50
    breakout = close > high20
    breakdown = close < low20
    volume_ratio = volume / vol_avg if vol_avg > 0 else 0.0
    volume_spike = volume_ratio > 1.5

    if trend_up and breakout and volume_spike:
        signal = "BUY"
    elif (not trend_up) and breakdown:
        signal = "SELL"
    else:
        signal = "HOLD"

    score = calculate_score(signal, trend_up, breakout, breakdown, volume_spike, rsi)
    reason = build_reason(signal, trend_up, breakout, breakdown, volume_spike, rsi)
    explanation = build_short_explanation(
        signal=signal,
        close=close,
        sma20=sma20,
        sma50=sma50,
        high20=high20,
        low20=low20,
        volume_ratio=volume_ratio,
        rsi=rsi,
        atr=atr,
        trend_up=trend_up,
        breakout=breakout,
        breakdown=breakdown,
        volume_spike=volume_spike,
    )
    stats = {
        "close": round(close, 2),
        "rsi": round(rsi, 2),
        "volume_ratio": round(volume_ratio, 2),
        "atr": round(atr, 2),
        "atr_pct": round((atr / close * 100), 2) if close > 0 else 0.0,
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "high20": round(high20, 2),
        "low20": round(low20, 2),
    }
    return signal, score, stats, reason, explanation


def generate_mid_term_signal(df: pd.DataFrame) -> tuple[str, str]:
    row = df.iloc[-1]
    close = float(row["Close"])
    sma50 = float(row["sma50"])
    sma200_val = row.get("sma200", np.nan)
    if pd.isna(sma200_val):
        return "HOLD", "insufficient data for SMA200"
    sma200 = float(sma200_val)
    rsi = float(row["rsi"])

    above_sma200 = close > sma200
    sma50_above200 = sma50 > sma200
    if above_sma200 and sma50_above200 and rsi > 45:
        return "BUY", f"mid-term BUY rule: close {close:.2f} > SMA200 {sma200:.2f}, SMA50 {sma50:.2f} > SMA200, RSI {rsi:.1f} > 45"
    if (not above_sma200) and (not sma50_above200) and rsi < 55:
        return "SELL", f"mid-term SELL rule: close {close:.2f} < SMA200 {sma200:.2f}, SMA50 {sma50:.2f} < SMA200, RSI {rsi:.1f} < 55"
    return "HOLD", f"mid-term HOLD: mixed trend, close {close:.2f}, SMA50 {sma50:.2f}, SMA200 {sma200:.2f}, RSI {rsi:.1f}"


def generate_long_term_signal(df: pd.DataFrame) -> tuple[str, str]:
    row = df.iloc[-1]
    close = float(row["Close"])
    sma200_val = row.get("sma200", np.nan)
    if pd.isna(sma200_val):
        return "HOLD", "insufficient data for SMA200"
    sma200 = float(sma200_val)
    if len(df) >= 20 and not pd.isna(df["sma200"].iloc[-20]):
        sma200_20ago = float(df["sma200"].iloc[-20])
        sma200_rising = sma200_20ago < sma200
    else:
        sma200_20ago = None
        sma200_rising = False

    if close > sma200 and sma200_rising:
        ago_text = f", SMA200 20d ago {sma200_20ago:.2f}" if sma200_20ago is not None else ""
        return "BUY", f"long-term BUY rule: price {close:.2f} above rising SMA200 {sma200:.2f}{ago_text}"
    if close < sma200 and (not sma200_rising):
        ago_text = f", SMA200 20d ago {sma200_20ago:.2f}" if sma200_20ago is not None else ""
        return "SELL", f"long-term SELL rule: price {close:.2f} below non-rising SMA200 {sma200:.2f}{ago_text}"
    if close > sma200:
        return "HOLD", f"long-term HOLD: price {close:.2f} above SMA200 {sma200:.2f}, but slope not confirmed"
    return "HOLD", f"long-term HOLD: price {close:.2f} below SMA200 {sma200:.2f}, but slope not decisively down"


# ── Value analysis ─────────────────────────────────────────────────────────

def compute_value_analysis(info: dict) -> dict:
    """
    Reference rule copied from the main scanner.

    Checks:
      - PE <= MAX_PE
      - PB <= MAX_PB
      - marketCap >= MIN_MARKET_CAP

    Score:
      (1 / trailingPE) + (1 / priceToBook) + (log10(marketCap) / 10)
    Missing metrics are skipped rather than guessed.
    """
    pe = safe_float(info.get("trailingPE"))
    pb = safe_float(info.get("priceToBook"))
    mcap = safe_float(info.get("marketCap"))

    checks: list[str] = []
    failed: list[str] = []
    strengths: list[str] = []
    score = 0.0
    metric_count = 0

    if pe is not None:
        metric_count += 1
        checks.append(f"PE={pe:.2f}")
        if pe > 0:
            score += 1.0 / pe
        if pe <= 0:
            failed.append(f"PE {pe:.2f} invalid")
        elif pe > MAX_PE:
            failed.append(f"PE {pe:.2f} > max {MAX_PE:.2f}")
        elif pe <= 20:
            strengths.append("reasonable PE")
    else:
        checks.append("PE=n/a")

    if pb is not None:
        metric_count += 1
        checks.append(f"PB={pb:.2f}")
        if pb > 0:
            score += 1.0 / pb
        if pb <= 0:
            failed.append(f"PB {pb:.2f} invalid")
        elif pb > MAX_PB:
            failed.append(f"PB {pb:.2f} > max {MAX_PB:.2f}")
        elif pb <= 3:
            strengths.append("asset valuation not stretched")
    else:
        checks.append("PB=n/a")

    if mcap is not None:
        metric_count += 1
        checks.append(f"MCap={fmt_market_cap(mcap)}")
        if mcap > 0:
            score += math.log10(mcap) / 10.0
        if mcap < MIN_MARKET_CAP:
            failed.append(f"MCap {fmt_market_cap(mcap)} < min {fmt_market_cap(MIN_MARKET_CAP)}")
        elif mcap >= 10_000_000_000:
            strengths.append("large-cap stability")
    else:
        checks.append("MCap=n/a")

    score_out = round(score, 4) if metric_count > 0 else None

    if metric_count == 0:
        status = "NO_DATA"
        label = "NO DATA"
        reason = "No usable fundamentals from yfinance"
    elif failed:
        status = "FAIL"
        label = "WEAK VALUE"
        reason = "; ".join(failed)
    else:
        status = "PASS"
        if score >= 1.35:
            label = "STRONG VALUE"
        elif score >= 1.15:
            label = "FAIR VALUE"
        else:
            label = "PASS VALUE"
        reason = "; ".join(strengths) if strengths else "Passed reference value checks"

    summary = " | ".join(checks) + f" | Rule: PE<={MAX_PE:.0f}, PB<={MAX_PB:.0f}, MCap>={fmt_market_cap(MIN_MARKET_CAP)}"

    return {
        "pe": pe,
        "pb": pb,
        "market_cap": mcap,
        "value_score": score_out,
        "value_status": status,
        "value_label": label,
        "value_reason": reason,
        "value_summary": summary,
    }


# ── Potential analysis ─────────────────────────────────────────────────────

def compute_potential_analysis(
    short_signal: str,
    mid_signal: str,
    long_signal: str,
    stats: dict,
) -> dict:
    close = float(stats["close"])
    high20 = float(stats["high20"])
    rsi = float(stats["rsi"])
    atr_pct = float(stats["atr_pct"])

    score = 0
    reasons: list[str] = []
    risks: list[str] = []

    if long_signal == "BUY":
        score += 2
        reasons.append("strong long-term trend")
    elif long_signal == "SELL":
        score -= 2
        risks.append("long-term downtrend")

    if mid_signal == "BUY":
        score += 1
        reasons.append("mid-term trend supportive")
    elif mid_signal == "SELL":
        score -= 1
        risks.append("mid-term trend weak")

    if short_signal == "BUY":
        score += 2
        reasons.append("short-term trigger already active")
    elif short_signal == "SELL":
        score -= 2
        risks.append("short-term selling pressure")

    if 50 <= rsi <= 65:
        score += 1
        reasons.append(f"RSI {rsi:.1f} in healthy momentum zone")
    elif rsi > 75:
        score -= 1
        risks.append(f"RSI {rsi:.1f} overbought")
    elif rsi < 35:
        score -= 1
        risks.append(f"RSI {rsi:.1f} weak")

    distance_to_breakout_pct = None
    if close > 0 and high20 > 0:
        distance_to_breakout_pct = (high20 - close) / close * 100
        if 0 < distance_to_breakout_pct < 3:
            score += 1
            reasons.append(f"near breakout ({distance_to_breakout_pct:.1f}% below 20d high)")
        elif distance_to_breakout_pct < 0:
            reasons.append("already above prior 20d high")

    if atr_pct > 4:
        score -= 1
        risks.append(f"high volatility (ATR {atr_pct:.1f}%)")
    elif atr_pct < 2.5:
        reasons.append(f"volatility controlled (ATR {atr_pct:.1f}%)")

    if score >= 4:
        label = "HIGH POTENTIAL"
    elif score >= 2:
        label = "MEDIUM POTENTIAL"
    else:
        label = "LOW / RISKY"

    if reasons and risks:
        summary = f"Potential mixed: {', '.join(reasons[:2])}; watch {', '.join(risks[:2])}."
    elif reasons:
        summary = f"Potential positive: {', '.join(reasons[:3])}."
    elif risks:
        summary = f"Potential limited: {', '.join(risks[:3])}."
    else:
        summary = "Potential neutral: no strong edge found."

    return {
        "potential_score": score,
        "potential_label": label,
        "potential_reason": "; ".join(reasons) if reasons else "no strong upside catalyst",
        "potential_risk": "; ".join(risks) if risks else "no major risk flags from current rules",
        "potential_summary": summary,
        "distance_to_breakout_pct": round(distance_to_breakout_pct, 2) if distance_to_breakout_pct is not None else None,
    }


# ── Summary builder ────────────────────────────────────────────────────────

def build_stock_summary(row: dict) -> str:
    if row["short_term_signal"] == "BUY" and row["value_status"] == "PASS" and row["potential_score"] >= 2:
        return "Triggered short-term entry with acceptable value support and positive upside profile."
    if row["long_term_signal"] == "BUY" and row["short_term_signal"] == "HOLD":
        return "Long-term trend is constructive, but short-term entry trigger has not fired yet."
    if row["short_term_signal"] == "SELL" and row["long_term_signal"] == "SELL":
        return "Weak across both short and long horizons; capital preservation matters more than entry timing."
    if row["value_status"] == "PASS" and row["potential_score"] < 2:
        return "Valuation is acceptable, but momentum and setup quality are not strong enough yet."
    if row["value_status"] == "FAIL" and row["long_term_signal"] == "BUY":
        return "Trend is still positive, but valuation looks stretched under the reference rules."
    return "Mixed profile: use the combined signal, value, and potential sections before making a decision."


# ── Scan ───────────────────────────────────────────────────────────────────

def scan_wishlist(symbols: list[str]) -> tuple[list[dict], list[str]]:
    results: list[dict] = []
    skipped: list[str] = []

    print("\n=== SCAN ===")
    for symbol in symbols:
        print(f"Scanning {symbol}...")
        df = load_data(symbol)
        if df is None:
            skipped.append(symbol)
            print("  - skipped: no data / insufficient history")
            continue

        df = calculate_indicators(df)
        required_cols = ["sma20", "sma50", "sma200", "rsi", "atr", "vol_avg20", "high20", "low20"]
        if df.iloc[-1][required_cols].isnull().any():
            skipped.append(symbol)
            print("  - skipped: NaN in indicators")
            continue

        short_signal, score, stats, reason, short_explanation = generate_short_term_signal(df)
        mid_signal, mid_reason = generate_mid_term_signal(df)
        long_signal, long_reason = generate_long_term_signal(df)
        fundamentals = load_fundamentals(symbol)
        value = compute_value_analysis(fundamentals)
        potential = compute_potential_analysis(short_signal, mid_signal, long_signal, stats)

        result = {
            "symbol": symbol,
            "market": get_market(symbol),
            "signal": short_signal,
            "short_term_signal": short_signal,
            "mid_term_signal": mid_signal,
            "long_term_signal": long_signal,
            "score": round(score, 2),
            "close": stats["close"],
            "rsi": stats["rsi"],
            "volume_ratio": stats["volume_ratio"],
            "atr": stats["atr"],
            "atr_pct": stats["atr_pct"],
            "high20": stats["high20"],
            "low20": stats["low20"],
            "reason": reason,
            "short_term_reason": reason,
            "short_term_explanation": short_explanation,
            "mid_term_reason": mid_reason,
            "long_term_reason": long_reason,
            "value_status": value["value_status"],
            "value_label": value["value_label"],
            "value_score": value["value_score"],
            "value_reason": value["value_reason"],
            "value_summary": value["value_summary"],
            "pe": value["pe"],
            "pb": value["pb"],
            "market_cap": value["market_cap"],
            "potential_score": potential["potential_score"],
            "potential_label": potential["potential_label"],
            "potential_reason": potential["potential_reason"],
            "potential_risk": potential["potential_risk"],
            "potential_summary": potential["potential_summary"],
            "distance_to_breakout_pct": potential["distance_to_breakout_pct"],
            "as_of": datetime.today().strftime("%Y-%m-%d"),
        }
        result["combined_summary"] = build_stock_summary(result)

        print(
            f"  - short={short_signal} mid={mid_signal} long={long_signal} "
            f"value={value['value_label']} potential={potential['potential_label']} "
            f"score={score:.2f} value_score={value['value_score']} potential_score={potential['potential_score']}"
        )

        results.append(result)

    results.sort(
        key=lambda x: (
            x["potential_score"] * -1,
            -999 if x["value_score"] is None else -x["value_score"],
            0 if x["signal"] != "HOLD" else 1,
            x["symbol"],
        )
    )
    return results, skipped


# ── Output ─────────────────────────────────────────────────────────────────

def _signal_badge(label: str, signal: str) -> str:
    colors = {
        "BUY": ("#166534", "#dcfce7", "#bbf7d0"),
        "SELL": ("#991b1b", "#fee2e2", "#fecaca"),
        "HOLD": ("#475569", "#f1f5f9", "#cbd5e1"),
    }
    fg, bg, bd = colors.get(signal, colors["HOLD"])
    return (
        f'<span style="color:{fg};background:{bg};border:1px solid {bd};'
        f'padding:5px 10px;border-radius:999px;font-size:11px;font-weight:800;'
        f'display:inline-block;">{label}: {signal}</span>'
    )


def _value_badge(status: str, label: str) -> str:
    colors = {
        "PASS": ("#166534", "#dcfce7", "#bbf7d0"),
        "FAIL": ("#991b1b", "#fee2e2", "#fecaca"),
        "NO_DATA": ("#475569", "#f1f5f9", "#cbd5e1"),
    }
    fg, bg, bd = colors.get(status, colors["NO_DATA"])
    return (
        f'<span style="color:{fg};background:{bg};border:1px solid {bd};'
        f'padding:5px 10px;border-radius:999px;font-size:11px;font-weight:800;'
        f'display:inline-block;">Value: {label}</span>'
    )


def _potential_badge(label: str) -> str:
    color_map = {
        "HIGH POTENTIAL": ("#7c2d12", "#ffedd5", "#fdba74"),
        "MEDIUM POTENTIAL": ("#854d0e", "#fef9c3", "#fde68a"),
        "LOW / RISKY": ("#475569", "#f1f5f9", "#cbd5e1"),
    }
    fg, bg, bd = color_map.get(label, color_map["LOW / RISKY"])
    return (
        f'<span style="color:{fg};background:{bg};border:1px solid {bd};'
        f'padding:5px 10px;border-radius:999px;font-size:11px;font-weight:800;'
        f'display:inline-block;">Potential: {label}</span>'
    )


def _metric_chip(label: str, value: str) -> str:
    return (
        '<div style="display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:12px;padding:10px 12px;margin:0 8px 8px 0;min-width:112px;vertical-align:top;">'
        f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;font-weight:800;">{label}</div>'
        f'<div style="font-size:15px;color:#0f172a;font-weight:800;margin-top:4px;line-height:1.3;">{value}</div>'
        '</div>'
    )


def _section_block(title: str, body: str, tone: str = "default") -> str:
    tone_map = {
        "default": ("#e2e8f0", "#0f172a", "#ffffff"),
        "muted": ("#e2e8f0", "#334155", "#f8fafc"),
        "value": ("#bbf7d0", "#166534", "#f0fdf4"),
        "potential": ("#fde68a", "#92400e", "#fffbeb"),
        "risk": ("#fecaca", "#991b1b", "#fef2f2"),
    }
    border, title_color, bg = tone_map.get(tone, tone_map["default"])
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:14px;'
        f'padding:14px 16px;margin-top:12px;">'
        f'<div style="font-size:11px;font-weight:900;color:{title_color};'
        f'text-transform:uppercase;letter-spacing:.09em;margin-bottom:8px;">{title}</div>'
        f'<div style="font-size:14px;color:#334155;line-height:1.7;">{body}</div>'
        '</div>'
    )


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def format_email_html(results: list[dict], skipped: list[str], invalid_th: list[str]) -> str:
    cards: list[str] = []

    for r in results:
        headline = _one_line(r["combined_summary"])
        short_reason = _one_line(r["short_term_reason"])
        mid_reason = _one_line(r["mid_term_reason"])
        long_reason = _one_line(r["long_term_reason"])
        value_reason = _one_line(r["value_reason"])
        potential_summary = _one_line(r["potential_summary"])
        potential_risk = _one_line(r["potential_risk"])

        header = (
            '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">'
            '<div>'
            f'<div style="font-size:24px;font-weight:900;color:#0f172a;letter-spacing:-.02em;">{r["symbol"]}</div>'
            f'<div style="font-size:12px;color:#64748b;margin-top:4px;">{r["market"]} wishlist • close {r["close"]:.2f} • as of {r["as_of"]}</div>'
            '</div>'
            '<div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">'
            f'{_signal_badge("Short", r["short_term_signal"])}'
            f'{_signal_badge("Mid", r["mid_term_signal"])}'
            f'{_signal_badge("Long", r["long_term_signal"])}'
            f'{_value_badge(r["value_status"], r["value_label"])}'
            f'{_potential_badge(r["potential_label"])}'
            '</div>'
            '</div>'
        )

        summary_box = (
            '<div style="margin-top:14px;background:#eff6ff;border:1px solid #bfdbfe;'
            'border-radius:14px;padding:14px 16px;">'
            '<div style="font-size:11px;font-weight:900;color:#1d4ed8;text-transform:uppercase;letter-spacing:.08em;">Quick Read</div>'
            f'<div style="font-size:16px;line-height:1.6;font-weight:800;color:#0f172a;margin-top:6px;">{headline}</div>'
            '</div>'
        )

        metrics = (
            _metric_chip("Price", f'{r["close"]:.2f}') +
            _metric_chip("RSI", f'{r["rsi"]:.1f}') +
            _metric_chip("Value Score", '-' if r["value_score"] is None else str(r["value_score"])) +
            _metric_chip("Potential", f'{r["potential_score"]}')
        )

        signal_body = (
            f'<div>• <strong>Short:</strong> {r["short_term_signal"]} — {short_reason}</div>'
            f'<div style="margin-top:7px;">• <strong>Mid:</strong> {r["mid_term_signal"]} — {mid_reason}</div>'
            f'<div style="margin-top:7px;">• <strong>Long:</strong> {r["long_term_signal"]} — {long_reason}</div>'
            f'<div style="margin-top:10px;color:#475569;"><strong>Why this matters:</strong> {_one_line(r["short_term_explanation"])}</div>'
        )

        value_body = (
            f'<div>• <strong>{r["value_label"]}</strong></div>'
            f'<div style="margin-top:7px;">• {value_reason}</div>'
            f'<div style="margin-top:7px;color:#64748b;">PE {fmt_num(r["pe"])} • PB {fmt_num(r["pb"])} • MCap {fmt_market_cap(r["market_cap"])}</div>'
        )

        potential_body = (
            f'<div>• <strong>{r["potential_label"]}</strong> (score {r["potential_score"]})</div>'
            f'<div style="margin-top:7px;">• {potential_summary}</div>'
        )

        risk_body = (
            f'<div>• {potential_risk}</div>'
            f'<div style="margin-top:7px;">• Breakout distance: '
            f'{"-" if r["distance_to_breakout_pct"] is None else str(r["distance_to_breakout_pct"]) + "%"}</div>'
            f'<div style="margin-top:7px;">• ATR: {r["atr"]:.2f} ({r["atr_pct"]:.2f}% of price)</div>'
        )

        card = (
            '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;'
            'padding:20px 20px 18px;margin-bottom:18px;box-shadow:0 4px 14px rgba(15,23,42,.05);">'
            + header +
            summary_box +
            f'<div style="margin-top:14px;">{metrics}</div>' +
            _section_block("Signals", signal_body, "muted") +
            _section_block("Value", value_body, "value") +
            _section_block("Potential", potential_body, "potential") +
            _section_block("Risk / Watchouts", risk_body, "risk") +
            '</div>'
        )
        cards.append(card)

    skipped_html = ", ".join(skipped) if skipped else "-"
    invalid_html = ", ".join(invalid_th) if invalid_th else "-"

    summary_cards = (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">'
        f'<div style="background:#f8fafc;padding:12px 16px;border-radius:12px;border:1px solid #e2e8f0;"><div style="font-size:12px;color:#64748b;">Scanned</div><div style="font-size:24px;font-weight:900;color:#0f172a;">{len(results)}</div></div>'
        f'<div style="background:#dcfce7;padding:12px 16px;border-radius:12px;border:1px solid #bbf7d0;"><div style="font-size:12px;color:#166534;">Short BUY</div><div style="font-size:24px;font-weight:900;color:#166534;">{sum(1 for r in results if r["signal"]=="BUY")}</div></div>'
        f'<div style="background:#f1f5f9;padding:12px 16px;border-radius:12px;border:1px solid #cbd5e1;"><div style="font-size:12px;color:#475569;">Short HOLD</div><div style="font-size:24px;font-weight:900;color:#475569;">{sum(1 for r in results if r["signal"]=="HOLD")}</div></div>'
        f'<div style="background:#ecfccb;padding:12px 16px;border-radius:12px;border:1px solid #bef264;"><div style="font-size:12px;color:#3f6212;">Value PASS</div><div style="font-size:24px;font-weight:900;color:#3f6212;">{sum(1 for r in results if r["value_status"]=="PASS")}</div></div>'
        f'<div style="background:#ffedd5;padding:12px 16px;border-radius:12px;border:1px solid #fdba74;"><div style="font-size:12px;color:#9a3412;">High Potential</div><div style="font-size:24px;font-weight:900;color:#9a3412;">{sum(1 for r in results if r["potential_label"]=="HIGH POTENTIAL")}</div></div>'
        '</div>'
    )

    return f"""
    <div style="font-family:Arial,Helvetica,sans-serif;background:#f8fafc;padding:20px;">
      <div style="max-width:880px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:20px;padding:24px;">
        <h1 style="margin:0 0 6px;font-size:28px;color:#0f172a;letter-spacing:-.02em;">USA / Thai Wishlist Scanner</h1>
        <div style="color:#64748b;margin-bottom:18px;font-size:14px;">Cleaner email version • easier to scan on desktop and mobile • includes HOLD names too</div>

        {summary_cards}

        <div style="margin-bottom:18px;color:#334155;line-height:1.7;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;">
          <div><strong>Skipped (no data / insufficient history):</strong> {skipped_html}</div>
          <div><strong>Invalid TH symbols:</strong> {invalid_html}</div>
          <div style="margin-top:8px;"><strong>Reference value rule:</strong> PE ≤ {MAX_PE:.0f}, PB ≤ {MAX_PB:.0f}, Market Cap ≥ {fmt_market_cap(MIN_MARKET_CAP)}.</div>
          <div><strong>Potential rule:</strong> long-term BUY +2, mid-term BUY +1, short-term BUY +2, short-term SELL -2, RSI 50-65 +1, RSI &gt; 75 -1, near breakout (&lt;3%) +1, ATR% &gt; 4 -1.</div>
        </div>

        <h2 style="font-size:18px;color:#0f172a;margin:0 0 10px;">All Wishlist Results</h2>
        <div style="color:#64748b;font-size:13px;margin-bottom:14px;">Each card now starts with a one-line takeaway, then shows only the most important numbers first.</div>
        {''.join(cards) if cards else '<div style="padding:18px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;color:#64748b;">No scannable results.</div>'}
      </div>
    </div>
    """


def send_email(subject: str, html_body: str) -> None:
    api_key = os.getenv("RESEND_API_KEY", "")
    email_from = os.getenv("EMAIL_FROM", "")
    to_raw = os.getenv("EMAIL_TO", "")
    to_list = [x.strip() for x in to_raw.split(",") if x.strip()]

    print("\n=== EMAIL CONFIG DEBUG ===")
    print("API KEY:", "OK" if api_key else "MISSING")
    print("FROM:", email_from)
    print("TO:", to_list)

    if not EMAIL_ENABLED:
        print("Email disabled by EMAIL_ENABLED=false")
        return

    if not api_key or not email_from or not to_list:
        print("Email skipped: RESEND_API_KEY, EMAIL_FROM, or EMAIL_TO missing")
        return

    payload = {
        "from": email_from,
        "to": to_list,
        "subject": subject,
        "html": html_body,
    }

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        if resp.status_code in (200, 201):
            print("Email sent successfully")
        else:
            print(f"Email failed: HTTP {resp.status_code} - {resp.text}")
    except requests.RequestException as exc:
        print(f"Email error: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    wishlist, _us, invalid_th = build_wishlist()
    if not wishlist:
        print("No symbols found in US_WISHLIST / TH_WISHLIST")
        return

    results, skipped = scan_wishlist(wishlist)

    print("\n=== SUMMARY ===")
    print(f"Scanned results: {len(results)}")
    print(f"BUY:  {sum(1 for r in results if r['signal'] == 'BUY')}")
    print(f"SELL: {sum(1 for r in results if r['signal'] == 'SELL')}")
    print(f"HOLD: {sum(1 for r in results if r['signal'] == 'HOLD')}")
    print(f"Value PASS: {sum(1 for r in results if r['value_status'] == 'PASS')}")
    print(f"Value FAIL: {sum(1 for r in results if r['value_status'] == 'FAIL')}")
    print(f"Value NO_DATA: {sum(1 for r in results if r['value_status'] == 'NO_DATA')}")
    print(f"High Potential: {sum(1 for r in results if r['potential_label'] == 'HIGH POTENTIAL')}")
    print(f"Skipped: {len(skipped)}")

    print("\n=== RESULTS JSON ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    html = format_email_html(results, skipped, invalid_th)
    send_email(
        subject=f"USA / Thai Wishlist Scanner - {datetime.today().strftime('%Y-%m-%d')}",
        html_body=html,
    )


if __name__ == "__main__":
    main()
