"""
optrl.data - bring your own data (Part 9).

Two inputs are all the selector needs:

1. MARKET DATA, daily, one row per trading day, columns:
       date, close, vix, vix3m                (required)
   plus optionally an earnings calendar (list of dates).
   -> build_features() turns it into the model features (all backward-looking).

2. STRATEGY P&L TABLE, one row per decision date, one column per strategy:
       date, strategy_1, strategy_2, ..., strategy_7
   Each cell = $ P&L of that strategy if opened at that date's close and managed
   by YOUR rules until exit, net of costs, for a FIXED risk unit (e.g. $1,000 max loss).
   -> this is what your existing backtester already produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["vix", "iv_rank", "term_structure", "ema_trend", "rv_iv_spread", "days_to_earnings", "ret_5d"]


def build_features(close: pd.Series, vix: pd.Series, vix3m: pd.Series,
                   earnings_dates=None, no_earnings_value: int = 63) -> pd.DataFrame:
    """Point-in-time features from raw daily data. Every column uses only past data."""
    f = pd.DataFrame(index=close.index)
    f["vix"] = vix
    lo = vix.rolling(252, min_periods=60).min()
    hi = vix.rolling(252, min_periods=60).max()
    f["iv_rank"] = ((vix - lo) / (hi - lo) * 100).clip(0, 100)
    f["term_structure"] = vix / vix3m
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    f["ema_trend"] = (ema9 - ema20) / close * 100
    logret = np.log(close).diff()
    f["rv_iv_spread"] = logret.rolling(20).std() * np.sqrt(252) * 100 - vix
    f["ret_5d"] = (close / close.shift(5) - 1) * 100
    if earnings_dates is None or len(earnings_dates) == 0:
        f["days_to_earnings"] = no_earnings_value
    else:
        pos = np.sort(np.searchsorted(close.index.values, pd.to_datetime(earnings_dates).values))
        nxt_idx = np.searchsorted(pos, np.arange(len(close)))           # first event at/after each day
        nxt = np.where(nxt_idx < len(pos), pos[np.minimum(nxt_idx, len(pos) - 1)] - np.arange(len(close)),
                       no_earnings_value)
        f["days_to_earnings"] = np.minimum(nxt, no_earnings_value)
    return f[FEATURES]


def load_market_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    missing = {"close", "vix", "vix3m"} - set(df.columns)
    if missing:
        raise ValueError(f"market CSV is missing columns: {missing}")
    return df


def load_pnl_csv(path: str, date_col: str = "date", add_no_trade: bool = True) -> pd.DataFrame:
    t = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    if add_no_trade and "no_trade" not in t.columns:
        t.insert(0, "no_trade", 0.0)
    return t


def validate_inputs(features: pd.DataFrame, pnl: pd.DataFrame, risk_unit: float = 1000.0) -> list[str]:
    """Cheap checks that catch the most common data problems. Returns a list of warnings."""
    w = []
    common = features.index.intersection(pnl.index)
    if len(common) < len(pnl):
        w.append(f"{len(pnl) - len(common)} P&L dates have no feature row (holidays? timezone?)")
    if not pnl.index.is_monotonic_increasing:
        w.append("P&L dates are not sorted")
    if pnl.index.duplicated().any():
        w.append("duplicate P&L dates")
    na = features.loc[common].isna().mean()
    if (na > 0).any():
        w.append(f"features have NaNs (warm-up?): {na[na > 0].round(3).to_dict()}")
    if pnl.isna().any().any():
        w.append("P&L table has NaNs: decide explicitly (0 = not tradable?) rather than silently dropping")
    worst = pnl.min()
    too_big = worst[worst < -3 * risk_unit]
    if len(too_big):
        w.append(f"losses > 3x risk unit (sizing? undefined risk?): {too_big.round(0).to_dict()}")
    gaps = pnl.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > 14:
        w.append(f"gap of {int(gaps.max())} days between decision dates")
    return w


def align(features: pd.DataFrame, pnl: pd.DataFrame, dropna: bool = True):
    """Keep dates present in both; drop feature warm-up rows."""
    idx = features.index.intersection(pnl.index)
    X, R = features.loc[idx], pnl.loc[idx]
    if dropna:
        ok = X.notna().all(axis=1) & R.notna().all(axis=1)
        X, R = X[ok], R[ok]
    return X, R
