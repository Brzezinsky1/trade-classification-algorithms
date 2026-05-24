"""
Feature engineering from (price, amount, time) only — no order book.

All features are computed with strict no-look-ahead: at row t, only
information from rows 0..t is used.  NaNs appear at window boundaries
and should be handled (imputed or masked) by downstream models.

Entry point: build_features(trades) -> pd.DataFrame
"""

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _filled_tick(price: pd.Series) -> pd.Series:
    """Forward-fill non-zero tick direction (+1 / -1), then backward-fill."""
    return np.sign(price.diff()).replace(0, np.nan).ffill().bfill().fillna(0)


def _run_length(price: pd.Series) -> pd.Series:
    """
    How many consecutive ticks have been in the same direction (including
    the current trade)?  Resets on every direction change.
    """
    tick = _filled_tick(price)
    group_id = (tick != tick.shift()).cumsum()
    return group_id.groupby(group_id).cumcount() + 1


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all trade-based features from a trades DataFrame.

    Parameters
    ----------
    trades : pd.DataFrame
        DatetimeIndex (UTC).  Must contain at least 'price' and 'amount'.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned 1-to-1 with trades.index.
        Column names are listed in FEATURE_NAMES.
    """
    price  = trades["price"]
    amount = trades["amount"]
    feats  = pd.DataFrame(index=trades.index)

    # ── tick direction (lags 1-5) ─────────────────────────────────────────────
    # raw sign of price change: +1 uptick, -1 downtick, 0 zero-tick
    for lag in [1, 2, 3, 4, 5]:
        feats[f"tick_{lag}"] = np.sign(price.diff(lag))

    # forward-filled tick: no zeros, useful as a standalone feature
    tick_ff = _filled_tick(price)
    feats["tick_ff"] = tick_ff
    feats["tick_rule_side"] = (tick_ff < 0).astype(int)

    # ── log-returns at multiple horizons ──────────────────────────────────────
    for n in [1, 3, 5, 10]:
        feats[f"ret_{n}"] = np.log(price / price.shift(n))

    # ── inter-trade time deltas ───────────────────────────────────────────────
    dt_s = trades.index.to_series().diff().dt.total_seconds()
    feats["dt_s"]     = dt_s.values
    feats["log_dt_s"] = np.log1p(np.maximum(feats["dt_s"].fillna(0), 0))

    # ── rolling volume z-score ────────────────────────────────────────────────
    for w in [10, 20, 50]:
        roll = amount.rolling(w, min_periods=2)
        feats[f"vol_z_{w}"] = (amount - roll.mean()) / (roll.std() + 1e-10)

    feats["log_amount"] = np.log1p(amount)

    # ── run-length / streak ───────────────────────────────────────────────────
    feats["run_length"] = _run_length(price)
    feats["run_dir"]    = tick_ff  # direction of the current run (+1 / -1)

    # ── price proximity to recent local high/low ──────────────────────────────
    for w in [20, 50]:
        lo   = price.rolling(w, min_periods=2).min()
        hi   = price.rolling(w, min_periods=2).max()
        span = hi - lo
        feats[f"pct_range_{w}"] = np.where(span > 0, (price - lo) / span, 0.5)

    # ── round-number / tick-size features ─────────────────────────────────────
    # fractional part of price (0 = round integer, ~1 = just below next integer)
    feats["frac_price"] = price - np.floor(price)

    # how far price sits from the nearest power-of-10 "big round" level
    # e.g. price=1.234 → magnitude=1 → round_prox=0.234
    #      price=123.4 → magnitude=100 → round_prox=0.234
    magnitude = 10.0 ** np.floor(np.log10(price.abs().clip(lower=1e-10)))
    feats["round_prox"] = (price % magnitude) / magnitude

    feats["log_price"] = np.log(price.clip(lower=1e-10))

    return feats


# ── Feature registry ──────────────────────────────────────────────────────────

FEATURE_NAMES: list = [
    # tick direction
    "tick_1", "tick_2", "tick_3", "tick_4", "tick_5", "tick_ff",
    # log-returns
    "ret_1", "ret_3", "ret_5", "ret_10",
    # time gaps
    "dt_s", "log_dt_s",
    # volume
    "vol_z_10", "vol_z_20", "vol_z_50", "log_amount",
    # streaks
    "run_length", "run_dir",
    # price position in recent range
    "pct_range_20", "pct_range_50",
    # round-number proximity
    "frac_price", "round_prox", "log_price",

    "tick_rule_side",
]
