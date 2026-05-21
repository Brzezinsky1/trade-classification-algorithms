"""
Public API — the single function the professor imports.

    from src.classify_side import classify_side
    sides = classify_side(trades)

Contract
--------
Input  : a trades DataFrame (DatetimeIndex, UTC) with at least 'price' and 'amount'.
         An extra 'side' column (ground truth) is ignored if present.
Output : a boolean pd.Series aligned 1-to-1 with the input index, where
            True  = sell aggressor
            False = buy aggressor

The function is defensive by design: it works on a single symbol even though the
model was trained on two, handles the first trades of the day (windows are
left-padded), tolerates NaNs, and falls back to the trades-only tick rule if the
trained artifacts are missing or fail to load — so it never raises on valid input.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .baselines import tick_rule
from .models import sequence as seq

# module-level cache so repeated calls don't reload the model from disk
_BUNDLE = None
_LOAD_FAILED = False

THRESHOLD = 0.5  # sell-aggressor probability cutoff


def _get_bundle():
    """Load (and cache) the trained sequence bundle, or return None if unavailable."""
    global _BUNDLE, _LOAD_FAILED
    if _BUNDLE is not None:
        return _BUNDLE
    if _LOAD_FAILED:
        return None
    try:
        if not seq.artifacts_exist():
            _LOAD_FAILED = True
            return None
        _BUNDLE = seq.load_artifacts()
        return _BUNDLE
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"classify_side: could not load model artifacts ({exc}); "
                      "falling back to tick rule.")
        _LOAD_FAILED = True
        return None


def _empty_bool_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.empty(0, dtype=bool), index=index, name="predicted_side")


def classify_side(trades: pd.DataFrame) -> pd.Series:
    """
    Classify the aggressor side of each trade.

    Parameters
    ----------
    trades : pd.DataFrame
        Time-indexed (DatetimeIndex, UTC). Must contain 'price' and 'amount'.

    Returns
    -------
    pd.Series
        Boolean, aligned to ``trades.index``. True = sell aggressor, False = buy.
    """
    if not isinstance(trades, pd.DataFrame):
        raise TypeError(f"classify_side expects a DataFrame, got {type(trades).__name__}")

    for col in ("price", "amount"):
        if col not in trades.columns:
            raise KeyError(f"classify_side requires a '{col}' column")

    # empty input → empty bool series, index preserved
    if len(trades) == 0:
        return _empty_bool_series(trades.index)

    bundle = _get_bundle()

    # fallback: trades-only tick rule (always valid, never raises)
    if bundle is None:
        return tick_rule(trades).astype(bool).rename("predicted_side")

    try:
        proba = seq.predict_proba(bundle, trades)
        pred = (proba >= THRESHOLD)
        # guard against any NaN probabilities (shouldn't happen) → tick fallback
        if pred.isna().any():
            tick = tick_rule(trades)
            pred = pred.fillna(tick)
        return pred.astype(bool).rename("predicted_side")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"classify_side: model inference failed ({exc}); using tick rule.")
        return tick_rule(trades).astype(bool).rename("predicted_side")
