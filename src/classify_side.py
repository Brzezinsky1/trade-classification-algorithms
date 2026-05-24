"""
Public API — the single function the professor imports.

    from src.classify_side import classify_side
    sides = classify_side(trades)

Input  : a trades DataFrame (DatetimeIndex, UTC) with at least 'price' and 'amount'.
         An extra 'side' column (ground truth) is ignored if present.
Output : a boolean pd.Series aligned 1-to-1 with the input index, where
            True  = sell aggressor
            False = buy aggressor

"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .baselines import tick_rule
from .models import ensemble, gbm

# module-level cache so repeated calls don't reload models from disk
_BUNDLE = None
_BACKEND = None          # "ensemble" | "gbm" | "tick"
_RESOLVED = False

THRESHOLD = 0.5          # used only by the gbm fallback path


def _resolve_backend():
    """
    Pick the best available backend once and cache it.

    Returns (backend_name, bundle_or_None).
    Order of preference: ensemble → gbm → tick rule.
    """
    global _BUNDLE, _BACKEND, _RESOLVED
    if _RESOLVED:
        return _BACKEND, _BUNDLE

    for name, module in (("ensemble", ensemble), ("gbm", gbm)):
        try:
            if module.artifacts_exist():
                _BUNDLE = module.load_artifacts()
                _BACKEND = name
                _RESOLVED = True
                return _BACKEND, _BUNDLE
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"classify_side: failed to load {name} artifacts ({exc}).")

    _BACKEND, _BUNDLE, _RESOLVED = "tick", None, True
    return _BACKEND, _BUNDLE


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

    if len(trades) == 0:
        return _empty_bool_series(trades.index)

    backend, bundle = _resolve_backend()

    if backend == "tick" or bundle is None:
        return tick_rule(trades).astype(bool).rename("predicted_side")

    try:
        if backend == "ensemble":
            pred = ensemble.predict(bundle, trades)
        else:  # gbm
            proba = gbm.predict_proba(bundle, trades)
            pred = (proba >= THRESHOLD)

        # guard against any NaN probabilities → tick fallback for those rows
        if pred.isna().any():
            tick = tick_rule(trades)
            pred = pred.fillna(tick)
        return pred.astype(bool).rename("predicted_side")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"classify_side: {backend} inference failed ({exc}); using tick rule.")
        return tick_rule(trades).astype(bool).rename("predicted_side")
