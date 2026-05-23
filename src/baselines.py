"""
Baseline classifiers: Tick Rule, Quote Rule, Lee-Ready.

Convention (matches the project-wide contract):
  True  = sell aggressor
  False = buy aggressor

Tick rule is trades-only and is a *fair* baseline under the professor's
evaluation constraint.  Quote rule and Lee-Ready require the order book
and serve as oracle upper-bound references during development.

All functions return a fully boolean pd.Series with no NaN — any
indeterminate cases are resolved by a tick-rule fallback so the output is
always aligned 1-to-1 with the input trades.
"""

import numpy as np
import pandas as pd

from .data import attach_midpoint


# ── Internal helper ───────────────────────────────────────────────────────────

def _filled_tick(price: pd.Series) -> pd.Series:
    """
    Forward-fill (then backward-fill) non-zero tick direction.
    Returns +1 / -1 Series with the same index.
    """
    tick = np.sign(price.diff()).replace(0, np.nan)
    return tick.ffill().bfill().fillna(0)


# ── Public baselines ──────────────────────────────────────────────────────────

def tick_rule(trades: pd.DataFrame) -> pd.Series:
    """
    Tick rule — trades only (fair baseline).

    Uptick   → False (buy aggressor)
    Downtick → True  (sell aggressor)
    Zero-tick → carry forward last non-zero tick
    No prior tick → backward-fill, then default False if all prices equal.

    Returns
    -------
    pd.Series[bool]  aligned to trades.index
    """
    tick = _filled_tick(trades["price"])
    result = (tick < 0).astype(bool)
    return result.rename("predicted_side")


def quote_rule(trades: pd.DataFrame, orderbook: pd.DataFrame) -> pd.Series:
    """
    Quote rule — requires order book (oracle baseline).

    price < midpoint → True  (sell aggressor)
    price > midpoint → False (buy aggressor)
    price = midpoint → tick rule fallback

    Returns
    -------
    pd.Series[bool]  aligned to trades.index
    """
    merged = attach_midpoint(trades, orderbook)
    mid = merged["midpoint"]
    price = trades["price"]

    tick_fb = tick_rule(trades)

    pred = pd.Series(np.nan, index=trades.index)
    pred[price < mid] = True
    pred[price > mid] = False
    at_mid = pred.isna()
    pred[at_mid] = tick_fb[at_mid]

    return pred.astype(bool).rename("predicted_side")


def lee_ready(trades: pd.DataFrame, orderbook: pd.DataFrame) -> pd.Series:
    """
    Lee-Ready — requires order book (oracle baseline).

    Quote rule primary; tick rule fallback at the midpoint.
    Remaining indeterminates (first trade ever at midpoint, no prior tick)
    are filled with the tick-rule fallback.

    Returns
    -------
    pd.Series[bool]  aligned to trades.index
    """
    merged = attach_midpoint(trades, orderbook)
    mid = merged["midpoint"]
    price = trades["price"]

    tick_raw = np.sign(price.diff()).replace(0, np.nan).ffill()
    at_mid = price == mid

    labels = pd.Series(
        np.select(
            [price < mid,
             price > mid,
             at_mid & (tick_raw < 0),
             at_mid & (tick_raw > 0)],
            [True, False, True, False],
            default=np.nan,
        ),
        index=trades.index,
    )

    # resolve any remaining NaN with the tick fallback
    still_nan = labels.isna()
    if still_nan.any():
        labels[still_nan] = tick_rule(trades)[still_nan].astype(float)

    return labels.astype(bool).rename("predicted_side")
