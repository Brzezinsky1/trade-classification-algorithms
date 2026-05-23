"""
Contract / sanity tests for the public ``classify_side`` API.

These verify the *interface* the professor relies on — type, dtype, length, index
alignment, and graceful behaviour on edge cases — not predictive accuracy.

Run with:  pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# make the repo root importable when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify_side import classify_side  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_trades(n: int = 200, seed: int = 0, with_side: bool = False) -> pd.DataFrame:
    """Synthetic trades with a UTC DatetimeIndex, like the real data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-12", periods=n, freq="s", tz="UTC")
    price = 0.025 + np.cumsum(rng.normal(0, 1e-4, n))
    amount = rng.lognormal(7, 1.5, n)
    df = pd.DataFrame({"price": price, "amount": amount}, index=idx)
    if with_side:
        df["side"] = rng.random(n) > 0.5
    return df


@pytest.fixture
def trades():
    return _make_trades(200)


# ── Core contract ─────────────────────────────────────────────────────────────

def test_returns_series(trades):
    out = classify_side(trades)
    assert isinstance(out, pd.Series)


def test_dtype_is_bool(trades):
    out = classify_side(trades)
    assert out.dtype == bool


def test_length_matches_input(trades):
    out = classify_side(trades)
    assert len(out) == len(trades)


def test_index_matches_input(trades):
    out = classify_side(trades)
    pd.testing.assert_index_equal(out.index, trades.index)


def test_no_nans(trades):
    out = classify_side(trades)
    assert not out.isna().any()


# ── Robustness to the 'side' column ───────────────────────────────────────────

def test_ignores_extra_side_column():
    with_side = _make_trades(150, with_side=True)
    without = with_side.drop(columns="side")
    out_with = classify_side(with_side)
    out_without = classify_side(without)
    # presence of the ground-truth column must not change predictions
    pd.testing.assert_series_equal(out_with, out_without)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_single_trade():
    df = _make_trades(1)
    out = classify_side(df)
    assert isinstance(out, pd.Series)
    assert len(out) == 1
    assert out.dtype == bool


def test_two_trades():
    df = _make_trades(2)
    out = classify_side(df)
    assert len(out) == 2
    assert out.dtype == bool


def test_empty_input():
    df = _make_trades(5).iloc[:0]
    out = classify_side(df)
    assert isinstance(out, pd.Series)
    assert len(out) == 0
    assert out.dtype == bool


def test_constant_price():
    df = _make_trades(50)
    df["price"] = 0.025  # all identical → ticks are all zero
    out = classify_side(df)
    assert len(out) == 50
    assert out.dtype == bool
    assert not out.isna().any()


def test_nan_in_features_are_handled():
    df = _make_trades(100)
    df.loc[df.index[10], "amount"] = np.nan
    out = classify_side(df)
    assert len(out) == 100
    assert not out.isna().any()


def test_missing_required_column_raises():
    df = _make_trades(10).drop(columns="amount")
    with pytest.raises(KeyError):
        classify_side(df)


def test_non_dataframe_raises():
    with pytest.raises(TypeError):
        classify_side([1, 2, 3])


# ── Determinism ───────────────────────────────────────────────────────────────

def test_deterministic(trades):
    a = classify_side(trades)
    b = classify_side(trades)
    pd.testing.assert_series_equal(a, b)
