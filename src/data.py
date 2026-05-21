"""Data loading, train/val/test splits, and oracle label reconstruction."""

from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["WIFUSDT", "ZAMAUSDT"]

DATES = {
    "train": "2026-04-12",
    "val":   "2026-04-13",
    "test":  "2026-04-14",
}

DATA_DIR = Path(__file__).parent.parent / "task_data"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_trades(symbol: str, date: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load trades parquet for a given symbol and date."""
    path = data_dir / symbol / f"{symbol}_trades_{date}.parquet"
    return pd.read_parquet(path)


def load_orderbook(symbol: str, date: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load order book parquet for a given symbol and date."""
    path = data_dir / symbol / f"{symbol}_orderbook_{date}.parquet"
    return pd.read_parquet(path)


def load_split(
    split: str,
    symbols: list = SYMBOLS,
    data_dir: Path = DATA_DIR,
) -> dict:
    """
    Load trades for a named split.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.

    Returns
    -------
    dict[str, pd.DataFrame]
        {symbol: trades_df}
    """
    date = DATES[split]
    return {sym: load_trades(sym, date, data_dir) for sym in symbols}


def load_split_with_books(
    split: str,
    symbols: list = SYMBOLS,
    data_dir: Path = DATA_DIR,
) -> dict:
    """
    Load both trades and order books for a named split.

    Returns
    -------
    dict[str, dict]
        {symbol: {"trades": df, "orderbook": df}}
    """
    date = DATES[split]
    return {
        sym: {
            "trades":    load_trades(sym, date, data_dir),
            "orderbook": load_orderbook(sym, date, data_dir),
        }
        for sym in symbols
    }


def load_all_trades(symbols: list = SYMBOLS, data_dir: Path = DATA_DIR) -> dict:
    """
    Load all trades across all splits and symbols.

    Returns
    -------
    dict[tuple[str, str], pd.DataFrame]
        {(symbol, date): trades_df}
    """
    return {
        (sym, date): load_trades(sym, date, data_dir)
        for sym in symbols
        for date in DATES.values()
    }


# ── Order-book helpers ────────────────────────────────────────────────────────

def compute_midpoint(orderbook: pd.DataFrame) -> pd.Series:
    """Best-bid/ask midpoint from the top-of-book."""
    return (orderbook["ask0"] + orderbook["bid0"]) / 2


def attach_midpoint(trades: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
    """
    Backward asof-join: attach the most recent midpoint to each trade.

    Adds a 'midpoint' column; original columns are preserved.
    Both DataFrames must have a sorted DatetimeIndex.
    """
    mid = compute_midpoint(orderbook).rename("midpoint")
    return pd.merge_asof(
        trades,
        mid.to_frame(),
        left_index=True,
        right_index=True,
        direction="backward",
    )


# ── Oracle label reconstruction ───────────────────────────────────────────────

def reconstruct_lee_ready_labels(
    trades: pd.DataFrame, orderbook: pd.DataFrame
) -> pd.Series:
    """
    Compute oracle Lee-Ready labels using the order book.

    Rule:
      price < midpoint  → True  (sell aggressor)
      price > midpoint  → False (buy aggressor)
      price = midpoint  → tick rule fallback
      no prior tick     → NaN   (indeterminate)

    Returns
    -------
    pd.Series
        float (True/False/NaN), same index as trades.
    """
    merged = attach_midpoint(trades, orderbook)
    mid = merged["midpoint"]
    price = trades["price"]

    tick = np.sign(price.diff()).replace(0, np.nan).ffill()
    at_mid = price == mid

    labels = pd.Series(
        np.select(
            [price < mid,
             price > mid,
             at_mid & (tick < 0),
             at_mid & (tick > 0)],
            [True, False, True, False],
            default=np.nan,
        ),
        index=trades.index,
        name="lee_ready",
    )
    return labels
