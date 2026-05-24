"""
Cross-validation splitters appropriate for autocorrelated trade time-series.

Two structure-respecting schemes are provided:

1. ``leave_one_symbol_out`` — group CV by instrument. Train on all-but-one symbol,
   test on the held-out symbol. This directly mirrors the professor's "unseen data"
   scenario and is the most defensible scheme given we only have two symbols.

2. ``PurgedBlockSplit`` — contiguous-block k-fold over a time-ordered sequence, with
   an embargo gap removed from the training set around each test block so no window
   straddles the boundary. Use this for intra-symbol CV (e.g. the meta-learner).
"""

from __future__ import annotations

import numpy as np


class PurgedBlockSplit:
    """
    Contiguous-block k-fold for a single time-ordered sequence.

    Parameters
    ----------
    n_splits : int
        Number of contiguous test blocks.
    embargo : int
        Rows removed from the training set on each side of the test block to
        prevent rolling-window leakage across the boundary.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 50):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, n: int):
        """Yield (train_idx, test_idx) integer arrays for a sequence of length n."""
        idx = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.concatenate([[0], np.cumsum(fold_sizes)])

        for k in range(self.n_splits):
            test_start, test_end = starts[k], starts[k + 1]
            if test_end <= test_start:
                continue
            test_idx = idx[test_start:test_end]
            lo = max(0, test_start - self.embargo)
            hi = min(n, test_end + self.embargo)
            train_mask = np.ones(n, dtype=bool)
            train_mask[lo:hi] = False
            train_idx = idx[train_mask]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx


def leave_one_symbol_out(frames: dict):
    """
    Group CV by symbol.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        {symbol: trades_df}

    Yields
    ------
    (test_symbol, train_frames, test_frame)
        ``train_frames`` is a list of the other symbols' DataFrames.
    """
    keys = list(frames)
    if len(keys) < 2:
        raise ValueError("leave_one_symbol_out needs at least 2 symbols")
    for test_key in keys:
        train_frames = [frames[k] for k in keys if k != test_key]
        yield test_key, train_frames, frames[test_key]


def purged_oof_proba(estimator_factory, X: np.ndarray, y: np.ndarray,
                     n_splits: int = 5, embargo: int = 50) -> np.ndarray:
    """
    Out-of-fold positive-class probabilities using PurgedBlockSplit.

    ``estimator_factory`` is a zero-arg callable returning a fresh estimator with
    ``fit`` and ``predict_proba``. Every row is predicted by a model that never saw
    it (nor its embargoed neighbours) during training.
    """
    oof = np.full(len(y), np.nan)
    splitter = PurgedBlockSplit(n_splits=n_splits, embargo=embargo)
    for tr, te in splitter.split(len(y)):
        est = estimator_factory()
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, 1]
    return oof
