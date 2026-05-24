"""
Ensemble — stacks the XGBoost and LSTM sell-probabilities with a
logistic-regression meta-learner.

Meta-learner training
----------------------
The logistic regression is fit on the **validation** day's base probabilities
(pooled across symbols), since both base models were *trained* on the train day.
The operating threshold is then chosen on validation to maximise macro-F1.

Artifacts (``artifacts/ensemble/``)
- ``meta.joblib``  — the fitted LogisticRegression + chosen threshold
- ``config.json``  — threshold, base order, and the learned coefficients

Interface mirrors ``gbm`` and ``sequence``:
``artifacts_exist`` / ``load_artifacts`` / ``predict_proba`` / ``predict`` / ``train_ensemble``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..data import load_split
from ..evaluate import metrics
from . import gbm
from . import sequence as seq

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "ensemble"
META_PATH = ARTIFACT_DIR / "meta.joblib"
CONFIG_PATH = ARTIFACT_DIR / "config.json"

EMBARGO = 50  # rows purged around each OOF test block (rolling-window leakage guard)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_probs(gbm_bundle, seq_bundle, trades: pd.DataFrame) -> np.ndarray:
    """Stack the two base models' sell-probabilities into an (N, 2) matrix."""
    p_gbm = gbm.predict_proba(gbm_bundle, trades).to_numpy()
    p_lstm = seq.predict_proba(seq_bundle, trades).to_numpy()
    return np.column_stack([p_gbm, p_lstm])


def _best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple:
    """Sweep thresholds and return the one maximising macro-F1 on the given data."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.30, 0.70, 41):
        pred = pd.Series(proba >= t, index=y_true.index)
        f1 = metrics(y_true, pred)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ── Training ──────────────────────────────────────────────────────────────────

def train_ensemble(artifact_dir: Path = ARTIFACT_DIR, verbose: bool = True) -> dict:
    """
    Fit the logistic-regression meta-learner on validation base-probabilities and
    save artifacts. Both base models must already be trained.
    """
    from ..cv import purged_oof_proba

    gbm_bundle = gbm.load_artifacts()
    seq_bundle = seq.load_artifacts()

    # Build base-probability features per symbol so the OOF blocks never span a
    # symbol boundary, then pool.
    X_parts, y_parts, oof_parts = [], [], []
    for sym, trades in load_split("val").items():
        Xs = _base_probs(gbm_bundle, seq_bundle, trades)
        ys = trades["side"].astype(int).to_numpy()
        # honest out-of-fold meta probabilities (purged-block CV within the symbol)
        oof_parts.append(
            purged_oof_proba(lambda: LogisticRegression(max_iter=1000),
                             Xs, ys, n_splits=5, embargo=EMBARGO)
        )
        X_parts.append(Xs)
        y_parts.append(ys)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    oof = np.concatenate(oof_parts)

    # threshold chosen on OUT-OF-FOLD probabilities (no in-sample optimism)
    y_ser = pd.Series(y.astype(bool))
    threshold, val_f1 = _best_threshold(y_ser, oof)

    # final meta-learner fit on all validation rows
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X, y)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"meta": meta, "threshold": threshold}, META_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "base": ["gbm", "lstm"],
                "coef": meta.coef_.tolist(),
                "intercept": meta.intercept_.tolist(),
            },
            f,
            indent=2,
        )

    if verbose:
        w_gbm, w_lstm = meta.coef_[0]
        print(f"meta weights — gbm: {w_gbm:.3f}  lstm: {w_lstm:.3f}  "
              f"intercept: {meta.intercept_[0]:.3f}")
        print(f"chosen threshold (out-of-fold): {threshold:.3f}  "
              f"(OOF val macro-F1 {val_f1:.4f})")

    return {"meta": meta, "threshold": threshold, "val_macro_f1": val_f1}


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def artifacts_exist() -> bool:
    """True only if the meta-learner *and* both base models are available."""
    return META_PATH.exists() and gbm.artifacts_exist() and seq.artifacts_exist()


def load_artifacts() -> dict:
    """
    Load the ensemble bundle, including both base-model bundles so that
    ``predict_proba`` is self-contained.
    """
    saved = joblib.load(META_PATH)
    return {
        "meta": saved["meta"],
        "threshold": saved.get("threshold", 0.5),
        "gbm_bundle": gbm.load_artifacts(),
        "seq_bundle": seq.load_artifacts(),
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba(bundle: dict, trades: pd.DataFrame) -> pd.Series:
    """Ensemble sell-aggressor probability, aligned to ``trades.index``."""
    X = _base_probs(bundle["gbm_bundle"], bundle["seq_bundle"], trades)
    proba = bundle["meta"].predict_proba(X)[:, 1]
    return pd.Series(proba, index=trades.index, name="sell_probability")


def predict(bundle: dict, trades: pd.DataFrame) -> pd.Series:
    """Boolean sell/buy prediction using the bundle's tuned threshold."""
    proba = predict_proba(bundle, trades)
    threshold = bundle.get("threshold", 0.5)
    return (proba >= threshold).astype(bool).rename("predicted_side")


if __name__ == "__main__":
    train_ensemble()
