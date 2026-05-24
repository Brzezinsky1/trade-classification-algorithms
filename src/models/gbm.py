
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ..features import build_features, FEATURE_NAMES
from ..data import load_split
from ..evaluate import metrics

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_trade_side.joblib"
META_PATH = ARTIFACT_DIR / "xgb_trade_side_meta.json"


def make_xy(split: str):
    frames = []
    labels = []

    for symbol, trades in load_split(split).items():
        X = build_features(trades)[FEATURE_NAMES]
        y = trades["side"].astype(bool)

        frames.append(X)
        labels.append(y)

    X_all = pd.concat(frames, axis=0)
    y_all = pd.concat(labels, axis=0)

    return X_all, y_all


def train_gbm():
    X_train, y_train = make_xy("train")
    X_val, y_val = make_xy("val")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_pred = pd.Series(
        model.predict(X_val).astype(bool),
        index=y_val.index,
        name="predicted_side",
    )

    val_metrics = metrics(y_val, val_pred)

    ARTIFACT_DIR.mkdir(exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
    }

    joblib.dump(bundle, MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    print("Validation metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    return model, val_metrics


def _make_symbol_frames(splits=("train", "val")):
    """Pool the given splits per symbol → {symbol: (X, y)} with no-look-ahead features."""
    from ..data import SYMBOLS
    frames = {sym: [] for sym in SYMBOLS}
    for split in splits:
        for sym, trades in load_split(split).items():
            frames[sym].append(trades)
    out = {}
    for sym, dfs in frames.items():
        merged = pd.concat(dfs)
        out[sym] = (build_features(merged)[FEATURE_NAMES], merged["side"].astype(int))
    return out


def _new_model(**overrides):
    params = dict(
        n_estimators=150, max_depth=3, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def cross_val_report(verbose: bool = True) -> dict:
    """
    Leave-one-symbol-out CV — honest out-of-sample XGBoost metrics on an unseen symbol.

    Trains on all-but-one symbol (train+val pooled), evaluates on the held-out symbol.
    Returns {held_out_symbol: metrics}.
    """
    from ..cv import leave_one_symbol_out

    frames = _make_symbol_frames(("train", "val"))
    results = {}
    for test_sym, train_xy, (X_te, y_te) in leave_one_symbol_out(frames):
        X_tr = pd.concat([xy[0] for xy in train_xy])
        y_tr = pd.concat([xy[1] for xy in train_xy])
        model = _new_model()
        model.fit(X_tr, y_tr)
        pred = pd.Series(model.predict(X_te).astype(bool), index=y_te.index)
        results[test_sym] = metrics(y_te.astype(bool), pred)
        if verbose:
            m = results[test_sym]
            print(f"[GBM CV] hold-out {test_sym:10s}  "
                  f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
    return results


# small, fast grid — extend if you have compute budget
DEFAULT_GRID = [
    {"max_depth": 3, "learning_rate": 0.03},
    {"max_depth": 3, "learning_rate": 0.10},
    {"max_depth": 5, "learning_rate": 0.03},
    {"max_depth": 5, "learning_rate": 0.10},
    {"max_depth": 7, "learning_rate": 0.05},
]


def tune_gbm(grid=None, n_estimators: int = 400, save: bool = True, verbose: bool = True):
    """
    Cross-symbol CV hyperparameter search.

    For each parameter combo, run leave-one-symbol-out CV and score by mean macro-F1.
    The best combo is refit on the production scheme (train day, both symbols) with
    early stopping on the validation day, and — if ``save`` — written to artifacts,
    replacing the hand-tuned model.

    Returns (best_params, leaderboard_df).
    """
    from ..cv import leave_one_symbol_out

    grid = grid or DEFAULT_GRID
    frames = _make_symbol_frames(("train", "val"))

    leaderboard = []
    for combo in grid:
        fold_f1, fold_iters = [], []
        for test_sym, train_xy, (X_te, y_te) in leave_one_symbol_out(frames):
            X_tr = pd.concat([xy[0] for xy in train_xy])
            y_tr = pd.concat([xy[1] for xy in train_xy])
            # per-fold early stopping on a time-tail of the training data, so the CV
            # metric reflects the same early-stopped model that gets saved (not a
            # fixed, over-grown tree count that overfits).
            cut = int(len(X_tr) * 0.85)
            model = _new_model(n_estimators=n_estimators, early_stopping_rounds=30, **combo)
            model.fit(X_tr.iloc[:cut], y_tr.iloc[:cut],
                      eval_set=[(X_tr.iloc[cut:], y_tr.iloc[cut:])], verbose=False)
            pred = pd.Series(model.predict(X_te).astype(bool), index=y_te.index)
            fold_f1.append(metrics(y_te.astype(bool), pred)["macro_f1"])
            fold_iters.append(model.best_iteration)
        mean_f1 = float(np.mean(fold_f1))
        leaderboard.append({**combo, "best_iter_mean": float(np.mean(fold_iters)),
                            "cv_macro_f1": mean_f1})
        if verbose:
            print(f"[tune] {combo}  cv_macro_f1={mean_f1:.4f}  "
                  f"(mean best_iter={np.mean(fold_iters):.0f})")

    leaderboard = pd.DataFrame(leaderboard).sort_values("cv_macro_f1", ascending=False)
    best = leaderboard.iloc[0].to_dict()
    best_params = {"max_depth": int(best["max_depth"]),
                   "learning_rate": float(best["learning_rate"])}

    if verbose:
        print(f"\nbest params: {best_params}  (cv_macro_f1={best['cv_macro_f1']:.4f})")

    if save:
        X_train, y_train = make_xy("train")
        X_val, y_val = make_xy("val")
        model = _new_model(n_estimators=n_estimators, early_stopping_rounds=30, **best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = pd.Series(model.predict(X_val).astype(bool), index=y_val.index)
        val_metrics = metrics(y_val, val_pred)
        ARTIFACT_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "threshold": 0.5}, MODEL_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)
        if verbose:
            print(f"saved tuned model (best_iteration={model.best_iteration})  "
                  f"val acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

    return best_params, leaderboard


def artifacts_exist() -> bool:
    return MODEL_PATH.exists()


def load_artifacts():
    return joblib.load(MODEL_PATH)


def predict_proba(bundle, trades: pd.DataFrame) -> pd.Series:
    X = build_features(trades)[bundle["feature_names"]]

    proba_sell = bundle["model"].predict_proba(X)[:, 1]

    return pd.Series(
        proba_sell,
        index=trades.index,
        name="sell_probability",
    )


def predict(bundle, trades: pd.DataFrame) -> pd.Series:
    proba = predict_proba(bundle, trades)
    threshold = bundle.get("threshold", 0.5)

    return (proba >= threshold).astype(bool).rename("predicted_side")


if __name__ == "__main__":
    train_gbm()


