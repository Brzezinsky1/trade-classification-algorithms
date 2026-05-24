
from __future__ import annotations

from pathlib import Path
import json
import joblib
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


