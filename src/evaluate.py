"""
Evaluation harness: aggregate metrics + breakdowns by size, time, gap, symbol.

Primary entry point: full_report() / print_report().
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


# ── Core metrics ──────────────────────────────────────────────────────────────

def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute accuracy, macro-F1, per-class F1, and balanced accuracy.

    Returns a plain dict so callers can build DataFrames easily.
    """
    y_t = y_true.astype(bool)
    y_p = y_pred.astype(bool)
    return {
        "n":                  len(y_t),
        "accuracy":           accuracy_score(y_t, y_p),
        "macro_f1":           f1_score(y_t, y_p, average="macro"),
        "buy_f1":             f1_score(y_t, y_p, pos_label=False, average="binary"),
        "sell_f1":            f1_score(y_t, y_p, pos_label=True,  average="binary"),
        "balanced_accuracy":  balanced_accuracy_score(y_t, y_p),
    }


# ── Breakdown helpers ─────────────────────────────────────────────────────────

def _size_groups(amount: pd.Series) -> pd.Series:
    """Quantile-based size buckets: small / medium / block."""
    try:
        return pd.qcut(amount, 3, labels=["small", "medium", "block"])
    except ValueError:
        # degenerate case: all amounts identical
        return pd.Series("small", index=amount.index, dtype="category")


def _time_groups(index: pd.DatetimeIndex) -> pd.Series:
    """UTC hour bands: 00-08 / 08-16 / 16-24."""
    hour = pd.Series(index.hour, index=index)
    return pd.cut(
        hour,
        bins=[0, 8, 16, 24],
        right=False,
        labels=["00-08 UTC", "08-16 UTC", "16-24 UTC"],
    )


def _gap_groups(index: pd.DatetimeIndex) -> pd.Series:
    """
    Inter-trade gap buckets: busy (short gaps) / normal / quiet (long gaps).
    Uses quantiles on seconds between consecutive trades.
    """
    dt = pd.Series(index, index=index).diff().dt.total_seconds().fillna(0)
    try:
        return pd.qcut(dt, 3, labels=["busy", "normal", "quiet"], duplicates="drop")
    except ValueError:
        return pd.Series("normal", index=index, dtype="category")


def _breakdown(y_true: pd.Series, y_pred: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Run metrics() on each group and return a DataFrame indexed by group label."""
    records = []
    for g in groups.cat.categories:
        mask = (groups == g).values
        if mask.sum() < 2:
            continue
        m = metrics(y_true.iloc[mask] if hasattr(y_true, "iloc") else y_true[mask],
                    y_pred.iloc[mask] if hasattr(y_pred, "iloc") else y_pred[mask])
        records.append({"group": str(g), **m})
    return pd.DataFrame(records).set_index("group") if records else pd.DataFrame()


# ── Public breakdown functions ────────────────────────────────────────────────

def breakdown_by_size(
    trades: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    """Metrics broken down by trade-size bucket (small / medium / block)."""
    return _breakdown(y_true, y_pred, _size_groups(trades["amount"]))


def breakdown_by_time(
    trades: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    """Metrics broken down by UTC hour band."""
    return _breakdown(y_true, y_pred, _time_groups(trades.index))


def breakdown_by_gap(
    trades: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    """Metrics broken down by inter-trade time gap (busy / normal / quiet)."""
    return _breakdown(y_true, y_pred, _gap_groups(trades.index))


# ── Full report ───────────────────────────────────────────────────────────────

def full_report(
    trades: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    symbol: str | None = None,
) -> dict:
    """
    Run all evaluations and return a report dict.

    Keys
    ----
    symbol, aggregate, confusion_matrix, by_size, by_time, by_gap
    """
    # drop rows where either label is missing — keep the DatetimeIndex intact
    # (the time/gap breakdowns rely on it; breakdown masks are positional)
    mask = (y_pred.notna() & y_true.notna()).to_numpy()
    y_t = y_true[mask]
    y_p = y_pred[mask]
    tr  = trades[mask]

    return {
        "symbol":           symbol,
        "aggregate":        metrics(y_t, y_p),
        "confusion_matrix": confusion_matrix(y_t.astype(bool), y_p.astype(bool)),
        "by_size":          breakdown_by_size(tr, y_t, y_p),
        "by_time":          breakdown_by_time(tr, y_t, y_p),
        "by_gap":           breakdown_by_gap(tr, y_t, y_p),
    }


def print_report(report: dict) -> None:
    """Pretty-print a report produced by full_report()."""
    sep = "=" * 62
    sym = report.get("symbol") or ""
    header = f"  {sym}" if sym else ""
    print(f"\n{sep}{header}")

    agg = report["aggregate"]
    print(
        f"  n={agg['n']:>9,}  "
        f"acc={agg['accuracy']:.4f}  "
        f"macro_f1={agg['macro_f1']:.4f}  "
        f"bal_acc={agg['balanced_accuracy']:.4f}"
    )
    print(f"  buy_f1={agg['buy_f1']:.4f}   sell_f1={agg['sell_f1']:.4f}")

    cm = report["confusion_matrix"]
    print("\n  Confusion matrix  (rows=true, cols=pred)  [Buy=False, Sell=True]")
    print(f"            pred_Buy  pred_Sell")
    print(f"  true_Buy   {cm[0,0]:>8,}  {cm[0,1]:>9,}")
    print(f"  true_Sell  {cm[1,0]:>8,}  {cm[1,1]:>9,}")

    for label, key in [
        ("By trade size", "by_size"),
        ("By time of day", "by_time"),
        ("By inter-trade gap", "by_gap"),
    ]:
        df = report[key]
        if df.empty:
            continue
        print(f"\n  {label}:")
        cols = ["n", "accuracy", "macro_f1"]
        print(
            df[cols]
            .rename(columns={"accuracy": "acc", "macro_f1": "mac_f1"})
            .to_string(
                float_format=lambda x: f"{x:.4f}",
                formatters={"n": lambda x: f"{int(x):>8,}"},
            )
        )
    print()


# ── Convenience: compare multiple classifiers ─────────────────────────────────

def compare_classifiers(
    trades: pd.DataFrame,
    y_true: pd.Series,
    predictions: dict,
) -> pd.DataFrame:
    """
    Compare several classifiers in one table.

    Parameters
    ----------
    predictions : dict[str, pd.Series]
        {name: predicted_side_series}

    Returns
    -------
    pd.DataFrame  indexed by classifier name
    """
    records = []
    for name, y_pred in predictions.items():
        mask = y_pred.notna() & y_true.notna()
        m = metrics(y_true[mask], y_pred[mask])
        records.append({"classifier": name, **m})
    return pd.DataFrame(records).set_index("classifier")
