"""
Sequence-aware trade-side classifier: a causal LSTM over windows of recent trades.

Aggressor sides are autocorrelated — a market order often sweeps several resting
orders, producing runs of same-side prints — so a model that sees the recent
*sequence* of trades should beat a per-trade independent classifier.

Artifacts (written to ``artifacts/sequence/``)
- ``model.pt``     — model weights (state_dict)
- ``scaler.npz``   — per-feature mean/scale for standardisation
- ``config.json``  — feature list, window size, and architecture hyper-parameters

Public entry points
- ``train_sequence_model(...)`` — fit and save artifacts
- ``load_artifacts(...)``       — load a trained bundle for inference
- ``predict_proba(bundle, trades)`` — sell-aggressor probabilities aligned to index
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..features import build_features

# ── Default feature set for the sequence model ────────────────────────────────
# Trade-only, price-*relative* features (no absolute-price features) so the model
# generalises across symbols.
SEQUENCE_FEATURES = [
    "tick_ff", "tick_1", "tick_2", "tick_3",
    "ret_1", "ret_3", "ret_10",
    "dt_s", "log_dt_s",
    "vol_z_10", "vol_z_50", "log_amount",
    "run_length", "run_dir",
    "pct_range_20", "pct_range_50",
]

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "sequence"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SequenceConfig:
    features: list = field(default_factory=lambda: list(SEQUENCE_FEATURES))
    window: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 25
    patience: int = 5
    seed: int = 42

    @property
    def n_features(self) -> int:
        return len(self.features)


# ── Model ─────────────────────────────────────────────────────────────────────

class TradeLSTM(nn.Module):
    """Causal LSTM → MLP head producing a single sell-aggressor logit per window."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, W, H)
        last = out[:, -1, :]           # (B, H) — last timestep = the trade being classified
        return self.head(last).squeeze(-1)  # (B,) logits


# ── Feature preparation & windowing ───────────────────────────────────────────

def _prepare_matrix(
    trades: pd.DataFrame,
    features: list,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple:
    """
    Build the feature matrix, standardise, and fill NaNs (window-boundary rows) with 0.

    If mean/scale are None they are computed (training); otherwise applied (inference).
    Returns (matrix float32 (N, F), mean, scale).
    """
    feats = build_features(trades)[features].to_numpy(dtype=np.float64)

    if mean is None or scale is None:
        mean = np.nanmean(feats, axis=0)
        scale = np.nanstd(feats, axis=0)
        scale[scale == 0] = 1.0

    feats = (feats - mean) / scale
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32), mean, scale


class _WindowDataset(Dataset):
    """Yields (window_tensor, label) pairs. Windows are materialised lazily per item."""

    def __init__(self, matrix: np.ndarray, labels: np.ndarray | None, window: int):
        self.matrix = matrix
        self.labels = labels
        self.window = window
        self.n = len(matrix)
        self._padded = np.zeros((window - 1 + self.n, matrix.shape[1]), dtype=np.float32)
        self._padded[window - 1:] = matrix

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        w = self._padded[i:i + self.window]
        x = torch.from_numpy(w)
        if self.labels is None:
            return x
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def _build_pooled_dataset(
    groups: list, config: SequenceConfig, mean=None, scale=None
) -> tuple:
    """
    Build a concatenated dataset across (trades_df) groups, windowing *within* each
    group so windows never span a symbol/day boundary.

    ``groups`` is a list of trades DataFrames (each must contain a 'side' column).
    Returns (concat_matrix, concat_labels, mean, scale) — windows are built per group
    via per-group padding inside _WindowDataset, so we keep a list of datasets instead.
    """
    mats, labels = [], []
    for df in groups:
        m, mean, scale = _prepare_matrix(df, config.features, mean, scale)
        mats.append(m)
        labels.append(df["side"].to_numpy(dtype=np.float32))
    return mats, labels, mean, scale


def _fit(
    train_groups: list,
    val_groups: list,
    config: SequenceConfig,
    verbose: bool = True,
) -> tuple:
    """
    Core training loop — fit the LSTM and return ``(model, mean, scale, history)``
    *without* saving artifacts. Shared by ``train_sequence_model`` (which saves) and
    ``cross_val_report`` (which does not, to avoid clobbering the production model).
    """
    from torch.utils.data import ConcatDataset

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # fit scaler on training data only
    train_mats, train_labels, mean, scale = _build_pooled_dataset(train_groups, config)
    val_mats, val_labels, _, _ = _build_pooled_dataset(val_groups, config, mean, scale)

    train_ds = ConcatDataset([
        _WindowDataset(m, y, config.window) for m, y in zip(train_mats, train_labels)
    ])
    val_ds = ConcatDataset([
        _WindowDataset(m, y, config.window) for m, y in zip(val_mats, val_labels)
    ])

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeLSTM(config.n_features, config.hidden_size, config.num_layers, config.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_ds)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * len(xb)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += len(xb)
        val_loss /= len(val_ds)
        val_acc = correct / total

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"epoch {epoch:2d}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val
    history["final_val_acc"] = history["val_acc"][-1] if history["val_acc"] else float("nan")
    return model, mean, scale, history


def train_sequence_model(
    train_groups: list,
    val_groups: list,
    config: SequenceConfig | None = None,
    artifact_dir: Path = ARTIFACT_DIR,
    verbose: bool = True,
) -> dict:
    """
    Train the LSTM on pooled trade groups and **save** artifacts to ``artifact_dir``.

    Parameters
    ----------
    train_groups, val_groups : list[pd.DataFrame]
        Each DataFrame is one symbol-day of trades with a 'side' column.
        Windows are built *within* each group (no cross-group leakage).

    Returns
    -------
    dict  training history + final val metrics.
    """
    config = config or SequenceConfig()
    model, mean, scale, history = _fit(train_groups, val_groups, config, verbose=verbose)
    _save_artifacts(model, mean, scale, config, artifact_dir)
    return history


def cross_val_report(
    symbol_frames: dict,
    config: SequenceConfig | None = None,
    val_fraction: float = 0.15,
    verbose: bool = True,
) -> dict:
    """
    Leave-one-symbol-out CV — an honest out-of-sample estimate of LSTM generalisation
    to an unseen instrument (the professor's scenario).

    For each held-out symbol, the model is trained on the *other* symbol(s); a
    time-tail (`val_fraction`) of each training symbol is reserved for early stopping.
    No production artifacts are written.

    Parameters
    ----------
    symbol_frames : dict[str, pd.DataFrame]
        {symbol: trades_df} — typically the train+val days pooled per symbol.

    Returns
    -------
    dict[str, dict]  {held_out_symbol: metrics}
    """
    from ..cv import leave_one_symbol_out
    from ..evaluate import metrics

    config = config or SequenceConfig()
    results = {}

    for test_sym, train_frames, test_df in leave_one_symbol_out(symbol_frames):
        # carve a time-ordered tail of each training symbol for early stopping
        tr_groups, va_groups = [], []
        for df in train_frames:
            cut = max(config.window + 1, int(len(df) * (1 - val_fraction)))
            tr_groups.append(df.iloc[:cut])
            va_groups.append(df.iloc[cut:])

        model, mean, scale, _ = _fit(tr_groups, va_groups, config, verbose=False)
        bundle = {"model": model, "mean": mean, "scale": scale, "config": config}
        proba = predict_proba(bundle, test_df)
        m = metrics(test_df["side"].astype(bool), proba >= 0.5)
        results[test_sym] = m
        if verbose:
            print(f"[LSTM CV] hold-out {test_sym:10s}  "
                  f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    return results


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def _save_artifacts(model, mean, scale, config: SequenceConfig, artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    np.savez(artifact_dir / "scaler.npz", mean=mean, scale=scale)
    with open(artifact_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> dict:
    """
    Load a trained bundle for inference.

    Returns dict with keys: model (eval mode), mean, scale, config.
    """
    artifact_dir = Path(artifact_dir)
    with open(artifact_dir / "config.json") as f:
        config = SequenceConfig(**json.load(f))

    model = TradeLSTM(config.n_features, config.hidden_size, config.num_layers, config.dropout)
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
    model.eval()

    scaler = np.load(artifact_dir / "scaler.npz")
    return {"model": model, "mean": scaler["mean"], "scale": scaler["scale"], "config": config}


def artifacts_exist(artifact_dir: Path = ARTIFACT_DIR) -> bool:
    artifact_dir = Path(artifact_dir)
    return all((artifact_dir / f).exists() for f in ("model.pt", "scaler.npz", "config.json"))


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_proba(bundle: dict, trades: pd.DataFrame, batch_size: int = 4096) -> pd.Series:
    """
    Sell-aggressor probability for each trade, aligned to ``trades.index``.

    ``bundle`` is the dict returned by ``load_artifacts``.
    """
    config: SequenceConfig = bundle["config"]
    matrix, _, _ = _prepare_matrix(trades, config.features, bundle["mean"], bundle["scale"])
    ds = _WindowDataset(matrix, None, config.window)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = bundle["model"]
    model.eval()
    probs = []
    for xb in dl:
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.array([])
    return pd.Series(probs, index=trades.index, name="sell_proba")
