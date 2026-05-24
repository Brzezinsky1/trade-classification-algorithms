# ATS Final Project — Trade-Side Classification

## 1. Project description

Every print on an exchange has an **aggressor**: the side that crossed the spread to take liquidity. The other side was resting on the book. The exchange's public trade feed often does **not** tell you which side was the aggressor — you only see price, size, and time. Reconstructing that label is the *trade-side classification* problem, and it matters because the sign of order flow drives almost every market-microstructure metric (VPIN, Kyle's lambda, order-flow imbalance, effective spread, etc.).

Our task: build a function that takes a `DataFrame` of trades (price, volume, time) and returns a boolean `Series` of aggressor sides — and beats the classical rule-based baselines.

### Convention
- `True`  → **sell aggressor** (someone hit the bid)
- `False` → **buy aggressor**  (someone lifted the ask)

This matches the `side` column in the sample data.

### Deliverable

```python
def classify_side(trades: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    trades : pd.DataFrame
        Time-indexed (DatetimeIndex, UTC). Must contain at least:
          - 'price'  : float
          - 'amount' : float (trade size)

    Returns
    -------
    pd.Series
        Boolean, aligned 1-to-1 with `trades.index`.
        True  = sell aggressor
        False = buy aggressor
    """
```

## 2. Data

Located in [task_data/](task_data/):

| Symbol     | Days                         | Files per day                      |
|------------|------------------------------|------------------------------------|
| WIFUSDT    | 2026-04-12, 04-13, 04-14     | `_trades_*.parquet`, `_orderbook_*.parquet` |
| ZAMAUSDT   | 2026-04-12, 04-13, 04-14     | `_trades_*.parquet`, `_orderbook_*.parquet` |

- **Trades** (~10k rows/day/symbol): `price`, `amount`, `side` (ground-truth label), `timestamp`.
- **Order books** (~127k snapshots/day/symbol): L2 with 10 levels each side (`ask0..ask9`, `askv0..askv9`, `bid0..bid9`, `bidv0..bidv9`).

The order book is an **oracle for development** — we use it to compute the "true" Lee–Ready / quote-rule labels, to validate, and to engineer features that we *learn from* but cannot *use at inference*. The professor's evaluation only feeds trades to our function.

Label balance is roughly 52/48, so accuracy is a meaningful metric (we don't have a degenerate prior).

## 3. Baslines to beat

These are the classical rules; our model has to clearly outperform them or motivate why our approach is better:

| Baseline       | Idea                                                                                  |
|----------------|----------------------------------------------------------------------------------------|
| **Tick rule**  | Uptick → buy aggressor, downtick → sell aggressor, zero-tick → carry the previous sign |
| **Quote rule** | Above midpoint → buy, below → sell, at mid → unclassified (needs quotes — book oracle)|
| **Lee–Ready**  | Quote rule when clearly above/below mid; tick rule at the midpoint                    |

The tick rule and BVC need only trades, so they are *fair* baselines under our constraint. Lee–Ready/Quote rule/EMO need quotes — they're upper-bound references computed via the order book.

## 4. How we plan to beat them

Three complementary angles:

1. **Better features from trades only** — micro-momentum, run-length, time-deltas, volume signatures, log-returns at multiple horizons, round-number price proximity, trade-clustering / burstiness.
2. **A model that learns serial dependence** — aggressor sides are auto-correlated (a market order often eats multiple resting orders, producing runs of same-side prints). A sequence model (HMM / CRF / small LSTM or 1D-CNN) should beat per-trade independent classifiers.

Training/validation/test split:
- **Train**: 2026-04-12 (both symbols)
- **Validation**: 2026-04-13
- **Hold-out test**: 2026-04-14 (touch only at the end)

We additionally simulate "professor's data" by training on one symbol and testing on the other to check cross-symbol generalisation.

## 5. Repo structure (reference)

```
ATS-Final-Project/
├── README.md
├── task_data/                      # raw parquet files (already here)
├── src/
│   ├── __init__.py
│   ├── classify_side.py            # the public API the professor imports
│   ├── baselines.py                # tick rule, Lee-Ready, EMO, BVC
│   ├── features.py                 # all feature engineering on trades
│   ├── models/
│   │   ├── gbm.py                  # LightGBM / XGBoost classifier
│   │   ├── sequence.py             # HMM / LSTM / CRF sequence model
│   │   └── dt.py                   # Decision Tree model
│   ├── data.py                     # loaders, splits, label utilities
│   └── evaluate.py                 # metrics, per-bucket breakdowns
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_features_and_gbm.ipynb
│   └── 04_sequence.ipynb
├── artifacts/                      # trained model files (loaded by classify_side)
├── tests/
│   └── test_classify_side.py       # contract / sanity tests
└── report/
    └── report.pdf
```
