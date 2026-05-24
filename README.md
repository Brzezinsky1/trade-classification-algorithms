# ATS Final Project — Trade-Side Classification

## 1. Project description

Every print on an exchange has an **aggressor**: the side that crossed the spread to take liquidity. The other side was resting on the book. The exchange's public trade feed often does **not** tell you which side was the aggressor — you only see price, size, and time. Reconstructing that label is the *trade-side classification* problem, and it matters because the sign of order flow drives almost every market-microstructure metric (VPIN, Kyle's lambda, order-flow imbalance, effective spread, etc.).

Our task: build a function that takes a `DataFrame` of trades (price, volume, time) and returns a boolean `Series` of aggressor sides — and beats the classical rule-based baselines.

### Convention
- `True`  → **sell aggressor** (someone hit the bid)
- `False` → **buy aggressor**  (someone lifted the ask)

This matches the `side` column in the sample data.

### Deliverable (the professor's interface)

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

The professor will `import` this function and run it on **his own unseen trade data**, so the function must:
- Work with trades **only** (no order book, no quotes at inference time).
- Be self-contained — any model artifacts must be loaded inside the package.
- Run in reasonable time on a day's worth of trades.

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
3. **Ensembling** — stack the sequence model on top of a gradient-boosting model and the rule baselines, calibrate, and pick a threshold on validation.

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
│   └── 04_sequence_and_ensemble.ipynb
├── artifacts/                      # trained model files (loaded by classify_side)
├── tests/
│   └── test_classify_side.py       # contract / sanity tests
└── report/
    └── report.pdf
```

## 6. Work split

### Michał — *Data, Baselines, Evaluation*

**Owns the foundation everyone else builds on.**

- Build [src/data.py](src/data.py): loaders for trades and order books, the train/val/test split, and a function that reconstructs the **quote-rule / Lee–Ready labels** from the order book (this is the oracle Member B & C train against where useful).
- Implement all baselines in [src/baselines.py](src/baselines.py):
  - Tick rule, Quote rule, Lee–Ready.
  - Each baseline = a `classify_side`-compatible function so we can plug them into the same evaluation harness.
- Build [src/evaluate.py](src/evaluate.py): accuracy, F1, balanced accuracy, confusion matrix, plus **breakdowns** by:
  - trade size bucket (small / medium / block),
  - time-of-day,
  - inter-trade gap (busy vs quiet),
  - symbol.
  Breakdowns are how we *justify* a model that wins on one regime but ties on another.
[notebooks/02_baselines.ipynb](notebooks/02_baselines.ipynb): exploratory plots and baseline numbers everyone refers back to.
- [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
- Owns the "Baselines & Evaluation" section of the final report.
- Builds Decision Tree model 

### Benek — *Feature Engineering & XGBoost Model*

**Builds the feature-engineering pipeline and the main classical ML classifier.**

- Implemented [src/features.py](src/features.py): a full trade-only feature engineering pipeline using strict no-look-ahead logic.
- Engineered features capturing:
  - tick direction and multi-lag ticks,
  - short-term log returns,
  - inter-trade timing,
  - rolling volume z-scores,
  - streak persistence and run direction,
  - local-range positioning,
  - round-number proximity effects.
- Trained an XGBoost classifier in [src/models/gbm.py](src/models/gbm.py) using only `(price, amount, timestamp)` information.
- Built the full training / validation pipeline with chronological out-of-sample evaluation:
  - Train: 2026-04-12
  - Validation: 2026-04-13
  - Hold-out Test: 2026-04-14
- Performed feature-importance analysis and model interpretation.
- Added notebook experiments and model comparison in [notebooks/03_features_and_gbm.ipynb](notebooks/03_features_and_gbm.ipynb).

**Key findings**
- The Tick Rule remained an extremely strong trades-only baseline.
- XGBoost achieved comparable out-of-sample performance while learning nonlinear trade-flow dynamics.
- Feature-importance analysis showed that:
  - `run_dir` (current order-flow direction),
  - `tick_ff` (forward-filled tick direction),
  
  were the dominant predictive signals.
- Results suggest strong order-flow persistence and short-term aggressor autocorrelation in crypto trade flow.

### Krzysiek — *Sequence Model, Ensemble & API*

**Captures temporal structure and ships the deliverable.**

- [src/models/sequence.py](src/models/sequence.py): a sequence-aware model — LSTM over windows of recent trades, using Member B's features as inputs.
- [src/classify_side.py](src/classify_side.py): the **single public function**. Loads artifacts from `artifacts/`, computes features, calls the ensemble, returns the boolean `Series`. This is what the professor imports — must be bullet-proof:
  - Handle DataFrames with or without an extra `side` column.
  - Handle a single-symbol input even though we trained on two.
  - Sensible behaviour on edge cases (first trade, ties, NaNs).
- [tests/test_classify_side.py](tests/test_classify_side.py): contract tests — output is a `pd.Series`, dtype is `bool`, length matches input, index matches input, runs on a small fixture.
- [notebooks/05_sequence.ipynb](notebooks/05_sequence.ipynb).
- [src/models/ensemble.py](src/models/ensemble.py): stack Benek's GBM and the sequence model (e.g. logistic regression on out-of-fold probabilities), calibrate, and pick the operating threshold against Member A's evaluation harness.
- [notebooks/06_ensemble.ipynb](notebooks/06_ensemble.ipynb).
- Owns the "Sequence Model, Ensemble & API" section of the final report.
