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

## 5. Repo structure (target)

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
- [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) + [notebooks/02_baselines.ipynb](notebooks/02_baselines.ipynb): exploratory plots and baseline numbers everyone refers back to.
- Owns the "Baselines & Evaluation" section of the final report.
- Builds Decision Tree model 

### Benek — *Feature Engineering & Classical ML*

**Builds the per-trade feature set and the first strong learned model.**

- [src/features.py](src/features.py): everything you can extract from `(price, amount, time)`. Suggested features:
  - tick direction (+ multi-lag versions),
  - log-returns over 1/3/5/10 trades,
  - inter-trade time deltas (and log thereof),
  - rolling volume statistics (zscore of `amount` over windows),
  - run-length / streak features (how many consecutive same-direction ticks?),
  - price-proximity to recent local high / low,
  - "round number" / tick-size features (where it makes sense).
- Train a gradient-boosting classifier in [src/models/gbm.py](src/models/gbm.py) (LightGBM is a good default — handles missing values, fast, robust). Hyper-parameter tune on validation only.
- Feature-importance analysis and ablations (which features actually carry signal? — drop the rest).
- [notebooks/03_features_and_gbm.ipynb](notebooks/03_features_and_gbm.ipynb).
- Owns the "Features & GBM" section of the final report.

### Krzysiek — *Sequence Model, Ensemble & API*

**Captures temporal structure and ships the deliverable.**

- [src/models/sequence.py](src/models/sequence.py): a sequence-aware model — LSTM over windows of recent trades, using Member B's features as inputs.
- [src/models/ensemble.py](src/models/ensemble.py): stack Member B's GBM and the sequence model (e.g. logistic regression on out-of-fold probabilities), calibrate, and pick the operating threshold against Member A's evaluation harness.
- [src/classify_side.py](src/classify_side.py): the **single public function**. Loads artifacts from `artifacts/`, computes features, calls the ensemble, returns the boolean `Series`. This is what the professor imports — must be bullet-proof:
  - Handle DataFrames with or without an extra `side` column.
  - Handle a single-symbol input even though we trained on two.
  - Sensible behaviour on edge cases (first trade, ties, NaNs).
- [tests/test_classify_side.py](tests/test_classify_side.py): contract tests — output is a `pd.Series`, dtype is `bool`, length matches input, index matches input, runs on a small fixture.
- [notebooks/04_sequence_and_ensemble.ipynb](notebooks/04_sequence_and_ensemble.ipynb).
- Owns the "Sequence Model, Ensemble & API" section of the final report.

## 7. Milestones (suggested)

| Week | Goal                                                                                    | Owner(s)        |
|------|------------------------------------------------------------------------------------------|-----------------|
| 1    | Data loaders, train/val/test split, baselines running end-to-end, evaluation harness     | Michał          |
| 2    | Full feature set + first GBM with validation numbers beating tick rule                   | Benek           |
| 3    | Sequence model trained, first ensemble                                                   | Krzysiek        |
| 4    | Cross-symbol generalisation check, error analysis, hyper-parameter polish                | All             |
| 5    | Final `classify_side` wrapper, tests, report, presentation                               | All             |

## 8. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell
pip install -r requirements.txt
```

(`requirements.txt` to be added in week 1: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`, `matplotlib`, `torch`)
