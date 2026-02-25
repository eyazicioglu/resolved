# Predicting Market Resolution in Prediction Markets

## Task

Binary classification: predict whether a prediction market resolves **YES** or **NO**.

## Dataset

- **Source:** `github.com/Jon-Becker/prediction-market-analysis`
- **Location:** Already extracted at `data/`
- **Structure:**
  ```
  data/
    kalshi/
      markets/    # Parquet files
      trades/     # Parquet files, per-ticker
    polymarket/
      markets/    # Parquet files
      trades/     # Parquet files
  ```

### Kalshi Markets Schema (primary)

| Column | Type | Notes |
|---|---|---|
| `ticker` | string | Unique market ID |
| `event_ticker` | string | Parent event, used for category extraction |
| `title` | string | Human-readable market question |
| `status` | string | `open`, `closed`, `finalized` |
| `result` | string | `yes`, `no`, or empty |
| `volume` | int | Total contracts traded |
| `last_price` | int (nullable) | Last traded price in cents (1-99) |
| `created_time` | datetime | |
| `open_time` | datetime | |
| `close_time` | datetime | |

### Kalshi Trades Schema

| Column | Type | Notes |
|---|---|---|
| `trade_id` | string | Unique |
| `ticker` | string | FK to markets |
| `count` | int | Contracts traded |
| `yes_price` | int | Price in cents (1-99) |
| `no_price` | int | Always `100 - yes_price` |
| `taker_side` | string | `yes` or `no` |
| `created_time` | datetime | |

### Filters

- Only `status = 'finalized'` and `result IN ('yes', 'no')`
- Exclude markets with < $100 total volume

## Pipeline

### Step 1: Aggregation

Use DuckDB to query Parquet files directly. Aggregate trade-level data into **one row per market**.

**Target:** `result` field (`yes` = 1, `no` = 0)

**Features to compute per market:**

| Feature | Source | Computation |
|---|---|---|
| `last_yes_price` | trades | Last `yes_price` before `close_time` |
| `avg_yes_price` | trades | Mean `yes_price` |
| `price_std` | trades | Std dev of `yes_price` |
| `price_slope` | trades | Linear regression slope of `yes_price` over time |
| `total_volume` | trades | `SUM(count)` |
| `trade_count` | trades | `COUNT(*)` |
| `taker_yes_ratio` | trades | `SUM(count WHERE taker_side='yes') / SUM(count)` |
| `market_duration` | markets | `close_time - open_time` |
| `category` | markets | Extract from `event_ticker`: `regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)` |
| `title` | markets | Raw text, for NLP features |

**Output:** Single Parquet/CSV file, <1GB expected.

**Critical: data leakage prevention.** All trade features must be computed from trades occurring **before** market close. Do not use post-resolution data.

### Step 2: Train/Test Split

- **Time-based split.** Train on markets resolved before 2025. Test on markets resolved in 2025.
- Stratify by `result` if class imbalance is significant.

### Step 3: Models

| Model | Library | Notes |
|---|---|---|
| Logistic Regression | `sklearn` | Baseline. Use `predict_proba`. |
| Random Forest | `sklearn` | Tune `n_estimators`, `max_depth` |
| XGBoost | `xgboost` | Tune `learning_rate`, `n_estimators`, `max_depth`, `reg_lambda` |
| Feedforward NN | `keras` or `torch` | 2-3 hidden layers, ReLU, sigmoid output. Adam optimizer. |

Hyperparameter tuning: grid search or random search with cross-validation on training set only.

### Step 4: NLP Features

- **TF-IDF** on `title` field using `sklearn.feature_extraction.text.TfidfVectorizer`. Concatenate with tabular features.
- **Optional extension:** `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to produce dense embeddings from titles.

### Step 5: Ablation Study

Train the best-performing model (likely XGBoost) under these feature configurations:

1. `last_yes_price` only
2. \+ all trade features (`avg_yes_price`, `price_std`, `price_slope`, `total_volume`, `trade_count`, `taker_yes_ratio`)
3. \+ `category`
4. \+ NLP features (TF-IDF on `title`)

Report metrics for each. Also report **per-category** performance breakdown.

### Step 6: Evaluation

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| AUC-ROC | Discrimination ability |
| Calibration curve | Are predicted probabilities reliable? |
| Per-category accuracy/AUC | Where do additional features help most? |

Compare all models against a **naive baseline**: predict `yes` if `last_yes_price > 50`, else `no`.

## Expected Outputs

- `aggregated_markets.parquet` — one row per market with all features
- Trained model files (`.pkl` for sklearn, `.json` for XGBoost, `.h5`/`.pt` for NN)
- Ablation results table (CSV)
- Per-category results table (CSV)
- Figures: ROC curves, calibration curves, feature importance plots, ablation bar chart

## Polymarket (optional extension)

The dataset also includes Polymarket data under `data/polymarket/`. Schema may differ from Kalshi — verify before use. If schemas align:

- Repeat the pipeline on Polymarket data
- Cross-platform experiment: train on Kalshi, test on Polymarket (and vice versa)
- Report transferability results

## Chrome Extension (future work)

End goal: browser extension overlaying model predictions on live Kalshi/Polymarket pages.

- Export trained XGBoost model as JSON (tree structure is trivially evaluable in JS)
- Compute features in real-time via Kalshi REST API (`trading-api.readme.io`) and Polymarket CLOB API (`docs.polymarket.com`)
- Ablation results determine which features to compute at inference time
- Display: model probability vs. market price, delta, contributing features

## Constraints

- **80GB extracted data.** Use DuckDB for all queries on raw Parquet. Do not load into pandas directly.
- **No data leakage.** Time-based split. No future information in features.
- **Course requirement:** 3 classical ML models + at least 1 deep learning model. Ablation/experimentation depth matters for grading.

## References

- Dataset: `github.com/Jon-Becker/prediction-market-analysis`
- Becker (2026). "The Microstructure of Wealth Transfer in Prediction Markets." `jbecker.dev/research/prediction-market-microstructure`
- Whelan (2025). "Makers and Takers: The Economics of the Kalshi Prediction Market." `karlwhelan.com/Papers/Kalshi.pdf`
