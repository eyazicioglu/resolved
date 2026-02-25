# Project Roadmap

## Current Status

- Data downloaded and extracted
- Confirmed: 7.68M markets, 72.1M trades, 7.3M resolved (76% NO, 24% YES)
- Category distribution explored — needs mapping to high-level groups

## Immediate Next Steps

### 1. Fix category extraction

The raw `event_ticker` prefixes (e.g., `KXMVENFLMULTIGAMEEXTENDED`) are too granular.
Check `research/analysis/util/categories.py` in the repo — Becker built a mapping utility.
Use `get_group()` to map to high-level groups: Sports, Crypto, Finance, Politics, Weather, Entertainment, etc.

Validate with:
```python
from research.analysis.util.categories import get_group
# Test: get_group("KXMVENFLMULTIGAMEEXTENDED") should return "Sports"
```

### 2. Build the aggregation script

Single DuckDB query: trades → market-level features. One row per resolved market.

Features to compute:
- `last_yes_price`: last trade's `yes_price` before `close_time`
- `avg_yes_price`: mean `yes_price`
- `price_std`: std dev of `yes_price`
- `price_slope`: linear trend of `yes_price` over time (can approximate with (last - first) / duration)
- `total_volume`: `SUM(count)`
- `trade_count`: `COUNT(*)`
- `taker_yes_ratio`: `SUM(count WHERE taker_side='yes') / SUM(count)`
- `market_duration`: `close_time - open_time` in hours
- `category`: from `event_ticker` via Becker's mapping
- `title`: raw text (for NLP later)
- `result`: target variable (yes=1, no=0)

Filter: `status='finalized'`, `result IN ('yes','no')`, volume >= 100.

Output: `aggregated_markets.parquet`

### 3. Validate the aggregated data

After aggregation, check:
- Row count (should be in the millions)
- No nulls in critical columns
- `last_yes_price` distribution (expect concentration near 0 and 100)
- YES/NO balance per category
- Sanity check: markets with `last_yes_price > 50` should mostly resolve YES

### 4. Train/test split

- Train: markets with `close_time < 2025-01-01`
- Test: markets with `close_time >= 2025-01-01`
- Verify both sets have reasonable size and class distribution

### 5. Baseline model

Naive baseline: predict YES if `last_yes_price > 50`, else NO.
Record accuracy and AUC-ROC. This is the floor to beat.

### 6. Classical ML models

On tabular features only (no NLP yet):
1. Logistic Regression — `sklearn.linear_model.LogisticRegression`
2. Random Forest — `sklearn.ensemble.RandomForestClassifier`
3. XGBoost — `xgboost.XGBClassifier`

Use `class_weight='balanced'` or equivalent to handle 76/24 imbalance.
Hyperparameter tuning: `RandomizedSearchCV` on training set.
Evaluate on test set: accuracy, AUC-ROC, calibration curve.

### 7. Ablation study

Using best classical model (likely XGBoost), train with:
1. `last_yes_price` only
2. + all trade features
3. + category (one-hot encoded)
4. + TF-IDF on title

Record metrics for each. Also break down per-category.

### 8. Deep learning model

Feedforward NN on tabular features:
- Keras: `Sequential([Dense(64, 'relu'), Dense(32, 'relu'), Dense(1, 'sigmoid')])`
- Adam optimizer, binary crossentropy loss
- Compare against classical models

### 9. NLP features

- `TfidfVectorizer(max_features=1000)` on `title`
- Concatenate with tabular features
- Re-run best model, measure improvement
- Optional: `sentence-transformers` embeddings

### 10. Final evaluation and figures

- ROC curves (all models on one plot)
- Calibration curves
- Feature importance (from XGBoost or RF)
- Ablation results bar chart
- Per-category accuracy/AUC table

## Key Reminders

- **Data leakage:** all features from trades before `close_time` only
- **Class imbalance:** 76% NO / 24% YES — use balanced class weights, evaluate with AUC not just accuracy
- **DuckDB for raw queries:** never load 80GB into pandas
- **Becker's utilities:** check `research/analysis/util/` for category mapping and other helpers
- **Time split, not random split:** train pre-2025, test 2025

## When to Come Back

- If the aggregation query is slow or produces unexpected results
- If class imbalance is causing model issues (all predicting NO)
- If you need help with the NN architecture or NLP pipeline
- For the final report structure
