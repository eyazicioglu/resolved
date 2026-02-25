# Resolved

Machine learning pipeline to predict whether prediction markets (Kalshi/Polymarket) resolve YES or NO based on trade-level data and market metadata. Built with DuckDB, scikit-learn, and XGBoost.

## Setup

1. Install `uv` if not already installed.
2. Clone this repository and `cd` into the project directory (`ml/`).
3. Run `uv sync` to install dependencies from `uv.lock`.
4. To run scripts, use `uv run python scripts/<script_name>.py`.

## Pipeline Steps
- **Data Aggregation**: `scripts/01_aggregate.py` reads raw Parquet datasets over DuckDB.
- **Validation and Split**: `scripts/02_validate_and_split.py` creates time-based train/test sets.
- **Baseline Evaluation**: `scripts/03_baseline.py` runs a naive threshold baseline for performance comparison.
- **Classical Models**: `scripts/04_train_classical.py` trains tuned XGBoost, RandomForest, and LogisticRegression models.

## Documentation

| File | Purpose |
|:---|:---|
| `AGENTS.md` | agent instructions — read before any code changes |
| `CONVENTIONS.md` | coding rules, architecture, style, folder structure |
| `docs/SESSION_LOG.md` | chronological log of development sessions |
| `docs/KNOWN_ISSUES.md` | resolved bugs and open technical debt |
| `docs/specs/` | per-feature specification files |
