# Session Log

Append new entries at the top. Keep each ≤5 lines.

Entry format:
```
## YYYY-MM-DD | title
- **what:** 2-3 sentences
- **files:** list of files touched
- **decisions:** key choices made
- **open:** follow-ups or unresolved items
```

---

## 2026-02-24 | Agent Template Standardization
- **what:** Applied `agent-template` structures retroactively to the codebase. Extracted logic from monolithic scripts to modular files nested across `src/data/` and `src/models/`. Updated code styles via Ruff and generated project `README.md`.
- **files:** `README.md`, `scripts/[01-04]*.py`, `src/data/*.py`, `src/models/*.py`, `docs/*`.
- **decisions:** Retained minimal shell entry points in `scripts/` calling logic imported explicitly from heavily typed `src/` modules. Suppressed `E402` to safely allow local system path hooks.
- **open:** Begin Step 7: Ablation Study leveraging the newly enforced `src/` modular codebase.

## 2026-02-24 | Train Classical Models
- **what:** Fixed XGBoost macOS missing openmp runtime. Trained LogisticRegression, RandomForest, and XGBoost with RandomizedSearchCV. Evaluated on test set. All models perform slightly under the naive baseline. 
- **files:** `scripts/04_train_classical.py`
- **decisions:** Added `class_weight=balanced` / `scale_pos_weight` to handle the target imbalance. Scaled numerics. 
- **open:** Analyze why classical ML on tabular features isn't beating the baseline (calibration curve). Begin Step 7: Ablation Study.


## 2026-02-24 | Initial Project Setup & Baseline Model
- **what:** Initialized uv project. Ran data aggregation with DuckDB. Computed train/test splits. Structured directories based on the agent-template. Created naive baseline evaluation script.
- **files:** `pyproject.toml`, `scripts/01_aggregate.py`, `scripts/02_validate_and_split.py`, `scripts/03_baseline.py`
- **decisions:** Moved output data and figures from top-level to `data/processed` and `data/figures` to abide by template conventions.
- **open:** Train classical ML models, add ablation study.
