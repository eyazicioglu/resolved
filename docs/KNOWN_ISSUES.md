# Known Issues & Gotchas

## Resolved
- **2026-02-24 | XGBoost MacOS Runtime Error:** XGBoost failed because `libomp.dylib` was missing. Resolved natively with `brew install libomp`.

## Open
- **2026-02-24 | Target Imbalance & Baseline Overperformance:** Classical models perform strictly worse compared to the `>50` threshold baseline because class re-balancing punishes highly calibrated efficient prediction markets. (Address in Ablation Study)
