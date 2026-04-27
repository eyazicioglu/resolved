import sys
from pathlib import Path

import pandas as pd

# ── app paths & configuration ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "prediction-market-analysis" / "data" / "kalshi"
MARKETS_GLOB = str(DATA_ROOT / "markets" / "*.parquet")
TRADES_GLOB = str(DATA_ROOT / "trades" / "*.parquet")

# Save as _v2
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "aggregated_markets_v2.parquet"

sys.path.insert(0, str(ROOT / "prediction-market-analysis"))
from src.analysis.kalshi.util.categories import CATEGORY_SQL, SUBCATEGORY_PATTERNS, get_group

import importlib.util
aggregate_v2_path = Path(__file__).resolve().parents[1] / "src" / "data" / "aggregate_v2.py"
spec = importlib.util.spec_from_file_location("aggregate_v2", str(aggregate_v2_path))
aggregate_v2_mod = importlib.util.module_from_spec(spec)
sys.modules["aggregate_v2"] = aggregate_v2_mod
spec.loader.exec_module(aggregate_v2_mod)
aggregate_markets_v2 = aggregate_v2_mod.aggregate_markets_v2


def main() -> None:
    """aggregate raw trades and markets into a temporal trajectory per market"""
    print(f"reading markets: {MARKETS_GLOB}")
    print(f"reading trades:  {TRADES_GLOB}")

    # map patterns to structure and register
    pattern_df = pd.DataFrame(
        [(p, g) for p, g, *_ in SUBCATEGORY_PATTERNS],
        columns=["prefix", "group_label"],
    )

    INTERVAL_CNT = 100

    # execute core logic
    df = aggregate_markets_v2(MARKETS_GLOB, TRADES_GLOB, CATEGORY_SQL, pattern_df, interval_cnt=INTERVAL_CNT)
    
    unique_markets = df["ticker"].nunique()
    print(f"raw result: {unique_markets:,} markets ({len(df):,} total rows)")

    # map explicitly and cleanup
    df["category"] = df["raw_prefix"].apply(get_group)
    df.drop(columns=["raw_prefix"], inplace=True)

    print(f"columns: {list(df.columns)}")
    print(f"class balance:\n{df.drop_duplicates(subset=['ticker'])['label'].value_counts()}")
    
    # save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    # verify constraints
    print(f"saved: {OUTPUT}")
    print(f"rows: {len(df):,} ({INTERVAL_CNT} per market)  |  size: {OUTPUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
