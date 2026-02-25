import sys
from pathlib import Path

import pandas as pd

# ── app paths & configuration ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "prediction-market-analysis" / "data" / "kalshi"
MARKETS_GLOB = str(DATA_ROOT / "markets" / "*.parquet")
TRADES_GLOB = str(DATA_ROOT / "trades" / "*.parquet")
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "aggregated_markets.parquet"

sys.path.insert(0, str(ROOT / "prediction-market-analysis"))
from src.analysis.kalshi.util.categories import CATEGORY_SQL, SUBCATEGORY_PATTERNS, get_group  # noqa: E402

# ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.aggregate import aggregate_markets  # noqa: E402


def main() -> None:
    """aggregate raw trades and markets into one parquet per active market"""
    print(f"reading markets: {MARKETS_GLOB}")
    print(f"reading trades:  {TRADES_GLOB}")

    # map patterns to structure and register
    pattern_df = pd.DataFrame(
        [(p, g) for p, g, *_ in SUBCATEGORY_PATTERNS],
        columns=["prefix", "group_label"],
    )

    # execute core logic
    df = aggregate_markets(MARKETS_GLOB, TRADES_GLOB, CATEGORY_SQL, pattern_df)
    print(f"raw result: {len(df):,} markets")

    # map explicitly and cleanup
    df["category"] = df["raw_prefix"].apply(get_group)
    df.drop(columns=["raw_prefix"], inplace=True)

    print(f"columns: {list(df.columns)}")
    print(f"class balance:\n{df['label'].value_counts()}")
    print(f"category breakdown:\n{df['category'].value_counts().head(15)}")

    # save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    # verify constraints
    print(f"saved: {OUTPUT}")
    print(f"rows: {len(df):,}  |  size: {OUTPUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
