import sys
from pathlib import Path

import pandas as pd

# ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.split import split_by_time  # noqa: E402
from src.data.validate import validate_aggregation  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed"
AGG = OUTPUT / "aggregated_markets.parquet"


def main() -> None:
    """runtime test and validate metrics"""
    df = pd.read_parquet(AGG)
    print(f"loaded {len(df):,} rows  ×  {df.shape[1]} columns")
    print(f"columns: {list(df.columns)}\n")

    # evaluate payload sanity constraints
    stats = validate_aggregation(df)

    # 1. nulls
    print("=== null check ===")
    for col, n in stats["nulls"].items():
        status = "✓" if n == 0 else f"✗  {n:,} nulls"
        print(f"  {col:<30} {status}")
    print()

    # 2. explicit yes boundaries
    print("=== sanity check: does probability dictate reality ===")
    print(f"  naive baseline accuracy (>50 → YES): {stats['naive_acc']:.3f}\n")

    # 3. test ratios constraint
    print("=== yes rate by category ===")
    for cat, metrics in stats["cat_rates"].items():
        bar = "█" * int(metrics["yes_rate"] * 40)
        print(f"  {cat:<20} n={int(metrics['count']):>8,}  YES={metrics['yes_rate']:.1%}  {bar}")
    print()

    # 4. feature stats
    print("=== feature summary ===")
    features = [
        "last_yes_price",
        "avg_yes_price",
        "price_std",
        "price_slope",
        "total_volume",
        "trade_count",
        "taker_yes_ratio",
        "market_duration_hours",
    ]
    print(df[features].describe().round(2).to_string())
    print()

    # 5. create discrete sets
    print("=== train / test split (close_time) ===")
    train, test = split_by_time(df, "2025-01-01")

    print(f"  train: {len(train):>8,} markets  (close < 2025-01-01)  YES={train['label'].mean():.1%}")
    print(f"  test:  {len(test):>8,} markets  (close >= 2025-01-01) YES={test['label'].mean():.1%}\n")

    # category coverage check
    train_cats = set(train["category"].unique())
    test_cats = set(test["category"].unique())
    missing = train_cats - test_cats
    if missing:
        print(f"  ⚠ categories in train but not test: {missing}")
    else:
        print(f"  ✓ all {len(train_cats)} categories represented in both sets")

    # 6. output state layer
    train_path = OUTPUT / "train.parquet"
    test_path = OUTPUT / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"\n✓ saved train → {train_path}  ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"✓ saved test  → {test_path}   ({test_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
