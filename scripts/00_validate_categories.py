"""
step 0: validate category mapping
usage: uv run scripts/00_validate_categories.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prediction-market-analysis"))

from src.analysis.kalshi.util.categories import get_group, get_hierarchy  # noqa: E402

# ── Unit tests ────────────────────────────────────────────────────────────
TESTS = [
    ("KXMVENFLMULTIGAMEEXTENDED", "Sports"),
    ("KXBTCD20250101", "Crypto"),
    ("KXPRESCLINTON", "Politics"),
    ("KXFEDDECISION", "Finance"),
    ("KXHIGHNY20250115", "Weather"),
    ("KXNETFLIXRANKMOVIE", "Entertainment"),
    ("KXLGBTQ", "Other"),  # expected fallthrough
]

print("=== Category mapping unit tests ===")
all_passed = True
for ticker, expected in TESTS:
    got = get_group(ticker)
    status = "✓" if got == expected else "✗"
    if got != expected:
        all_passed = False
    hier = get_hierarchy(ticker)
    print(f"  {status} {ticker:<40} → {got!r:<20}  (expected {expected!r})  hierarchy={hier}")

print(f"\n{'All tests passed!' if all_passed else 'Some tests FAILED — check patterns.'}\n")

# ── Distribution in actual market data ────────────────────────────────────
import duckdb  # noqa: E402

DATA_ROOT = ROOT / "prediction-market-analysis" / "data" / "kalshi"
MARKETS_GLOB = str(DATA_ROOT / "markets" / "*.parquet")

print("=== Category distribution in finalized markets ===")
con = duckdb.connect(":memory:")
df = con.execute(f"""
    SELECT
        CASE
            WHEN event_ticker IS NULL OR event_ticker = '' THEN 'independent'
            WHEN regexp_extract(event_ticker, '^([A-Z0-9]+)', 1) = '' THEN 'independent'
            ELSE regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)
        END AS raw_prefix,
        result
    FROM read_parquet('{MARKETS_GLOB}')
    WHERE status = 'finalized' AND result IN ('yes', 'no') AND volume >= 100
""").df()
con.close()

df["category"] = df["raw_prefix"].apply(get_group)

total = len(df)
print(f"Total finalized markets (vol>=100): {total:,}")
print(f"\nYES rate overall: {(df['result'] == 'yes').mean():.1%}")
print("\nTop categories:\n")

cat_stats = (
    df.groupby("category")
    .agg(count=("result", "size"), yes_rate=("result", lambda x: (x == "yes").mean()))
    .sort_values("count", ascending=False)
)
cat_stats["pct"] = cat_stats["count"] / total * 100

for cat, row in cat_stats.iterrows():
    bar = "█" * int(row["pct"] / 2)
    print(f"  {cat:<20} {row['count']:>8,} ({row['pct']:5.1f}%)  YES={row['yes_rate']:.1%}  {bar}")
