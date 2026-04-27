import duckdb
import pandas as pd
from pathlib import Path


def aggregate_markets_v2(markets_glob: str, trades_glob: str, category_sql: str, pattern_df: pd.DataFrame, interval_cnt: int = 100) -> pd.DataFrame:
    """read raw kalshi parquet files via duckdb and produce temporally sampled trajectories per market"""

    query = f"""
    WITH
    markets AS (
        SELECT
            ticker,
            event_ticker,
            title,
            status,
            result,
            volume,
            open_time,
            close_time,
            ({category_sql}) AS raw_prefix,
            CAST(result = 'yes' AS INTEGER) AS label
        FROM read_parquet('{markets_glob}')
        WHERE
            status = 'finalized'
            AND result IN ('yes', 'no')
            AND volume >= 100
    ),
    market_stats AS (
        SELECT
            ticker,
            open_time,
            close_time,
            (EPOCH(close_time) - EPOCH(open_time)) / {interval_cnt}.0 AS step_seconds
        FROM markets
    ),
    trade_buckets AS (
        SELECT
            t.ticker,
            FLOOR((EPOCH(t.created_time) - EPOCH(ms.open_time)) / ms.step_seconds) AS step_idx,
            t.yes_price,
            t.count,
            t.created_time
        FROM read_parquet('{trades_glob}') AS t
        INNER JOIN market_stats AS ms ON t.ticker = ms.ticker
        WHERE t.created_time >= ms.open_time AND t.created_time < ms.close_time
    ),
    interval_aggs AS (
        SELECT
            ticker,
            CAST(step_idx AS INTEGER) AS step_idx,
            -- take the last price in each interval
            ARG_MAX(yes_price, created_time) AS interval_price,
            SUM(count) AS interval_volume
        FROM trade_buckets
        WHERE step_idx >= 0 AND step_idx < {interval_cnt}
        GROUP BY ticker, step_idx
    ),
    steps AS (
        SELECT UNNEST(GENERATE_SERIES(0, {interval_cnt} - 1)) AS step_idx
    ),
    time_grid AS (
        SELECT
            ms.ticker,
            s.step_idx,
            ms.open_time + TO_SECONDS(CAST(s.step_idx * ms.step_seconds AS BIGINT)) AS grid_time
        FROM market_stats AS ms
        CROSS JOIN steps AS s
    ),
    filled_intervals AS (
        SELECT
            tg.ticker,
            tg.step_idx,
            tg.grid_time,
            ia.interval_price,
            ia.interval_volume,
            -- Forward fill price using window function
            LAST_VALUE(ia.interval_price IGNORE NULLS) OVER (PARTITION BY tg.ticker ORDER BY tg.step_idx) AS ffill_price,
            -- Cumulative volume
            SUM(COALESCE(ia.interval_volume, 0)) OVER (PARTITION BY tg.ticker ORDER BY tg.step_idx) AS cum_volume
        FROM time_grid AS tg
        LEFT JOIN interval_aggs AS ia ON tg.ticker = ia.ticker AND tg.step_idx = ia.step_idx
    ),
    assembled AS (
        SELECT
            m.ticker,
            m.title,
            m.raw_prefix,
            m.label,
            fi.step_idx,
            fi.grid_time,
            -- fill with 50 (neutral) if no trades happened yet
            COALESCE(fi.ffill_price, 50) AS yes_price,
            fi.cum_volume,
            EPOCH(m.close_time - m.open_time) / 3600.0 AS market_duration_hours,
            m.close_time
        FROM markets AS m
        INNER JOIN filled_intervals AS fi ON m.ticker = fi.ticker
    )
    SELECT 
        ticker, title, raw_prefix, label, step_idx, grid_time, yes_price, cum_volume, market_duration_hours
    FROM assembled
    ORDER BY close_time, ticker, step_idx
    """

    import os
    temp_dir = Path(__file__).resolve().parents[2] / "data" / "duckdb_temp"
    os.makedirs(temp_dir, exist_ok=True)
    con = duckdb.connect(database=":memory:", config={'temp_directory': str(temp_dir), 'preserve_insertion_order': 'false'})
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.register("category_patterns", pattern_df)
    df = con.execute(query).df()
    con.close()

    return df
