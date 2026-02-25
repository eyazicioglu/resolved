import duckdb
import pandas as pd


def aggregate_markets(markets_glob: str, trades_glob: str, category_sql: str, pattern_df: pd.DataFrame) -> pd.DataFrame:
    """read raw kalshi parquet files via duckdb and produce one row per finalized market"""

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
    pre_close_trades AS (
        SELECT
            t.ticker,
            t.yes_price,
            t.count,
            t.taker_side,
            t.created_time
        FROM read_parquet('{trades_glob}') AS t
        INNER JOIN markets AS m
            ON t.ticker = m.ticker
        WHERE t.created_time < m.close_time
    ),
    first_last AS (
        SELECT
            ticker,
            FIRST_VALUE(yes_price) OVER (
                PARTITION BY ticker ORDER BY created_time ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS first_yes_price,
            LAST_VALUE(yes_price) OVER (
                PARTITION BY ticker ORDER BY created_time ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS last_yes_price_raw
        FROM pre_close_trades
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_time ASC) = 1
    ),
    trade_aggs AS (
        SELECT
            ticker,
            AVG(yes_price)                                      AS avg_yes_price,
            STDDEV_POP(yes_price)                               AS price_std,
            SUM(count)                                          AS total_volume,
            COUNT(*)                                            AS trade_count,
            SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END)
                / NULLIF(SUM(count), 0)                         AS taker_yes_ratio
        FROM pre_close_trades
        GROUP BY ticker
    ),
    assembled AS (
        SELECT
            m.ticker,
            m.title,
            m.raw_prefix,
            m.label,
            m.close_time,
            m.open_time,
            fl.last_yes_price_raw                               AS last_yes_price,
            ta.avg_yes_price,
            COALESCE(ta.price_std, 0)                           AS price_std,
            CASE
                WHEN EPOCH(m.close_time - m.open_time) / 3600.0 < 1 THEN 0
                ELSE (fl.last_yes_price_raw - fl.first_yes_price)
                     / (EPOCH(m.close_time - m.open_time) / 3600.0)
            END                                                 AS price_slope,
            COALESCE(ta.total_volume, 0)                        AS total_volume,
            COALESCE(ta.trade_count, 0)                         AS trade_count,
            COALESCE(ta.taker_yes_ratio, 0.5)                   AS taker_yes_ratio,
            EPOCH(m.close_time - m.open_time) / 3600.0          AS market_duration_hours
        FROM markets AS m
        LEFT JOIN trade_aggs  AS ta ON m.ticker = ta.ticker
        LEFT JOIN first_last  AS fl ON m.ticker = fl.ticker
        WHERE fl.last_yes_price_raw IS NOT NULL
    )
    SELECT * FROM assembled
    ORDER BY close_time
    """

    con = duckdb.connect(database=":memory:")
    con.register("category_patterns", pattern_df)
    df = con.execute(query).df()
    con.close()

    return df
