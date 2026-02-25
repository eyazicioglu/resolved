import duckdb

con = duckdb.connect()

df = con.execute("SELECT COUNT(*) as rows FROM 'aggregated_markets.parquet'").df()
print(df)

cols = con.execute("SELECT * FROM 'aggregated_markets.parquet' LIMIT 0").df()
print(f"Columns: {len(cols.columns)}")
print(list(cols.columns))