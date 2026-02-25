import pandas as pd


def split_by_time(df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split dataset based on an explicitly provided cutoff timestamp using proper type hints"""
    # ensure proper datetime
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")

    # execute query-based splits
    train = df[df["close_time"] < cutoff].copy()
    test = df[df["close_time"] >= cutoff].copy()

    # discard extraneous columns safely
    if "_price_bucket" in train.columns:
        train.drop(columns=["_price_bucket"], inplace=True)
    if "_price_bucket" in test.columns:
        test.drop(columns=["_price_bucket"], inplace=True)

    return train, test
