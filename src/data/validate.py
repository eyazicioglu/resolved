import pandas as pd


def validate_aggregation(df: pd.DataFrame) -> dict:
    """validate and analyze distribution statics for aggregation layer payload"""
    stats = {}

    critical = [
        "ticker",
        "label",
        "last_yes_price",
        "avg_yes_price",
        "total_volume",
        "trade_count",
        "close_time",
        "category",
    ]
    stats["nulls"] = df[critical].isnull().sum().to_dict()

    # generate discrete distribution bins for prediction visualization
    lp = df["last_yes_price"]
    bins = list(range(0, 110, 10))
    labels = [f"{b}-{b + 10}" for b in bins[:-1]]
    df["_price_bucket"] = pd.cut(lp, bins=bins, labels=labels, include_lowest=True)

    # baseline metrics based exclusively on the final raw price threshold
    naive_pred = (df["last_yes_price"] > 50).astype(int)
    stats["naive_acc"] = float((naive_pred == df["label"]).mean())

    stats["cat_rates"] = (
        df.groupby("category", observed=True)["label"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "yes_rate"})
        .sort_values("count", ascending=False)
        .to_dict(orient="index")
    )

    return stats
