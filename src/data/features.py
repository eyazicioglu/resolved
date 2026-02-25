from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

NUMERIC_FEATURES = [
    "last_yes_price",
    "avg_yes_price",
    "price_std",
    "price_slope",
    "total_volume",
    "trade_count",
    "taker_yes_ratio",
    "market_duration_hours",
]


def load_data(
    train_path: Path, test_path: Path, models_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """load train/test, encode category explicitly, scale numerics securely"""
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    le = LabelEncoder()
    le.fit(pd.concat([train["category"], test["category"]]))
    train["category_enc"] = le.transform(train["category"])
    test["category_enc"] = le.transform(test["category"])

    feature_cols = NUMERIC_FEATURES + ["category_enc"]

    X_train = train[feature_cols].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)
    y_train = train["label"].values.astype(int)
    y_test = test["label"].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # serialize state safely
    joblib.dump(scaler, models_path / "scaler.pkl")
    joblib.dump(le, models_path / "label_encoder.pkl")

    return X_train, y_train, X_test, y_test, feature_cols
