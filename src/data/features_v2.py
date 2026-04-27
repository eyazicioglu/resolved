from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

N_STEPS = 100
SEQ_DIM = 2    # [yes_price_norm, step_volume_log]
N_STATIC = 2   # [market_duration_log, category_enc]


def load_v2_sequences(
    parquet_path: Path,
    models_path: Path,
    cutoff_date: str = "2025-10-01",
    traj_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """load v2 parquet and return train/test sequence + static arrays

    traj_frac: fraction of each market's 100-step trajectory to expose as input (applied to both splits)
    returns: X_seq_train, X_static_train, y_train, X_seq_test, X_static_test, y_test
    X_seq shape: (n_markets, traj_steps, 2)  — [yes_price_norm, step_volume_log]
    X_static shape: (n_markets, 2)           — [market_duration_log, category_enc]
    """
    cols = [
        "ticker", "label", "step_idx", "grid_time",
        "yes_price", "cum_volume", "market_duration_hours", "category",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df.sort_values(["ticker", "step_idx"], inplace=True, ignore_index=True)

    n_markets = df["ticker"].nunique()
    assert len(df) == n_markets * N_STEPS, f"expected {n_markets * N_STEPS} rows, got {len(df)}"

    # step_volume: diff of cum_volume; fix step-0 boundary across markets
    df["step_volume"] = df["cum_volume"].diff().clip(lower=0).fillna(0)
    step0_mask = df["step_idx"] == 0
    df.loc[step0_mask, "step_volume"] = df.loc[step0_mask, "cum_volume"].clip(lower=0)

    df["yes_price_norm"] = (df["yes_price"] / 99.0).astype(np.float32)
    df["step_volume_log"] = np.log1p(df["step_volume"]).astype(np.float32)

    # one row per market at step 0 (metadata) and step 99 (close_time proxy)
    meta = df.iloc[::N_STEPS][["ticker", "label", "market_duration_hours", "category"]].reset_index(drop=True)
    close_grid = df.iloc[N_STEPS - 1 :: N_STEPS]["grid_time"].reset_index(drop=True)
    meta["close_time"] = pd.to_datetime(close_grid.values, utc=True)

    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    train_mask = (meta["close_time"] < cutoff).values
    test_mask = ~train_mask
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # (n_markets, 100, 2) → truncate to first traj_frac of each trajectory
    seq_arr = df[["yes_price_norm", "step_volume_log"]].values.reshape(n_markets, N_STEPS, SEQ_DIM)
    traj_steps = max(1, int(N_STEPS * traj_frac))
    seq_arr = seq_arr[:, :traj_steps, :]

    # scale step_volume_log (col 1) using training distribution
    train_vol = seq_arr[train_idx, :, 1].ravel().reshape(-1, 1)
    vol_scaler = StandardScaler().fit(train_vol)
    seq_arr[:, :, 1] = vol_scaler.transform(
        seq_arr[:, :, 1].reshape(-1, 1)
    ).reshape(n_markets, traj_steps)

    # static features
    le = LabelEncoder().fit(meta["category"])
    meta["category_enc"] = le.transform(meta["category"]).astype(np.float32)
    meta["market_duration_log"] = np.log1p(meta["market_duration_hours"].clip(lower=0)).astype(np.float32)
    static_arr = meta[["market_duration_log", "category_enc"]].values.astype(np.float32)

    static_scaler = StandardScaler().fit(static_arr[train_idx])
    static_arr = static_scaler.transform(static_arr).astype(np.float32)

    labels = meta["label"].values.astype(int)

    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(vol_scaler, models_path / "vol_scaler_v2.pkl")
    joblib.dump(static_scaler, models_path / "static_scaler_v2.pkl")
    joblib.dump(le, models_path / "label_encoder_v2.pkl")

    return (
        seq_arr[train_idx].astype(np.float32),
        static_arr[train_idx],
        labels[train_idx],
        seq_arr[test_idx].astype(np.float32),
        static_arr[test_idx],
        labels[test_idx],
    )
