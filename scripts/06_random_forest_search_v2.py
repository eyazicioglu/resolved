import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FEATURES = [
    "step_idx",
    "yes_price",
    "cum_volume",
    "market_duration_hours",
    "category_enc",
]

PARAM_DIST = {
    "n_estimators": [100, 200, 500],
    "max_depth": [8, 16, 32, None],
    "min_samples_leaf": [1, 5, 20],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run random forest hyperparameter search on v2 market grid parquet")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/denizkurtulan/Downloads/aggregated_markets_v2.parquet"),
    )
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/rf_search_v2"))
    return parser.parse_args()


def load_sample(path: Path, max_tickers: int, random_state: int) -> pd.DataFrame:
    con = duckdb.connect()
    if max_tickers <= 0:
        query = """
            WITH encoded AS (
                SELECT
                    label,
                    step_idx,
                    yes_price,
                    cum_volume,
                    market_duration_hours,
                    DENSE_RANK() OVER (ORDER BY category) - 1 AS category_enc,
                    hash(ticker) AS group_id
                FROM read_parquet(?)
            )
            SELECT *
            FROM encoded
        """
        return con.execute(query, [str(path)]).df()

    query = """
        WITH ticker_labels AS (
            SELECT ticker, ANY_VALUE(label) AS label
            FROM read_parquet(?)
            GROUP BY ticker
        ),
        sampled_tickers AS (
            SELECT ticker
            FROM (
                SELECT
                    ticker,
                    label,
                    ROW_NUMBER() OVER (
                        PARTITION BY label
                        ORDER BY hash(ticker || ?)
                    ) AS label_rank,
                    COUNT(*) OVER (PARTITION BY label) AS label_count
                FROM ticker_labels
            )
            WHERE label_rank <= GREATEST(1, CAST(? * label_count / (SELECT COUNT(*) FROM ticker_labels) AS INTEGER))
        )
        SELECT
            p.label,
            p.step_idx,
            p.yes_price,
            p.cum_volume,
            p.market_duration_hours,
            DENSE_RANK() OVER (ORDER BY p.category) - 1 AS category_enc,
            hash(p.ticker) AS group_id
        FROM read_parquet(?) AS p
        INNER JOIN sampled_tickers AS s USING (ticker)
        ORDER BY p.ticker, p.step_idx
    """
    return con.execute(query, [str(path), str(random_state), max_tickers, str(path)]).df()


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    groups = df["group_id"].to_numpy()
    return X, y, groups


def split_holdout(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], groups[train_idx], groups[test_idx]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    models_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading parquet: {args.input}")
    df = load_sample(args.input, args.max_tickers, args.random_state)
    print(f"sample rows: {len(df):,}")
    print(f"sample tickers: {df['group_id'].nunique():,}")
    print("label counts:")
    print(df.groupby("label")["group_id"].nunique().to_string())

    X, y, groups = prepare_data(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_holdout(X, y, groups, args.random_state)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=args.random_state, n_jobs=args.n_jobs),
        param_distributions=PARAM_DIST,
        n_iter=args.n_iter,
        cv=GroupKFold(n_splits=args.cv),
        scoring="roc_auc",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=2,
        return_train_score=True,
    )
    search.fit(X_train, y_train, groups=groups_train)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    result_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
    ]
    cv_results[result_cols].to_csv(output_dir / "random_forest_cv_results.csv", index=False)

    best_model = search.best_estimator_
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob > 0.5).astype(int)
    holdout_metrics = {
        "sample_rows": int(len(df)),
        "sample_tickers": int(df["group_id"].nunique()),
        "train_rows": int(len(y_train)),
        "train_tickers": int(pd.Series(groups_train).nunique()),
        "holdout_rows": int(len(y_test)),
        "holdout_tickers": int(pd.Series(groups_test).nunique()),
        "cv": args.cv,
        "n_iter": args.n_iter,
        "features": FEATURES,
        "param_dist": PARAM_DIST,
        "best_params": search.best_params_,
        "best_cv_auc": float(search.best_score_),
        "holdout_auc": float(roc_auc_score(y_test, test_prob)),
        "holdout_accuracy": float(accuracy_score(y_test, test_pred)),
    }

    with (output_dir / "random_forest_best_result.json").open("w") as f:
        json.dump(holdout_metrics, f, indent=2)

    joblib.dump(best_model, models_dir / "random_forest_v2_best.pkl")

    print("\nbest params:")
    print(search.best_params_)
    print(f"best cv AUC: {search.best_score_:.4f}")
    print(f"holdout AUC: {holdout_metrics['holdout_auc']:.4f}")
    print(f"holdout accuracy: {holdout_metrics['holdout_accuracy']:.4f}")
    print(f"saved cv results: {output_dir / 'random_forest_cv_results.csv'}")
    print(f"saved summary: {output_dir / 'random_forest_best_result.json'}")


if __name__ == "__main__":
    main()
