import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

matplotlib.use("Agg")

# ensure src is in path safely
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import load_data  # noqa: E402
from src.models.classical import train_logistic_regression, train_random_forest, train_xgboost  # noqa: E402
from src.models.evaluate import evaluate, plot_calibration, plot_roc_curves  # noqa: E402

# ── app boundaries ────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
FIGURES.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)


def main() -> None:
    print("loading data constraints...")
    train_path = DATA / "train.parquet"
    test_path = DATA / "test.parquet"
    X_train, y_train, X_test, y_test, feature_cols = load_data(train_path, test_path, MODELS)

    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print(f"features: {feature_cols}")
    print(f"class balance (train): NO={int((y_train == 0).sum())}  YES={int((y_train == 1).sum())}\n")

    models = {}

    print("--- logistic regression ---")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)

    print("\n--- random forest ---")
    models["Random Forest"] = train_random_forest(X_train, y_train)

    print("\n--- xgboost ---")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    results = []
    for name, model in models.items():
        r = evaluate(name, model, X_test, y_test)
        results.append(r)

    # ── native baseline integration ────────────────────────────────────
    naive_prob = pd.read_parquet(test_path)["last_yes_price"].values / 100.0
    naive_acc = float(accuracy_score(y_test, (naive_prob > 0.5).astype(int)))
    naive_auc = float(roc_auc_score(y_test, naive_prob))
    results.insert(
        0,
        {
            "model": "Naive Baseline",
            "accuracy": naive_acc,
            "auc": naive_auc,
            "prob": naive_prob,
        },
    )

    print(f"\n{'=' * 50}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 50}")
    comparison = pd.DataFrame([{"model": r["model"], "accuracy": r["accuracy"], "auc": r["auc"]} for r in results])
    print(comparison.to_string(index=False, float_format="{:.4f}".format))
    comparison.to_csv(DATA / "results_classical.csv", index=False)
    print(f"\nsaved {DATA / 'results_classical.csv'}")

    for name, model in models.items():
        slug = name.lower().replace(" ", "_")
        path = MODELS / f"{slug}.pkl"
        joblib.dump(model, path)
        print(f"saved {path}")

    plot_roc_curves(results, y_test, FIGURES / "roc_classical.png")
    plot_calibration(results, y_test, FIGURES / "calibration_classical.png")
    print("generated evaluation charts.")


if __name__ == "__main__":
    main()
