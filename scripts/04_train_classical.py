import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

matplotlib.use("Agg")

# ensure src is in path safely
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import load_data  # noqa: E402
from src.data.features_v2 import load_v2_sequences  # noqa: E402
from src.models.classical import train_logistic_regression, train_random_forest, train_xgboost  # noqa: E402
from src.models.evaluate import evaluate, plot_calibration, plot_roc_curves  # noqa: E402

# ── app boundaries ────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
FIGURES.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)


def main() -> None:
    """measure how trajectory truncation affects performance on v2 features"""
    parquet_path = DATA / "aggregated_markets_v2.parquet"
    if not parquet_path.exists():
        raise RuntimeError(f"missing required dataset: {parquet_path}")

    trainer_map = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest": train_random_forest,
        "XGBoost": train_xgboost,
    }

    rows = []
    for traj_frac in [i / 10 for i in range(4, 11)]:
        start = time.time()

        X_seq_train, X_static_train, y_train, X_seq_test, X_static_test, y_test = load_v2_sequences(
            parquet_path=parquet_path,
            models_path=MODELS,
            traj_frac=traj_frac,
        )

        X_train = np.concatenate([X_seq_train.reshape(X_seq_train.shape[0], -1), X_static_train], axis=1)
        X_test = np.concatenate([X_seq_test.reshape(X_seq_test.shape[0], -1), X_static_test], axis=1)

        for model_name, trainer in trainer_map.items():
            model = trainer(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob > 0.5).astype(int)
            acc = float(accuracy_score(y_test, pred))
            auc = float(roc_auc_score(y_test, prob))

            rows.append(
                {
                    "model": model_name,
                    "traj_frac": traj_frac,
                    "traj_steps": int(X_seq_train.shape[1]),
                    "accuracy": acc,
                    "auc": auc,
                    "elapsed_s": round(time.time() - start, 2),
                }
            )
            print(
                f"model={model_name} traj_frac={traj_frac:.1f} "
                f"steps={int(X_seq_train.shape[1])} accuracy={acc:.4f} auc={auc:.4f}"
            )

    result_df = pd.DataFrame(rows).sort_values(["model", "traj_frac"])
    out_path = DATA / "results_knowledge_cutoff_classical.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\nsaved {out_path}")
    print(result_df.to_string(index=False, float_format="{:.4f}".format))



def _run_standard_classical_training() -> None:
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
    naive_prob = pd.read_parquet(test_path)["last_yes_price"].to_numpy(dtype=float) / 100.0
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
