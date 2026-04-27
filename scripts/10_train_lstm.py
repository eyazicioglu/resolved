import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features_v2 import load_v2_sequences  # noqa: E402
from src.models.evaluate import plot_calibration, plot_roc_curves  # noqa: E402
from src.models.lstm import pack_X, train_lstm  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
FIGURES.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)


def main() -> None:
    print("loading v2 sequences...")
    X_seq_train, X_static_train, y_train, X_seq_test, X_static_test, y_test = load_v2_sequences(
        DATA / "aggregated_markets_v2.parquet",
        MODELS,
        traj_frac=0.9,
    )
    print(f"train: {X_seq_train.shape}  static: {X_static_train.shape}")
    print(f"test:  {X_seq_test.shape}   static: {X_static_test.shape}")
    print(f"class balance (train): NO={int((y_train == 0).sum())}  YES={int((y_train == 1).sum())}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("--- LSTM ---")

    clf, cv_results = train_lstm(X_seq_train, X_static_train, y_train, device=device)

    cv_df = pd.DataFrame(cv_results)[
        ["param_pos_weight", "mean_test_roc_auc", "std_test_roc_auc", "rank_test_roc_auc"]
    ].sort_values("rank_test_roc_auc")
    print("\nCV results:")
    print(cv_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"best: pos_weight={clf.pos_weight}\n")

    prob = clf.predict_proba(pack_X(X_seq_test, X_static_test))[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print(f"\n{'=' * 50}")
    print("  LSTM")
    print(f"{'=' * 50}")
    print(f"  accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(classification_report(y_test, pred, target_names=["NO", "YES"], digits=4))

    results = [{"model": "LSTM", "accuracy": acc, "auc": auc, "prob": prob}]

    comparison = pd.DataFrame(
        [{"model": r["model"], "accuracy": r["accuracy"], "auc": r["auc"]} for r in results]
    )
    print(comparison.to_string(index=False, float_format="{:.4f}".format))
    comparison.to_csv(DATA / "results_lstm.csv", index=False)

    model_path = MODELS / "lstm.pt"
    torch.save(clf.model_.state_dict(), model_path)
    print(f"\nsaved {model_path}")

    plot_roc_curves(results, y_test, FIGURES / "roc_lstm.png")
    plot_calibration(results, y_test, FIGURES / "calibration_lstm.png")
    print("generated evaluation charts.")


if __name__ == "__main__":
    main()
