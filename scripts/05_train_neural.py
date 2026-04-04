import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

matplotlib.use("Agg")

# ensure src is in path safely
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import load_data  # noqa: E402
from src.models.evaluate import evaluate, plot_calibration, plot_roc_curves  # noqa: E402
from src.models.neural import train_neural_network  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
FIGURES.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)


def main() -> None:
    print("loading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        DATA / "train.parquet", DATA / "test.parquet", MODELS
    )
    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print(f"features: {feature_cols}")
    print(f"class balance (train): NO={int((y_train == 0).sum())}  YES={int((y_train == 1).sum())}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("--- feedforward neural network ---")
    nn_model = train_neural_network(X_train, y_train)

    # override device if CUDA is available
    nn_model.device = torch.device(device)
    if nn_model.model is not None:
        nn_model.model = nn_model.model.to(nn_model.device)

    results = []
    r = evaluate("Feedforward NN", nn_model, X_test, y_test)
    results.append(r)

    # ── naive baseline ─────────────────────────────────────────────────────
    naive_prob = pd.read_parquet(DATA / "test.parquet")["last_yes_price"].values / 100.0
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
    comparison = pd.DataFrame(
        [{"model": r["model"], "accuracy": r["accuracy"], "auc": r["auc"]} for r in results]
    )
    print(comparison.to_string(index=False, float_format="{:.4f}".format))
    comparison.to_csv(DATA / "results_neural.csv", index=False)
    print(f"\nsaved {DATA / 'results_neural.csv'}")

    model_path = MODELS / "feedforward_nn.pt"
    torch.save(nn_model.model.state_dict(), model_path)
    print(f"saved {model_path}")

    plot_roc_curves(results, y_test, FIGURES / "roc_neural.png")
    plot_calibration(results, y_test, FIGURES / "calibration_neural.png")
    print("generated evaluation charts.")


if __name__ == "__main__":
    main()
