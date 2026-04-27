import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features_v2 import load_v2_sequences  # noqa: E402
from src.models.lstm import LSTMClassifier, pack_X  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
FIGURES.mkdir(exist_ok=True, parents=True)

POS_WEIGHTS = [0.5, 1.0, 1.5, 2.0]
COLORS = ["steelblue", "darkorange", "green", "crimson"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading sequences...")
    X_seq_train, X_static_train, y_train, X_seq_test, X_static_test, y_test = load_v2_sequences(
        DATA / "aggregated_markets_v2.parquet", MODELS, traj_frac=0.9
    )
    print(f"train: {X_seq_train.shape}  test: {X_seq_test.shape}\n")

    X_train = pack_X(X_seq_train, X_static_train)
    X_test = pack_X(X_seq_test, X_static_test)

    results = []

    for pw in POS_WEIGHTS:
        print(f"=== pos_weight={pw} ===")
        clf = LSTMClassifier(
            seq_dim=X_seq_train.shape[2],
            static_dim=X_static_train.shape[1],
            hidden_size=64,
            num_layers=2,
            lr=1e-3,
            epochs=30,
            batch_size=1024,
            patience=5,
            pos_weight=pw,
            device=device,
        )
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, prob))
        fpr, tpr, _ = roc_curve(y_test, prob)
        print(f"  AUC-ROC: {auc:.4f}\n")
        results.append({"pw": pw, "auc": auc, "fpr": fpr, "tpr": tpr})

    for r, color in zip(results, COLORS):
        _plot_single(r, color)


def _plot_single(r: dict, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r["fpr"], r["tpr"], linewidth=1.5, color=color,
            label=f"pos_weight={r['pw']}  AUC={r['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve — pos_weight={r['pw']}\n(batch_size=1024, lr=1e-3, traj_frac=0.9)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    pw_str = str(r["pw"]).replace(".", "_")
    out = FIGURES / f"roc_pos_weight_{pw_str}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
