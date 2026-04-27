import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features_v2 import load_v2_sequences  # noqa: E402
from src.models.lstm import LSTMClassifier, pack_X  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
FIGURES.mkdir(exist_ok=True, parents=True)

FRACS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading full sequences (once)...")
    X_seq_full_train, X_static_train, y_train, X_seq_full_test, X_static_test, y_test = (
        load_v2_sequences(DATA / "aggregated_markets_v2.parquet", MODELS, traj_frac=1.0)
    )
    N_STEPS = X_seq_full_train.shape[1]
    print(f"full shape — train: {X_seq_full_train.shape}  test: {X_seq_full_test.shape}")
    print(f"class balance — train NO={int((y_train==0).sum())} YES={int((y_train==1).sum())}\n")

    records = []

    for frac in FRACS:
        steps = max(1, int(N_STEPS * frac))
        pct = int(frac * 100)
        X_seq_train = X_seq_full_train[:, :steps, :]
        X_seq_test = X_seq_full_test[:, :steps, :]

        print(f"=== {pct}%  ({steps} steps) ===")
        clf = LSTMClassifier(
            seq_dim=X_seq_train.shape[2],
            static_dim=X_static_train.shape[1],
            hidden_size=64,
            num_layers=2,
            lr=1e-3,
            epochs=30,
            batch_size=1024,
            patience=5,
            pos_weight=1.0,
            device=device,
        )
        clf.fit(pack_X(X_seq_train, X_static_train), y_train)

        prob = clf.predict_proba(pack_X(X_seq_test, X_static_test))[:, 1]
        auc = float(roc_auc_score(y_test, prob))
        print(f"  AUC-ROC: {auc:.4f}\n")
        records.append({"frac": frac, "pct": pct, "auc": auc})

    print("=== sweep results ===")
    for r in records:
        print(f"  {r['pct']}%  AUC-ROC={r['auc']:.4f}")

    _plot(records)


def _plot(records: list[dict]) -> None:
    pcts = [r["pct"] for r in records]
    aucs = [r["auc"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcts, aucs, marker="o", linewidth=2, color="steelblue")
    for pct, auc in zip(pcts, aucs):
        ax.annotate(
            f"{auc:.4f}",
            (pct, auc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("trajectory cutoff (%)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("LSTM AUC-ROC vs trajectory cutoff\n(batch_size=1024, lr=1e-3, pos_weight=1.0)")
    ax.set_xticks(pcts)
    ax.set_ylim(min(aucs) - 0.01, max(aucs) + 0.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGURES / "traj_sweep_lstm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
