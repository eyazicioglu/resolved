from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve


def evaluate(name: str, model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """evaluate a model, print report, return generated evaluation constraints"""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(classification_report(y_test, pred, target_names=["NO", "YES"], digits=4))

    return {"model": name, "accuracy": acc, "auc": auc, "prob": prob}


def plot_roc_curves(results: list[dict], y_test: np.ndarray, save_path: Path) -> None:
    """overlay ROC curves for all models constraints boundaries"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['auc']:.4f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves - classical models")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration(results: list[dict], y_test: np.ndarray, save_path: Path) -> None:
    """overlay calibration curves securely"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        CalibrationDisplay.from_predictions(
            y_test,
            r["prob"],
            n_bins=20,
            ax=ax,
            name=r["model"],
        )
    ax.set_title("calibration curves - classical models")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
