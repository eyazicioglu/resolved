import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import classification_report, confusion_matrix

# ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.baseline import compute_per_category_metrics, evaluate_naive_baseline  # noqa: E402

# -- app paths -------------------------------------------------------------
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES = Path(__file__).resolve().parents[1] / "data" / "figures"
FIGURES.mkdir(exist_ok=True, parents=True)


def main() -> None:
    # load memory
    test = pd.read_parquet(OUTPUT / "test.parquet")

    # ── metrics evaluation ───────────────────────────────────────────
    y_test, prob, acc, auc = evaluate_naive_baseline(test)
    pred = (prob > 0.50).astype(int)

    print("=" * 50)
    print("  NAIVE BASELINE  (last_yes_price > 50 → YES)")
    print("=" * 50)
    print(f"  test accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"  test AUC-ROC  : {auc:.4f}")
    print()
    print(classification_report(y_test, pred, target_names=["NO", "YES"], digits=4))

    # visualize distributions safely
    cm = confusion_matrix(y_test, pred)
    print("confusion matrix (rows=actual, cols=predicted):")
    print(f"  TN={cm[0, 0]:,}  FP={cm[0, 1]:,}")
    print(f"  FN={cm[1, 0]:,}  TP={cm[1, 1]:,}")
    print()

    # category boundary constraints
    print("=== per-category accuracy ===")
    cat_metrics = compute_per_category_metrics(test, prob, pred)
    print(cat_metrics.to_string(float_format="{:.4f}".format))
    print()

    # state save boundary
    results = {
        "model": "Naive Baseline",
        "accuracy": acc,
        "auc": auc,
    }
    baseline_df = pd.DataFrame([results])
    baseline_df.to_csv(OUTPUT / "results_baseline.csv", index=False)
    print(f"✓ saved results → {OUTPUT / 'results_baseline.csv'}")

    # export reporting boundaries
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(
        y_test,
        prob,
        n_bins=20,
        ax=ax,
        name="Naive (last_yes_price)",
        color="tab:blue",
    )
    ax.set_title("calibration curve — naive baseline")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_baseline.png", dpi=150)
    plt.close(fig)
    print(f"✓ saved figure → {FIGURES / 'calibration_baseline.png'}")

    # export probability distribution
    bins = np.arange(0, 105, 5)
    test_eval = test[["label", "last_yes_price"]].copy()
    test_eval["price_bin"] = pd.cut(test["last_yes_price"], bins=bins, include_lowest=True)
    bin_stats = test_eval.groupby("price_bin", observed=True)["label"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    bin_stats.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white", width=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xlabel("last_yes_price bucket")
    ax.set_ylabel("YES rate (actual)")
    ax.set_title("actual YES rate vs. last_yes_price (test set)")
    ax.set_xticklabels([str(b) for b in bin_stats.index], rotation=45, ha="right", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "yes_rate_by_price_baseline.png", dpi=150)
    plt.close(fig)
    print(f"✓ saved figure → {FIGURES / 'yes_rate_by_price_baseline.png'}")


if __name__ == "__main__":
    main()
