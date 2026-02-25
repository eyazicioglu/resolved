import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_naive_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    predict YES if last_yes_price > 50, else NO.
    returns test labels, predicted probabilities, accuracy, and auc.
    """
    y_test = df["label"].values.astype(int)
    prob = df["last_yes_price"].values / 100.0
    pred = (prob > 0.50).astype(int)

    acc = float(accuracy_score(y_test, pred))
    auc = float(roc_auc_score(y_test, prob))

    return y_test, prob, acc, auc


def compute_per_category_metrics(df: pd.DataFrame, prob: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """
    evaluate metrics partitioned by explicit classification categories.
    """
    test_eval = df[["label", "last_yes_price", "category"]].copy()
    test_eval["prob"] = prob
    test_eval["pred"] = pred

    cat_metrics = (
        test_eval.groupby("category", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "accuracy": accuracy_score(g["label"], g["pred"]),
                    "auc": roc_auc_score(g["label"], g["prob"]) if g["label"].nunique() > 1 else float("nan"),
                    "yes_rate": g["label"].mean(),
                }
            ),
            include_groups=False,
        )
        .sort_values("n", ascending=False)
    )
    return cat_metrics
