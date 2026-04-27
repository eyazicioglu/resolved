import sys
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.classical import train_random_forest
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Hide the annoying Scikit-Learn FutureWarnings
warnings.filterwarnings('ignore')

# ── Paths ──
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "data" / "figures"
DATA_PATH = str(ROOT / "data" / "processed" / "aggregated_markets_v2.parquet").replace('\\', '/')
CUTOFF_DATE = "2025-10-01"
FIGURES.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.split import split_by_time  # noqa: E402


def load_market_prefix(step_idx: int) -> pd.DataFrame:
    """load one row per market using only steps up to the requested cutoff"""

    query = f"""
    WITH all_rows AS (
        SELECT *
        FROM read_parquet('{DATA_PATH}')
    ),
    market_meta AS (
        SELECT
            ticker,
            ANY_VALUE(title) AS title,
            ANY_VALUE(category) AS category,
            ANY_VALUE(label) AS label,
            MAX(grid_time) AS close_time,
            ANY_VALUE(market_duration_hours) AS market_duration_hours
        FROM all_rows
        GROUP BY ticker
    ),
    prefix AS (
        SELECT *
        FROM all_rows
        WHERE step_idx <= {step_idx}
    )
    SELECT
        prefix.ticker,
        meta.title,
        meta.category,
        meta.label,
        meta.close_time,
        MAX(step_idx) AS cutoff_step,
        ARG_MAX(yes_price, step_idx) AS last_yes_price,
        AVG(yes_price) AS avg_yes_price,
        COALESCE(STDDEV_POP(yes_price), 0) AS price_std,
        COALESCE(REGR_SLOPE(yes_price, step_idx), 0) AS price_slope,
        MAX(cum_volume) AS total_volume,
        COUNT(*) AS trade_count,
        meta.market_duration_hours
    FROM prefix
    INNER JOIN market_meta AS meta
        ON prefix.ticker = meta.ticker
    GROUP BY
        prefix.ticker,
        meta.title,
        meta.category,
        meta.label,
        meta.close_time,
        meta.market_duration_hours
    ORDER BY prefix.ticker
    """

    return duckdb.execute(query).df()

def tune_and_run_regression(X_train, y_train, X_test, y_test):
    """Uses GridSearchCV to find the absolute best hyperparameters based on the specified search space."""
    
    print("  Starting Search...")
    
    model = train_random_forest(X_train, y_train)
    
    # 6. Test the champion model
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    return auc, model.feature_importances_, preds

def analyze_step(step_idx):
    """loads a prefix of each market up to the requested step cutoff"""
    print(f"\n" + "="*60)
    print(f"  ANALYZING STEP {step_idx} (market prefix cutoff)")
    print("="*60)
    
    # 1. load market prefixes up to the target step
    df_step = load_market_prefix(step_idx)
    
    # 2. ONE-HOT ENCODING
    df_step = pd.get_dummies(df_step, columns=['category'], drop_first=True)
    
    # 3. Define features dynamically
    ignore_cols = ['ticker', 'title', 'label', 'close_time', 'cutoff_step']
    features = [col for col in df_step.columns if col not in ignore_cols]
    
    y = df_step['label']
    
    # 4. Train/Test Split
    df_train, df_test = split_by_time(df_step, CUTOFF_DATE)
    y_train = df_train['label']
    y_test = df_test['label']
    
    # 5. Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    X_test = scaler.transform(df_test[features])
    
    # 6. Run Tuned Model
    auc, weights, preds = tune_and_run_regression(X_train, y_train, X_test, y_test)
    pred = (preds > 0.5).astype(int)
    
    # 7. Print Results
    print(f"AUC-ROC Score: {auc:.4f}\n")
    print("Test-set precision/recall/F1:")
    print(classification_report(y_test, pred, target_names=["NO", "YES"], digits=4))
    print("Feature Weights (Ranked by Importance):")
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Weight': weights,
        'Absolute_Importance': np.abs(weights)
    }).sort_values(by='Absolute_Importance', ascending=False)
    
    print(feature_importance[['Feature', 'Weight']].to_string(index=False))

    return {
        "step": step_idx,
        "auc": auc,
        "prob": preds,
        "y_test": y_test.to_numpy(),
    }

def main():
    print(f"Targeting dataset: {DATA_PATH}")
    
    # Testing the market at various points of its lifespan
    cutoff_percentage = [50, 60, 70, 80, 90]
    milestone_results = []
    
    for step in cutoff_percentage:
        milestone_results.append(analyze_step(step))

    fig, ax = plt.subplots(figsize=(7, 6))
    for result in milestone_results:
        fpr, tpr, _ = roc_curve(result["y_test"], result["prob"])
        ax.plot(fpr, tpr, label=f"Trajectory Percentage: {result['step']}% (AUC={result['auc']:.4f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC Curves by Trajectory Percentage")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    roc_path = FIGURES / "roc_random_forest.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {roc_path}")

if __name__ == "__main__":
    main()