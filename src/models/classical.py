import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """logistic regression with randomized search boundaries"""
    param_dist = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500],
    }
    model = LogisticRegression(class_weight="balanced", random_state=42)
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=8,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}")
    print(f"  best cv AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """random forest with randomized search distributions"""
    param_dist = {
        "n_estimators": [100, 200, 500],
        "max_depth": [8, 16, 32, None],
        "min_samples_leaf": [1, 5, 20],
    }
    model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=12,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}")
    print(f"  best cv AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """xgboost with randomized search and class imbalance handling constraint"""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    param_dist = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
        "max_depth": [4, 6, 8, 10],
        "reg_lambda": [0.1, 1.0, 10.0],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    model = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=20,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}")
    print(f"  best cv AUC: {search.best_score_:.4f}")
    return search.best_estimator_
