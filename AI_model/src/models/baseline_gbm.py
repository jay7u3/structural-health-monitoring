"""LightGBM baseline on tabular features."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np


def train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_path: Path,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_child_samples: int = 50,
    class_weight: str | dict = "balanced",
    early_stopping_rounds: int = 50,
    feature_names: list[str] | None = None,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        class_weight=class_weight,
        objective="binary",
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "average_precision"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)],
        feature_name=feature_names if feature_names else "auto",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return model
