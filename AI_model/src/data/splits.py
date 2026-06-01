"""Strict chronological train / val / test split.

Reads the Parquet store produced by `build_features`, applies a single global
StandardScaler fit on the training months only, and persists it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    test_meta: pd.DataFrame  # datatime, year, month, weather_tag for stratified eval


META_COLS = {"damage_tag", "weather_tag", "datatime", "year", "month"}


def _read_parquet_dir(parquet_dir: Path) -> pd.DataFrame:
    return pq.read_table(parquet_dir).to_pandas()


def make_splits(parquet_dir: str | Path, train_end: str, val_end: str, scaler_path: Path | None = None) -> SplitData:
    df = _read_parquet_dir(Path(parquet_dir)).sort_values("datatime").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    train_mask = df["datatime"] <= pd.Timestamp(train_end)
    val_mask = (df["datatime"] > pd.Timestamp(train_end)) & (df["datatime"] <= pd.Timestamp(val_end))
    test_mask = df["datatime"] > pd.Timestamp(val_end)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(df.loc[val_mask, feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(df.loc[test_mask, feature_cols].to_numpy(dtype=np.float32))

    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "feature_names": feature_cols}, scaler_path)

    return SplitData(
        X_train=X_train.astype(np.float32),
        y_train=df.loc[train_mask, "damage_tag"].to_numpy(dtype=np.int8),
        X_val=X_val.astype(np.float32),
        y_val=df.loc[val_mask, "damage_tag"].to_numpy(dtype=np.int8),
        X_test=X_test.astype(np.float32),
        y_test=df.loc[test_mask, "damage_tag"].to_numpy(dtype=np.int8),
        feature_names=feature_cols,
        test_meta=df.loc[test_mask, ["datatime", "year", "month", "weather_tag"]].reset_index(drop=True),
    )
