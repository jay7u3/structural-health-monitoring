"""PyTorch datasets that stream guided-wave signals from monthly pickles
without materializing the whole 100 GB corpus.

A two-pass design:
- Pass 1 (cheap): build an index of (file_path, sample_idx, label) tuples.
- Pass 2 (training): open one file at a time, keep the current batch in RAM.

For real training, an LRU cache of recently opened files speeds things up
because shuffling within months keeps locality high.
"""
from __future__ import annotations

import pickle
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data.loader import list_month_files


META_FILE = "index.parquet"


@dataclass
class IndexRow:
    file_path: str
    sample_idx: int
    datatime: pd.Timestamp
    damage_tag: int
    weather_tag: int
    year: int
    month: int


def build_index(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Pre-scan every monthly pickle and persist a per-sample index.

    This reads only the scalar columns, not the giant `guided wave` arrays —
    but pickle does not support partial loads, so each file is opened once.
    """
    rows = []
    for path in list_month_files(raw_dir):
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = len(d["damage tag"])
        dts = pd.to_datetime(np.asarray(d["datatime"]))
        for i in range(n):
            rows.append((str(path), i, dts[i], int(d["damage tag"][i]), int(d["weather tag"][i])))
    df = pd.DataFrame(rows, columns=["file_path", "sample_idx", "datatime", "damage_tag", "weather_tag"])
    df["year"] = df["datatime"].dt.year
    df["month"] = df["datatime"].dt.month
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


class _MonthCache:
    """LRU cache of opened pickles, key = file_path."""

    def __init__(self, max_files: int = 1):
        self.max_files = max_files
        self._data: OrderedDict[str, dict] = OrderedDict()

    def get(self, path: str) -> dict:
        if path in self._data:
            self._data.move_to_end(path)
            return self._data[path]
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._data[path] = d
        while len(self._data) > self.max_files:
            self._data.popitem(last=False)
        return d


class GuidedWaveDataset(Dataset):
    """Returns (wave (8,2000), env_features (D,), label) tuples.

    Pass an env_scaler so features are normalized consistently with the GBM pipeline.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        env_scaler=None,
        n_weather_classes: int = 6,
        normalize_signal: bool = True,
        cache_files: int = 1,
    ):
        self.df = index_df.reset_index(drop=True)
        self.env_scaler = env_scaler
        self.nw = n_weather_classes
        self.normalize = normalize_signal
        self._cache = _MonthCache(max_files=cache_files)

    def __len__(self) -> int:
        return len(self.df)

    def _env_vector(self, d: dict, i: int) -> np.ndarray:
        dt = pd.Timestamp(d["datatime"][i])
        hours = dt.hour + dt.minute / 60.0
        doy = dt.timetuple().tm_yday
        env = np.array(
            [
                d["temperature"][i],
                d["pressure"][i],
                d["brightness"][i],
                d["humidity"][i],
                np.sin(2 * np.pi * hours / 24.0),
                np.cos(2 * np.pi * hours / 24.0),
                np.sin(2 * np.pi * doy / 366.0),
                np.cos(2 * np.pi * doy / 366.0),
            ],
            dtype=np.float32,
        )
        onehot = np.zeros(self.nw, dtype=np.float32)
        w = int(d["weather tag"][i])
        if 0 <= w < self.nw:
            onehot[w] = 1.0
        v = np.concatenate([env, onehot])
        if self.env_scaler is not None:
            v = self.env_scaler.transform(v.reshape(1, -1)).ravel().astype(np.float32)
        return v

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        d = self._cache.get(row["file_path"])
        i = int(row["sample_idx"])
        wave = np.asarray(d["guided wave"][i], dtype=np.float32)  # (8, 2000)
        if self.normalize:
            mu = wave.mean(axis=1, keepdims=True)
            sd = wave.std(axis=1, keepdims=True) + 1e-6
            wave = (wave - mu) / sd
        env = self._env_vector(d, i)
        y = np.float32(row["damage_tag"])
        return torch.from_numpy(wave), torch.from_numpy(env), torch.tensor(y)


def chronological_split(
    index_df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = index_df.sort_values("datatime").reset_index(drop=True)
    train = df[df["datatime"] <= pd.Timestamp(train_end)]
    val = df[(df["datatime"] > pd.Timestamp(train_end)) & (df["datatime"] <= pd.Timestamp(val_end))]
    test = df[df["datatime"] > pd.Timestamp(val_end)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
