"""Streaming loader for `datas/measurements YYYY_MM.pickle`.

Each pickle is ~2 GB. We never load more than one at a time.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


FILENAME_RE = re.compile(r"measurements\s+(\d{4})_(\d{2})\.pickle$")


@dataclass
class MonthBatch:
    year: int
    month: int
    datatime: np.ndarray        # (N,) object array of datetime
    temperature: np.ndarray     # (N,)
    pressure: np.ndarray        # (N,)
    brightness: np.ndarray      # (N,)
    humidity: np.ndarray        # (N,)
    damage_tag: np.ndarray      # (N,) binary 0/1
    weather_tag: np.ndarray     # (N,) categorical
    excitation_signal: np.ndarray  # (1000,)
    guided_wave: np.ndarray     # (N, 8, 2000)

    @property
    def n(self) -> int:
        return self.guided_wave.shape[0]


def list_month_files(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = []
    for p in raw_dir.iterdir():
        m = FILENAME_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), int(m.group(2)), p))
    files.sort()
    return [p for _, _, p in files]


def load_month(path: str | Path) -> MonthBatch:
    path = Path(path)
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected filename: {path.name}")
    year, month = int(m.group(1)), int(m.group(2))

    with open(path, "rb") as f:
        d = pickle.load(f)

    return MonthBatch(
        year=year,
        month=month,
        datatime=np.asarray(d["datatime"]),
        temperature=np.asarray(d["temperature"], dtype=np.float32),
        pressure=np.asarray(d["pressure"], dtype=np.float32),
        brightness=np.asarray(d["brightness"], dtype=np.float32),
        humidity=np.asarray(d["humidity"], dtype=np.float32),
        damage_tag=np.asarray(d["damage tag"], dtype=np.float32),
        weather_tag=np.asarray(d["weather tag"], dtype=np.int8),
        excitation_signal=np.asarray(d["excitation signal"], dtype=np.float32),
        guided_wave=np.asarray(d["guided wave"], dtype=np.float32),
    )


def month_iterator(raw_dir: str | Path) -> Iterator[MonthBatch]:
    """Yield one MonthBatch at a time in chronological order.

    Memory-bounded: each batch is released as soon as the consumer moves on.
    """
    for path in list_month_files(raw_dir):
        yield load_month(path)


def iter_chunks(batch: MonthBatch, chunk: int = 256) -> Iterator[tuple[np.ndarray, ...]]:
    """Yield mini-batches of guided waves + per-measurement metadata."""
    n = batch.n
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        yield (
            batch.guided_wave[sl],
            batch.temperature[sl],
            batch.pressure[sl],
            batch.brightness[sl],
            batch.humidity[sl],
            batch.weather_tag[sl],
            batch.damage_tag[sl],
            batch.datatime[sl],
        )
