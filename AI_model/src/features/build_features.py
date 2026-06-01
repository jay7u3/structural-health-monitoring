"""Convert raw monthly pickles into a compact partitioned Parquet store.

Run once. Output: one Parquet file per month under `parquet/year=YYYY/month=MM.parquet`.
The signal columns (8 x 2000 floats) are reduced to ~128 tabular features plus
environmental and time features (~140 columns total).

Usage:
    python -m src.features.build_features --config config.yaml
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from ..data.loader import MonthBatch, list_month_files, load_month
from ..data.preprocessing import (
    env_feature_block,
    extract_signal_features,
    signal_feature_names,
)


def _extract_one(args):
    gw, exc = args
    return extract_signal_features(gw, exc)


def process_month(batch: MonthBatch, n_jobs: int = max(1, mp.cpu_count() - 1)) -> pd.DataFrame:
    feats_sig = np.empty((batch.n, 128), dtype=np.float32)
    exc = batch.excitation_signal

    # Parallel feature extraction across measurements
    args = ((batch.guided_wave[i], exc) for i in range(batch.n))
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            for i, vec in enumerate(pool.imap(_extract_one, args, chunksize=64)):
                feats_sig[i] = vec
    else:
        for i, a in enumerate(args):
            feats_sig[i] = _extract_one(a)

    env, env_names = env_feature_block(
        batch.temperature,
        batch.pressure,
        batch.brightness,
        batch.humidity,
        batch.weather_tag,
        batch.datatime,
    )

    sig_names = signal_feature_names()
    df = pd.DataFrame(feats_sig, columns=sig_names)
    df_env = pd.DataFrame(env, columns=env_names)

    df = pd.concat([df, df_env], axis=1)
    df["damage_tag"] = batch.damage_tag.astype(np.int8)
    df["weather_tag"] = batch.weather_tag.astype(np.int8)
    df["datatime"] = pd.to_datetime(batch.datatime)
    df["year"] = batch.year
    df["month"] = batch.month
    return df


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["paths"]["raw_data_dir"]).resolve()
    out_dir = Path(cfg["paths"]["parquet_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_month_files(raw_dir)
    print(f"[build_features] {len(files)} monthly files in {raw_dir}")

    for path in tqdm(files, desc="months"):
        batch = load_month(path)
        df = process_month(batch)
        sub = out_dir / f"year={batch.year}" / f"month={batch.month:02d}.parquet"
        sub.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), sub, compression="snappy")
        del batch, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
