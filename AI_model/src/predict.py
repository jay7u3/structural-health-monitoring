"""Inference on a single monthly pickle.

    python -m src.predict --input "../datas/measurements 2022_10.pickle" --output 2022_10.csv

Produces a CSV with columns: datatime, damage_proba, health_score.
`health_score` = 100 * (1 - smoothed damage probability), capped to [0, 100].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data.loader import load_month
from .features.build_features import process_month


def predict_month(input_path: str, cfg: dict) -> pd.DataFrame:
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    model = joblib.load(artifacts / "gbm.joblib")
    scaler_bundle = joblib.load(artifacts / "tabular_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    feature_names = scaler_bundle["feature_names"]

    batch = load_month(input_path)
    df = process_month(batch)

    X = scaler.transform(df[feature_names].to_numpy(dtype=np.float32))
    p = model.predict_proba(X)[:, 1]

    # Smooth health score with a simple rolling mean (1h window ≈ 21 measures)
    smoothed = pd.Series(p).rolling(window=21, min_periods=1, center=True).mean().to_numpy()
    health = (100.0 * (1.0 - smoothed)).clip(0.0, 100.0)

    out = pd.DataFrame({
        "datatime": df["datatime"],
        "damage_proba": p,
        "health_score": health,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to one monthly .pickle")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = predict_month(args.input, cfg)
    out.to_csv(args.output, index=False)
    print(f"[predict] wrote {len(out)} rows -> {args.output}")
    print(out.describe())


if __name__ == "__main__":
    main()
