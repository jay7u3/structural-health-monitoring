"""Evaluation: ROC/PR, temporal drift, weather stratification, calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _metrics(y, p, threshold=0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
        "confusion": confusion_matrix(y, yhat).tolist(),
        "pos_rate": float(y.mean()),
    }


def evaluate_gbm(cfg: dict) -> dict:
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    model = joblib.load(artifacts / "gbm.joblib")
    bundle = joblib.load(artifacts / "test_set.joblib")
    X, y, meta = bundle["X_test"], bundle["y_test"], bundle["test_meta"]
    p = model.predict_proba(X)[:, 1]

    out = {"overall": _metrics(y, p)}

    # Temporal drift: F1 per month
    df = meta.copy()
    df["p"] = p
    df["y"] = y
    drift = (
        df.assign(ym=df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2))
        .groupby("ym")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "pos_rate": g["y"].mean(),
            "roc_auc": roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan,
            "f1@0.5": f1_score(g["y"], (g["p"] >= 0.5).astype(int)) if g["y"].nunique() == 2 else np.nan,
        }))
        .reset_index()
    )
    drift.to_csv(artifacts / "drift_per_month.csv", index=False)

    # Weather stratification
    weather = (
        df.groupby("weather_tag")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "pos_rate": g["y"].mean(),
            "roc_auc": roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan,
        }))
        .reset_index()
    )
    weather.to_csv(artifacts / "weather_strata.csv", index=False)

    # ROC and PR plots
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC – AUC={out['overall']['roc_auc']:.3f}"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.savefig(artifacts / "roc_test.png", dpi=120, bbox_inches="tight"); plt.close()

    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(); plt.plot(rec, prec)
    plt.title(f"PR – AP={out['overall']['pr_auc']:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(artifacts / "pr_test.png", dpi=120, bbox_inches="tight"); plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0, 1], [0, 1], "k--")
    plt.title("Reliability"); plt.xlabel("Mean predicted prob"); plt.ylabel("Empirical frequency")
    plt.savefig(artifacts / "calibration.png", dpi=120, bbox_inches="tight"); plt.close()

    print("[eval gbm] overall:", out["overall"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=["gbm"], default="gbm",
                    help="cnn/ae evaluation share the same metric helpers; "
                         "wire them in train.py output files first.")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate_gbm(cfg)


if __name__ == "__main__":
    main()
