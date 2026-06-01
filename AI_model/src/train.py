"""Single entry point for training each model level.

    python -m src.train --model gbm
    python -m src.train --model cnn
    python -m src.train --model ae
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.splits import make_splits
from .datasets_torch import GuidedWaveDataset, build_index, chronological_split, META_FILE
from .models.autoencoder import ConvAE, reconstruction_anomaly_score
from .models.baseline_gbm import train_gbm
from .models.cnn1d import CNN1D, FocalLoss


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_baseline(cfg: dict) -> None:
    parquet_dir = Path(cfg["paths"]["parquet_dir"])
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    scaler_path = artifacts / "tabular_scaler.joblib"
    sp = make_splits(parquet_dir, cfg["splits"]["train_end"], cfg["splits"]["val_end"], scaler_path)
    print(f"[gbm] train={len(sp.y_train)} val={len(sp.y_val)} test={len(sp.y_test)} "
          f"pos_train={sp.y_train.mean():.3%}")

    p = cfg["baseline_gbm"]
    model = train_gbm(
        sp.X_train, sp.y_train, sp.X_val, sp.y_val,
        out_path=artifacts / "gbm.joblib",
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        min_child_samples=p["min_child_samples"],
        class_weight=p["class_weight"],
        early_stopping_rounds=p["early_stopping_rounds"],
        feature_names=sp.feature_names,
    )
    joblib.dump(
        {"X_test": sp.X_test, "y_test": sp.y_test, "test_meta": sp.test_meta,
         "feature_names": sp.feature_names},
        artifacts / "test_set.joblib",
    )
    print("[gbm] saved to", artifacts / "gbm.joblib")


def _make_loaders(cfg: dict, mode: str):
    raw_dir = cfg["paths"]["raw_data_dir"]
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    idx_path = artifacts / META_FILE
    if idx_path.exists():
        idx_df = pd.read_parquet(idx_path)
    else:
        print("[index] building per-sample index...")
        idx_df = build_index(raw_dir, idx_path)

    train_df, val_df, test_df = chronological_split(
        idx_df, cfg["splits"]["train_end"], cfg["splits"]["val_end"]
    )
    if mode == "ae":
        train_df = train_df[train_df["damage_tag"] == 0].reset_index(drop=True)

    bs = cfg["cnn1d"]["batch_size"] if mode == "cnn" else cfg["autoencoder"]["batch_size"]
    train_ds = GuidedWaveDataset(train_df)
    val_ds = GuidedWaveDataset(val_df)
    test_ds = GuidedWaveDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def train_cnn(cfg: dict) -> None:
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    train_loader, val_loader, _, _, val_df, _ = _make_loaders(cfg, "cnn")
    p = cfg["cnn1d"]
    device = _device()
    sample_env = next(iter(train_loader))[1]
    env_dim = sample_env.shape[1]
    model = CNN1D(in_channels=8, env_dim=env_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    crit = FocalLoss(gamma=p["focal_gamma"])
    best_val = float("inf")
    patience_left = p["patience"]

    for epoch in range(p["epochs"]):
        model.train()
        tr_loss = 0.0
        for wave, env, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{p['epochs']}"):
            wave, env, y = wave.to(device), env.to(device), y.to(device)
            opt.zero_grad()
            logits = model(wave, env)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * wave.size(0)
        tr_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for wave, env, y in val_loader:
                wave, env, y = wave.to(device), env.to(device), y.to(device)
                logits = model(wave, env)
                val_loss += crit(logits, y).item() * wave.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"epoch {epoch+1}: train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience_left = p["patience"]
            torch.save({"state_dict": model.state_dict(), "env_dim": env_dim},
                       artifacts / "cnn1d.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[cnn] early stop")
                break
    print("[cnn] best val loss:", best_val)


def train_autoencoder(cfg: dict) -> None:
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    train_loader, val_loader, _, _, _, _ = _make_loaders(cfg, "ae")
    p = cfg["autoencoder"]
    device = _device()
    model = ConvAE(in_channels=8, latent_dim=p["latent_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"])
    crit = torch.nn.MSELoss()
    best_val = float("inf")
    patience_left = p["patience"]

    for epoch in range(p["epochs"]):
        model.train()
        tr = 0.0
        for wave, _, _ in tqdm(train_loader, desc=f"ae epoch {epoch+1}"):
            wave = wave.to(device)
            opt.zero_grad()
            out = model(wave)
            loss = crit(out, wave)
            loss.backward()
            opt.step()
            tr += loss.item() * wave.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        v = 0.0
        with torch.no_grad():
            for wave, _, _ in val_loader:
                wave = wave.to(device)
                v += crit(model(wave), wave).item() * wave.size(0)
        v /= len(val_loader.dataset)
        print(f"ae epoch {epoch+1}: train={tr:.6f} val={v:.6f}")

        if v < best_val - 1e-6:
            best_val = v
            patience_left = p["patience"]
            torch.save({"state_dict": model.state_dict(), "latent_dim": p["latent_dim"]},
                       artifacts / "autoencoder.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[ae] early stop")
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gbm", "cnn", "ae"], required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    np_seed = cfg.get("seed", 42)
    np.random.seed(np_seed)
    torch.manual_seed(np_seed)

    if args.model == "gbm":
        train_baseline(cfg)
    elif args.model == "cnn":
        train_cnn(cfg)
    else:
        train_autoencoder(cfg)


if __name__ == "__main__":
    main()
