# AI Model — Structural Health Monitoring

Implementation of [AI_Training_Plan.md](../AI_Training_Plan.md).

## Quick start

```bash
cd AI_model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. One-shot: extract ~150 tabular features from every monthly pickle.
#    Expect ~5–10 h on 8 CPU cores. Output: parquet/year=YYYY/month=MM.parquet
python -m src.features.build_features --config config.yaml

# 2. Level A — baseline LightGBM (~5–15 min once features are built).
python -m src.train --model gbm

# 3. Level B — 1D-CNN on raw signals (GPU recommended, ~3–12 h).
python -m src.train --model cnn

# 4. Level C — convolutional autoencoder on healthy signals only (~2–6 h on GPU).
python -m src.train --model ae

# 5. Evaluation report (ROC/PR, temporal drift, weather strata, calibration).
python -m src.evaluate --model gbm

# 6. Inference on one unseen month.
python -m src.predict --input "../datas/measurements 2022_10.pickle" --output 2022_10.csv
```

## Layout

```
AI_model/
├── config.yaml                    # paths, splits, hyperparams
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── loader.py              # streaming pickle reader
│   │   ├── preprocessing.py       # signal -> ~128 features
│   │   └── splits.py              # chronological train/val/test + scaler
│   ├── features/build_features.py # pickles -> partitioned Parquet
│   ├── models/
│   │   ├── baseline_gbm.py        # Level A — LightGBM
│   │   ├── cnn1d.py               # Level B — 1D CNN + focal loss
│   │   └── autoencoder.py         # Level C — conv autoencoder
│   ├── datasets_torch.py          # streaming PyTorch Dataset
│   ├── train.py                   # entrypoint for all 3 levels
│   ├── evaluate.py                # ROC/PR + drift + calibration
│   └── predict.py                 # one-month inference -> CSV
├── parquet/                       # compact intermediate store
└── artifacts/                     # models, scalers, plots
```

## Key design choices (mirroring the plan)

- **Streaming**: monthly pickles are never all in RAM. `build_features` opens
  one file at a time and writes a Parquet shard. PyTorch loaders use an LRU
  cache of size 1 by default so peak memory stays bounded.
- **Strict temporal split**: train ends 2020-12-31, val ends 2021-06-30, test
  is everything after. No random shuffling across time.
- **Scaler fit on train only**: persisted to `artifacts/tabular_scaler.joblib`
  and reused at inference time.
- **Imbalance handling**: LightGBM gets `class_weight="balanced"`; the CNN uses
  focal loss with γ=2.
- **Evaluation beyond accuracy**: per-month F1 (drift), per-weather-tag AUC
  (robustness), reliability curve + Brier score (calibration). All saved to
  `artifacts/`.

## Notes on the data

- `damage tag` in the pickle columns is the **binary 0/1 damage label** despite
  the swapped description in `data inf`.
- Positive class is ~4.5 % — use PR-AUC, not accuracy.
- Signal columns are `(N, 8, 2000)` — 8 propagation paths (5-1..5-4, 6-1..6-4)
  excited by the shared 1 ms reference signal.
