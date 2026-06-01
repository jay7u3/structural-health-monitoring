"""Signal → tabular features.

Given a single guided wave of shape (8, 2000), produce ~150 features
that summarize it across time-domain, frequency-domain, and wavelet domains.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import correlate
from scipy.stats import kurtosis


N_CHANNELS = 8
N_SAMPLES = 2000
FFT_BANDS = 5
WAVELET = "db4"
WAVELET_LEVELS = 4


def _time_features(x: np.ndarray) -> np.ndarray:
    """Per-channel time-domain stats. x: (8, 2000) -> (8, 5)."""
    rms = np.sqrt(np.mean(x**2, axis=1))
    energy = np.sum(x**2, axis=1)
    peak = np.max(np.abs(x), axis=1)
    crest = np.where(rms > 1e-12, peak / rms, 0.0)
    kurt = kurtosis(x, axis=1, fisher=True, bias=False)
    return np.stack([rms, energy, peak, crest, kurt], axis=1)  # (8, 5)


def _fft_band_energies(x: np.ndarray, n_bands: int = FFT_BANDS) -> np.ndarray:
    """Per-channel FFT energy in `n_bands` log-spaced bands. (8, 2000) -> (8, n_bands)."""
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2  # (8, 1001)
    n = spec.shape[1]
    edges = np.linspace(1, n, n_bands + 1, dtype=int)
    bands = np.empty((x.shape[0], n_bands), dtype=np.float32)
    for b in range(n_bands):
        bands[:, b] = np.sum(spec[:, edges[b]:edges[b + 1]], axis=1)
    return bands


def _wavelet_energies(x: np.ndarray, wavelet: str = WAVELET, levels: int = WAVELET_LEVELS) -> np.ndarray:
    """Per-channel wavelet detail-band energies. (8, 2000) -> (8, levels+1)."""
    out = np.empty((x.shape[0], levels + 1), dtype=np.float32)
    for ch in range(x.shape[0]):
        coeffs = pywt.wavedec(x[ch], wavelet=wavelet, level=levels)
        for i, c in enumerate(coeffs):
            out[ch, i] = float(np.sum(c**2))
    return out


def _first_echo_lag(x: np.ndarray, excitation: np.ndarray) -> np.ndarray:
    """Cross-correlation peak lag between excitation and each channel response.

    excitation: (1000,), x: (8, 2000) -> (8,) lag in samples.
    """
    exc = (excitation - excitation.mean()) / (excitation.std() + 1e-12)
    lags = np.empty(x.shape[0], dtype=np.float32)
    for ch in range(x.shape[0]):
        sig = x[ch]
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)
        c = correlate(sig, exc, mode="valid")
        lags[ch] = float(np.argmax(c))
    return lags


def extract_signal_features(guided_wave: np.ndarray, excitation: np.ndarray) -> np.ndarray:
    """Full feature vector for one measurement.

    guided_wave: (8, 2000), excitation: (1000,) -> 1D array of fixed length.

    Layout (per channel × 8 channels, then flattened):
      5 time stats + FFT_BANDS + (WAVELET_LEVELS + 1) wavelet + 1 echo-lag
      = 5 + 5 + 5 + 1 = 16 features/channel  -> 128 features
    """
    t = _time_features(guided_wave)                              # (8, 5)
    f = _fft_band_energies(guided_wave)                           # (8, 5)
    w = _wavelet_energies(guided_wave)                            # (8, 5)
    lag = _first_echo_lag(guided_wave, excitation)[:, None]       # (8, 1)
    per_ch = np.concatenate([t, f, w, lag], axis=1)               # (8, 16)
    return per_ch.ravel().astype(np.float32)                       # (128,)


def signal_feature_names() -> list[str]:
    names = []
    block_names = (
        ["rms", "energy", "peak", "crest", "kurtosis"]
        + [f"fft_band{i}" for i in range(FFT_BANDS)]
        + [f"wl{i}_energy" for i in range(WAVELET_LEVELS + 1)]
        + ["echo_lag"]
    )
    for ch in range(N_CHANNELS):
        for nm in block_names:
            names.append(f"ch{ch}_{nm}")
    return names


def env_feature_block(
    temperature: np.ndarray,
    pressure: np.ndarray,
    brightness: np.ndarray,
    humidity: np.ndarray,
    weather_tag: np.ndarray,
    datatime,
    n_weather_classes: int = 6,
) -> tuple[np.ndarray, list[str]]:
    """Per-batch environmental + cyclical time + one-hot weather features.

    Returns (X_env (N, D), feature_names).
    """
    n = len(temperature)
    # cyclical time
    hours = np.array([dt.hour + dt.minute / 60.0 for dt in datatime], dtype=np.float32)
    doys = np.array([dt.timetuple().tm_yday for dt in datatime], dtype=np.float32)
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    doy_sin = np.sin(2 * np.pi * doys / 366.0)
    doy_cos = np.cos(2 * np.pi * doys / 366.0)

    onehot = np.zeros((n, n_weather_classes), dtype=np.float32)
    valid = (weather_tag >= 0) & (weather_tag < n_weather_classes)
    onehot[np.arange(n)[valid], weather_tag[valid].astype(int)] = 1.0

    X = np.column_stack(
        [
            temperature, pressure, brightness, humidity,
            hour_sin, hour_cos, doy_sin, doy_cos,
            onehot,
        ]
    ).astype(np.float32)

    names = [
        "temperature", "pressure", "brightness", "humidity",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ] + [f"weather_{i}" for i in range(n_weather_classes)]
    return X, names
