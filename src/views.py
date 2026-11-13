"""Extra series for a frozen Mantis-8M encoder.

The checkpoint embeds each channel on its own and reads a fixed length of 512.
Nothing here changes those weights. The functions build new arrays the same
encoder can embed, plus two reductions of an embedding matrix the encoder has
already produced.

Cross-channel views are computed inside each series. They do not use other
series, so a test series never sees the training set.
"""
from __future__ import annotations

import numpy as np

# Halves are embedded only when every raw series is at least this long.
# A half is then at least 512 steps, which is the length the checkpoint expects.
LONG_SERIES = 1024


def cross_channel_views(X: np.ndarray) -> np.ndarray:
    """Three univariate series per example: mean, std, and leading component.

    X has shape (n, channels, length) with channels >= 2.
    The result has shape (n, 3, length) and is float32.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (n, channels, length), got {X.shape}")
    n, channels, length = X.shape
    if channels < 2:
        raise ValueError("cross-channel views need at least 2 channels")
    values = X.astype(np.float64, copy=False)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    pcs = np.empty((n, 1, length), dtype=np.float64)
    for i in range(n):
        pcs[i, 0] = _leading_component(values[i])
    return np.concatenate([mean, std, pcs], axis=1).astype(np.float32)


def _leading_component(series: np.ndarray) -> np.ndarray:
    """Time series of the first principal component across channels.

    series is (channels, length). The sign is chosen so the score agrees with
    the channel it matches most strongly. That makes the feature deterministic.
    """
    observed = series.T  # (length, channels)
    centered = observed - observed.mean(axis=0, keepdims=True)
    try:
        left, singular, _right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(series.shape[-1], dtype=np.float64)
    score = left[:, 0] * singular[0]
    best = 0.0
    flip = False
    for channel in series:
        agreement = float(np.dot(score, channel))
        if abs(agreement) > abs(best):
            best = agreement
            flip = agreement < 0
    if flip:
        score = -score
    return score


def split_series(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one series of shape (channels, length) into temporal halves."""
    series = np.asarray(series)
    if series.ndim != 2:
        raise ValueError(f"expected (channels, length), got {series.shape}")
    length = series.shape[-1]
    if length < 2:
        return series, series.copy()
    mid = length // 2
    return series[..., :mid], series[..., mid:]


def channel_pool(Z: np.ndarray, n_channels: int) -> np.ndarray:
    """Average per-channel blocks of a concatenated embedding.

    With one channel this is the identity, so univariate accuracy must match
    the published frozen embedding.
    """
    Z = np.asarray(Z)
    n, width = Z.shape
    if n_channels < 1 or width % n_channels != 0:
        raise ValueError(f"width {width} is not divisible by {n_channels} channels")
    hidden = width // n_channels
    return Z.reshape(n, n_channels, hidden).mean(axis=1)


def shuffle_channels(Z: np.ndarray, n_channels: int, seed: int = 0) -> np.ndarray:
    """Permute channel blocks. The same permutation is used for every row.

    The forest is trained on the true channel order and scored on this copy.
    A drop means the forest was using which channel a block came from.
    """
    Z = np.asarray(Z)
    n, width = Z.shape
    if n_channels < 2 or width % n_channels != 0:
        raise ValueError("channel shuffle needs two or more equal blocks")
    hidden = width // n_channels
    order = np.random.default_rng(seed).permutation(n_channels)
    if np.array_equal(order, np.arange(n_channels)):
        order = np.roll(order, 1)
    blocks = Z.reshape(n, n_channels, hidden)
    return np.ascontiguousarray(blocks[:, order, :]).reshape(n, width)
