"""Frozen Mantis-8M embeddings for the inference experiments.

Published embeddings (last layer, CLS token) are read from the cache written
by scripts/02_zero_shot.py. Extra views are stored in a separate folder.
The reproduction cache is only read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common import CACHE_DIR, CHECKPOINT_REVISION


def revision_tag() -> str:
    return CHECKPOINT_REVISION or "latest"


def base_cache_path(name: str) -> Path:
    # Same filename as src.models.cached_embeddings for layer=-1, token=cls_token.
    tag = f"mantis8m_rev-{revision_tag()}_layer-1_cls_token"
    return CACHE_DIR / "emb" / tag / f"{name}.npz"


class Encoder:
    """Loads the checkpoint once, and only if a cache file is missing."""

    def __init__(self, device: str, batch_size: int = 256):
        self.device = device
        self.batch_size = batch_size
        self._net = None

    def net(self):
        if self._net is None:
            from src.models import load_mantis

            self._net = load_mantis(self.device, pretrained=True, layer=-1, token="cls_token")
        return self._net

    def base(self, name: str, data: dict):
        """Last-layer CLS embeddings, concatenated across channels.

        A finished file is the same cache `scripts/02_zero_shot.py` writes.
        A missing file is built one channel at a time, and each channel is
        saved on its own, so an interrupt resumes at the next channel.
        """
        from src.models import cached_embeddings

        path = base_cache_path(name)
        if path.exists():
            return cached_embeddings(
                name, data, self.device, layer=-1, token="cls_token",
                batch_size=self.batch_size, net=None,
            )

        X_train, X_test = data["X_train"], data["X_test"]
        n_channels = int(X_train.shape[1])
        print(
            f"{name}: embedding {n_channels} channels, {len(X_train)} train / {len(X_test)} test",
            flush=True,
        )
        train_parts, test_parts = [], []
        seconds = 0.0
        for index in range(n_channels):
            channel_path = self._channel_path(name, index)
            if channel_path.exists():
                stored = np.load(channel_path)
                train_parts.append(stored["Z_train"])
                test_parts.append(stored["Z_test"])
                seconds += float(stored["seconds"])
                print(f"{name}: channel {index + 1}/{n_channels} cached", flush=True)
                continue
            Z_train, t_train = self.embed(X_train[:, index:index + 1, :])
            Z_test, t_test = self.embed(X_test[:, index:index + 1, :])
            taken = float(t_train + t_test)
            channel_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(channel_path, Z_train=Z_train, Z_test=Z_test, seconds=np.float64(taken))
            train_parts.append(Z_train)
            test_parts.append(Z_test)
            seconds += taken
            print(f"{name}: channel {index + 1}/{n_channels}  {taken:.1f}s", flush=True)

        Z_train = np.concatenate(train_parts, axis=1).astype(np.float32)
        Z_test = np.concatenate(test_parts, axis=1).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Z_train=Z_train, Z_test=Z_test, seconds=np.float64(seconds))
        return Z_train, Z_test, seconds, False

    def _channel_path(self, name: str, index: int) -> Path:
        folder = CACHE_DIR / "emb" / f"abel-channels-rev-{revision_tag()}" / name
        return folder / f"ch{index:04d}.npz"

    def extra(self, kind: str, name: str, build):
        """Return cached (Z_train, Z_test, seconds, from_cache).

        `build` is called on a miss and must return those three values
        without the cache flag.
        """
        path = CACHE_DIR / "emb" / f"abel-{kind}-rev-{revision_tag()}" / f"{name}.npz"
        if path.exists():
            stored = np.load(path)
            return stored["Z_train"], stored["Z_test"], float(stored["seconds"]), True
        Z_train, Z_test, seconds = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Z_train=Z_train, Z_test=Z_test, seconds=np.float64(seconds))
        return Z_train, Z_test, float(seconds), False

    def embed(self, X: np.ndarray):
        from src.models import embed

        return embed(self.net(), X, self.device, batch_size=self.batch_size)
