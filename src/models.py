"""Mantis-8M loading and embedding helpers (package: mantis-tsfm 1.1.0)."""
import time

import numpy as np
import torch
import torch.nn as nn

from src.common import CACHE_DIR, CHECKPOINT, CHECKPOINT_REVISION, sync


def load_mantis(device: str, pretrained: bool = True, layer: int = -1,
                token: str = "cls_token", seed: int | None = None):
    """v1 protocol defaults: last transformer layer (-1) and the CLS token.

    pretrained=False returns a randomly initialised network, which is how we train
    from scratch. The package's fine_tuning_type="scratch" does not re-initialise
    the weights; it trains every parameter of the pretrained model, the same as "full".
    """
    from mantis.architecture import MantisV1
    if seed is not None:
        torch.manual_seed(seed)
    net = MantisV1(device=device, return_transf_layer=layer, output_token=token)
    if pretrained:
        net = net.from_pretrained(CHECKPOINT, revision=CHECKPOINT_REVISION)
    return net.to(device)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def embed(net, X: np.ndarray, device: str, batch_size: int = 256) -> tuple[np.ndarray, float]:
    """Frozen-encoder embeddings via the package's own MantisTrainer.transform.
    Multichannel data: each channel is embedded independently and concatenated."""
    from mantis.trainer import MantisTrainer
    trainer = MantisTrainer(device=device, network=net)
    sync(device)
    t0 = time.perf_counter()
    Z = trainer.transform(X, batch_size=batch_size)
    sync(device)
    return Z.astype(np.float32), time.perf_counter() - t0


def cached_embeddings(name: str, data: dict, device: str, layer: int = -1,
                      token: str = "cls_token", batch_size: int = 256, net=None):
    """Embeddings for train and test, cached per (checkpoint, layer, token)."""
    tag = f"mantis8m_rev-{CHECKPOINT_REVISION or 'latest'}_layer{layer}_{token}"
    path = CACHE_DIR / "emb" / tag / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["Z_train"], z["Z_test"], float(z["seconds"]), True
    if net is None:
        net = load_mantis(device, layer=layer, token=token)
    Ztr, t1 = embed(net, data["X_train"], device, batch_size)
    Zte, t2 = embed(net, data["X_test"], device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Z_train=Ztr, Z_test=Zte, seconds=t1 + t2)
    return Ztr, Zte, t1 + t2, False


def paper_head(in_dim: int, n_classes: int) -> nn.Module:
    """v1 fine-tuning head: LayerNorm + Linear.
    mantis-tsfm 1.1.0 defaults to BatchNorm1d + Linear, which differs from the paper."""
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_classes))
