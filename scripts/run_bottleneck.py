"""Where frozen Mantis-8M spends its forward pass.

Three measurements, all on synthetic batches, none of them a change to the
weights:

1. Parameter count of each repeated transformer block, so later blocks can be
   sized even though this run does not delete them.
2. Wall time when the forward pass is asked for the last layer and for the
   first layer. A similar time means the implementation still executes the
   later blocks.
3. Wall time against the number of series, the number of channels, and the
   raw length that has to be resized onto 512.

  python scripts/run_bottleneck.py
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inference_paths", Path(__file__).resolve().parents[1] / "src" / "paths.py"
)
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)
PROJECT, _ROOT = _paths.prepare()

from src.common import pick_device, require_env, save_json, set_seed, sync  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

OUT_DIR = PROJECT / "results"


def transformer_blocks(net) -> list:
    """Repeated blocks inside the transformer, found by walking the module."""
    transformer = net.transf_unit.transformer
    layers = getattr(transformer, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
        return list(layers)
    found = [module for module in transformer.modules()
             if isinstance(module, nn.ModuleList) and len(module) >= 4]
    if not found:
        return []
    return list(max(found, key=len))


def parameter_report(net) -> dict:
    total = sum(parameter.numel() for parameter in net.parameters())
    blocks = transformer_blocks(net)
    rows = []
    for index, block in enumerate(blocks):
        rows.append({
            "index": index,
            "params": int(sum(parameter.numel() for parameter in block.parameters())),
        })
    after_third = int(sum(row["params"] for row in rows[3:]))
    return {
        "total": int(total),
        "n_blocks": len(blocks),
        "blocks": rows,
        "params_in_blocks_after_the_third": after_third,
    }


def time_embed(net, features: np.ndarray, device: str, batch_size: int) -> float:
    from src.models import embed

    warmup = features[: min(32, len(features))]
    embed(net, warmup, device, batch_size=min(batch_size, len(warmup)))
    sync(device)
    started = time.perf_counter()
    embed(net, features, device, batch_size=batch_size)
    sync(device)
    return time.perf_counter() - started


def synthetic(n: int, channels: int, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, channels, length)).astype(np.float32)


def main():
    require_env()
    device = pick_device("auto")
    set_seed(0)
    from src.models import load_mantis

    print(f"device {device}")
    net = load_mantis(device, pretrained=True, layer=-1, token="cls_token")
    report = {"device": device, "notes": []}
    try:
        report["params"] = parameter_report(net)
        print(f"parameters {report['params']['total']}  blocks {report['params']['n_blocks']}")
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"parameter census failed: {exc}")

    batch = synthetic(256, 1, 512, seed=0)
    try:
        full = time_embed(net, batch, device, batch_size=128)
        report["early_exit"] = [{"request": "layer -1, model built that way", "seconds": full}]
        if hasattr(net, "return_transf_layer"):
            previous = net.return_transf_layer
            try:
                net.return_transf_layer = 0
                early = time_embed(net, batch, device, batch_size=128)
            finally:
                net.return_transf_layer = previous
            report["early_exit"].append({
                "request": "layer 0, attribute set on the full model",
                "seconds": early,
            })
            ratio = early / full if full else float("nan")
            report["early_exit_time_ratio_layer0_over_last"] = ratio
            if ratio > 0.9:
                report["notes"].append(
                    "Asking for the first layer took at least 90% of the full forward time. "
                    "The later blocks are still being executed."
                )
        early_net = load_mantis(device, pretrained=True, layer=0, token="cls_token")
        built = time_embed(early_net, batch, device, batch_size=128)
        report["early_exit"].append({
            "request": "layer 0, model constructed with that request",
            "seconds": built,
        })
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"early-exit timing failed: {exc}")

    report["scale_n"] = []
    for n in (64, 256, 1024):
        try:
            features = synthetic(n, 1, 512, seed=1)
            seconds = time_embed(net, features, device, batch_size=128)
            report["scale_n"].append({"n": n, "channels": 1, "seconds": seconds, "series_per_s": n / seconds})
            print(f"n {n:5d}  {seconds:.3f}s")
        except Exception as exc:  # noqa: BLE001
            report["notes"].append(f"scale_n {n} failed: {exc}")

    report["scale_channels"] = []
    for channels in (1, 4, 16, 32):
        try:
            features = synthetic(128, channels, 512, seed=2)
            seconds = time_embed(net, features, device, batch_size=64)
            report["scale_channels"].append({
                "n": 128, "channels": channels, "seconds": seconds,
                "seconds_per_channel": seconds / channels,
            })
            print(f"channels {channels:3d}  {seconds:.3f}s")
        except Exception as exc:  # noqa: BLE001
            report["notes"].append(f"scale_channels {channels} failed: {exc}")

    report["resize"] = []
    try:
        from src.data import preprocess

        for length in (64, 512, 4096, 16384):
            raw = synthetic(128, 1, length, seed=3)
            started = time.perf_counter()
            preprocess(raw, length=512)
            report["resize"].append({"n": 128, "raw_length": length, "seconds": time.perf_counter() - started})
            print(f"resize {length:5d} -> 512  {report['resize'][-1]['seconds']:.3f}s")
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"resize timing failed: {exc}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(report, OUT_DIR / "bottleneck.json")
    print(f"wrote {OUT_DIR / 'bottleneck.json'}")


if __name__ == "__main__":
    main()
