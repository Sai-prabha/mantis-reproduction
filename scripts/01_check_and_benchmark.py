"""Environment checks and timing benchmark. Run this first on a new machine.

Checks (results are reported in Section 3):
  1. Versions, device, and the checkpoint commit (pinned in src/common.py).
  2. Parameter count of Mantis-8M (paper: ~8M; Hugging Face: 8.11M).
  3. MantisV1 vs the legacy Mantis8M class: identical embeddings?
  4. MPS vs CPU: how far apart are the embeddings?
  5. Determinism: same seed twice on the accelerator, identical results?
Benchmark:
  6. Zero-shot embedding throughput (series per second).
  7. Full fine-tuning time per sample (forward + backward), then an estimate of
     the tier-1 fine-tuning cost from results/dataset_stats.csv if present.

Usage
  python scripts/01_check_and_benchmark.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (CHECKPOINT, CHECKPOINT_REVISION, RESULTS_DIR, env_info,  # noqa: E402
                        pick_device, require_env, save_json, set_seed, sync)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.models import count_params, load_mantis, paper_head  # noqa: E402


def embed_raw(net, X, device, bs=256):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(torch.tensor(X[i:i + bs]).to(device)).float().cpu())
    return torch.cat(out).numpy()


def time_embedding(net, device, n=4096, bs=256):
    X = np.random.default_rng(0).standard_normal((n, 1, 512)).astype(np.float32)
    embed_raw(net, X[:bs], device, bs)  # warm-up
    sync(device)
    t0 = time.perf_counter()
    embed_raw(net, X, device, bs)
    sync(device)
    return n / (time.perf_counter() - t0)


def time_finetune_step(device, n_steps=6, bs=256, n_classes=5):
    """Seconds per training sample for full fine-tuning (forward + backward + step)."""
    from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
    enc = load_mantis(device)
    model = FineTuningNetwork(enc, paper_head(enc.hidden_dim, n_classes).to(device)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    x = torch.randn(bs, 1, 512, device=device)
    y = torch.randint(0, n_classes, (bs,), device=device)
    crit = nn.CrossEntropyLoss()
    model.train()
    for i in range(n_steps + 2):
        if i == 2:  # skip warm-up steps
            sync(device)
            t0 = time.perf_counter()
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync(device)
    return (time.perf_counter() - t0) / (n_steps * bs)


def main():
    require_env()
    device = pick_device("auto")
    report = {"env": env_info(device)}
    print("== environment ==")
    for k, v in report["env"].items():
        print(f"  {k:10s} {v}")

    # 1. checkpoint commit
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(CHECKPOINT, revision=CHECKPOINT_REVISION).sha
        report["checkpoint_sha"] = sha
        print(f"\n== checkpoint ==\n  {CHECKPOINT} @ {sha}")
        if not CHECKPOINT_REVISION:
            print(f"  -> pin it: export MANTIS_REVISION={sha}  (and record it in the README)")
    except Exception as e:  # noqa: BLE001
        print(f"  could not query the Hub for the commit hash: {e}")

    # 2. parameter count
    net = load_mantis(device)
    report["n_params"] = count_params(net)
    report["n_buffers"] = sum(b.numel() for b in net.buffers())
    print(f"\n== parameters ==\n  MantisV1 / Mantis-8M: {report['n_params']:,} trainable"
          f" + {report['n_buffers']:,} in buffers (e.g. positional encoding)"
          f" = {report['n_params'] + report['n_buffers']:,} stored values")

    X = np.random.default_rng(1).standard_normal((512, 1, 512)).astype(np.float32)
    Z_dev = embed_raw(net, X, device)

    # 3. legacy class
    try:
        from mantis.architecture import Mantis8M
        legacy = Mantis8M.from_pretrained(CHECKPOINT, revision=CHECKPOINT_REVISION, device=device).to(device)
        Z_leg = embed_raw(legacy, X, device)
        diff = float(np.abs(Z_dev - Z_leg).max())
        report["legacy_class_max_abs_diff"] = diff
        print(f"\n== MantisV1 vs legacy Mantis8M ==\n  max |diff| = {diff:.3e}")
    except Exception as e:  # noqa: BLE001
        report["legacy_class_error"] = str(e)
        print(f"\n== legacy Mantis8M class failed to load: {e}")

    # 4. accelerator vs CPU
    if device != "cpu":
        Z_cpu = embed_raw(load_mantis("cpu"), X, "cpu")
        diff = float(np.abs(Z_dev - Z_cpu).max())
        rel = float(np.abs(Z_dev - Z_cpu).max() / (np.abs(Z_cpu).max() + 1e-12))
        report["device_vs_cpu_max_abs_diff"] = diff
        report["device_vs_cpu_max_rel_diff"] = rel
        print(f"\n== {device} vs cpu embeddings ==\n  max |diff| = {diff:.3e} (relative {rel:.3e})")

    # 5. determinism of a short training run on the accelerator
    def short_run(seed):
        set_seed(seed)
        from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
        enc = load_mantis(device)
        m = FineTuningNetwork(enc, paper_head(256, 3).to(device)).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(64, 1, 512, generator=g).to(device)
        y = torch.randint(0, 3, (64,), generator=g).to(device)
        m.train()
        for _ in range(3):
            loss = nn.functional.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            return m(x).float().cpu().numpy()
    a, b = short_run(0), short_run(0)
    report["same_seed_max_abs_diff"] = float(np.abs(a - b).max())
    print(f"\n== determinism ({device}, same seed twice) ==\n  max |diff| = {report['same_seed_max_abs_diff']:.3e}")

    # 6. embedding throughput
    print("\n== timing ==")
    report["embed_series_per_s"] = {device: time_embedding(net, device)}
    if device != "cpu":
        report["embed_series_per_s"]["cpu"] = time_embedding(load_mantis("cpu"), "cpu", n=1024)
    for d, v in report["embed_series_per_s"].items():
        print(f"  embedding  {d:4s}: {v:8.0f} series/s")

    # 7. fine-tuning cost
    report["ft_seconds_per_sample"] = {device: time_finetune_step(device)}
    if device != "cpu":
        report["ft_seconds_per_sample"]["cpu"] = time_finetune_step("cpu", n_steps=2)
    for d, v in report["ft_seconds_per_sample"].items():
        print(f"  fine-tune  {d:4s}: {v * 1e3:8.3f} ms per sample per epoch")

    stats = RESULTS_DIR / "dataset_stats.csv"
    if stats.exists():
        s = pd.read_csv(stats)
        spp = report["ft_seconds_per_sample"][device]
        # per dataset: full + scratch, each = 3 LR-selection runs + 2 more seeds = 5 runs,
        # on 80% of train for 100 epochs; evaluation adds roughly a third (eval every 5 epochs)
        train_cost = (0.8 * s["n_train"] * 100 * 10 * spp).sum()
        eval_cost = ((0.2 * s["n_train"] + s["n_test"]) * 20 * 10 * spp / 3).sum()
        hours = (train_cost + eval_cost) / 3600
        report["tier1_estimate_hours"] = hours
        print(f"\n== estimate ==\n  tier-1 fine-tuning ({len(s)} datasets): about {hours:.1f} h on {device}")
    else:
        print("\n  (run scripts/00_dataset_stats.py first to get a tier-1 time estimate)")

    save_json(report, RESULTS_DIR / "check_and_benchmark.json")
    print(f"\nsaved {RESULTS_DIR / 'check_and_benchmark.json'}")


if __name__ == "__main__":
    main()
