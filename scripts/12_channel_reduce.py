"""Improvement: channel reduction before the encoder (src/channels.py).

Question: Mantis embeds each channel separately, so cost and feature count grow
with the channel count. If a multivariate series is first reduced to k channels
with a standard method (PCA fitted on the training series, a random channel
subset, or summary channels), how much accuracy is lost and how much time saved?

Datasets: those with more than --min-channels channels (default 8), fixed from
metadata before any result is seen.

Methods (k = --k, default 8):
  full        all channels, the published embedding (cached by 02_zero_shot.py)
  pca{k}      StandardScaler + PCA over channels, fitted on train only
  random{k}   k channels chosen at random (seeded)
  summary     3 derived channels: L2 magnitude, mean, std across channels
  magnitude   1 derived channel: L2 magnitude
Heads: RF200 and ridge. The forward-pass time of the reduced input is measured
here; the full input's time comes from the embedding cache.

Results: results/channel_reduce.csv. Report: scripts/15_improve_report.py.

Usage
  python scripts/12_channel_reduce.py --datasets uea,others
  python scripts/12_channel_reduce.py --datasets DuckDuckGeese,PEMS-SF --k 8 16
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (CACHE_DIR, CHECKPOINT_REVISION, RESULTS_DIR, CsvLog, env_info,  # noqa: E402
                        pick_device, require_env)

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from configs.datasets import archive_of, resolve  # noqa: E402
from src.channels import reduce_channels  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import make_ridge  # noqa: E402
from src.models import cached_embeddings, embed, load_mantis  # noqa: E402

HEADS = ["RF200", "ridge"]


def resolve_groups(selection: str) -> list[str]:
    names = []
    for part in selection.split(","):
        names += resolve(part.strip())
    return list(dict.fromkeys(names))


def reduced_embeddings(name, method, k, seed, data, device, batch_size, net):
    tag = f"mantis8m_rev-{CHECKPOINT_REVISION or 'latest'}_{method}{k}_s{seed}"
    path = CACHE_DIR / "emb" / "channels" / tag / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["Z_train"], z["Z_test"], float(z["seconds"]), int(z["k_used"])
    # anything fitted (PCA) uses the training series only
    Xtr, Xte = reduce_channels(method, data["X_train"], data["X_test"], k, seed)
    Ztr, t1 = embed(net, Xtr, device, batch_size)
    Zte, t2 = embed(net, Xte, device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Z_train=Ztr, Z_test=Zte, seconds=t1 + t2, k_used=Xtr.shape[1])
    return Ztr, Zte, t1 + t2, int(Xtr.shape[1])


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="uea,others")
    ap.add_argument("--min-channels", type=int, default=8)
    ap.add_argument("--k", type=int, nargs="+", default=[8])
    ap.add_argument("--methods", nargs="+", default=["pca", "random", "summary", "magnitude"])
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(RESULTS_DIR / "channel_reduce.csv", ["dataset", "seed", "method", "classifier"])
    failures = CsvLog(RESULTS_DIR / "channel_reduce_failures.csv", ["dataset", "seed", "method", "classifier"])

    methods = ["full"]
    for m in args.methods:
        methods += [f"{m}{k}" for k in args.k] if m in ("pca", "random") else [m]
    runs = [(m, h) for m in methods for h in args.heads]

    def pending(name, seed):
        return [(m, h) for m, h in runs if not log.done(dataset=name, seed=seed, method=m, classifier=h)]

    net = None
    for name in resolve_groups(args.datasets):
        if not any(pending(name, s) for s in args.seeds):
            continue
        try:
            data = load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue
        meta, ytr, yte = data["meta"], data["y_train"], data["y_test"]
        if meta["n_channels"] <= args.min_channels:
            continue

        for seed in args.seeds:
            for method, head in pending(name, seed):
                try:
                    if method == "full":
                        Ztr, Zte, emb_secs, _ = cached_embeddings(name, data, device, batch_size=args.batch_size)
                        k_used = meta["n_channels"]
                    else:
                        # 'pca8' means method pca with k = 8; summary and magnitude have no k
                        base = method.rstrip("0123456789")
                        k = int(method[len(base):]) if method != base else 0
                        if net is None:
                            net = load_mantis(device)
                        Ztr, Zte, emb_secs, k_used = reduced_embeddings(
                            name, base, k, seed, data, device, args.batch_size, net)
                    if head == "ridge":
                        clf = make_ridge([Ztr.shape[1]])
                    else:
                        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                                     random_state=seed, n_jobs=args.n_jobs)
                    t0 = time.perf_counter()
                    clf.fit(Ztr, ytr)
                    fit_secs = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = clf.predict(Zte)
                    pred_secs = time.perf_counter() - t0
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    failures.write({"dataset": name, "seed": seed, "method": method, "classifier": head,
                                    "error": f"{type(e).__name__}: {e}"[:500]})
                    continue
                acc = float((pred == yte).mean())
                log.write({
                    "archive": archive_of(name), "dataset": name, "seed": seed,
                    "method": method, "classifier": head, "channels_in": meta["n_channels"],
                    "channels_used": k_used, "accuracy": acc,
                    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                    "n_features": int(Ztr.shape[1]),
                    "feature_mb": round(Ztr.shape[0] * Ztr.shape[1] * 4 / 2**20, 3),
                    "embed_seconds": round(emb_secs, 3), "fit_seconds": round(fit_secs, 3),
                    "predict_seconds": round(pred_secs, 3),
                    "n_train": meta["n_train"], "n_test": meta["n_test"],
                    "n_classes": meta["n_classes"], "max_len": meta["max_len"],
                    "device": device, "sklearn": info.get("sklearn"), "hardware": info["hardware"],
                    "checkpoint_revision": CHECKPOINT_REVISION,
                })
                print(f"{name:28s} seed {seed}  {method:10s} {head:6s} acc {acc:.4f}  "
                      f"embed {emb_secs:7.1f}s  ch {k_used}/{meta['n_channels']}")


if __name__ == "__main__":
    main()
