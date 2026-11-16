"""Improvement: multi-scale (windowed) embeddings for long series (src/windows.py).

Question: the published protocol resizes every series to 512 points. For
series longer than that, does embedding k windows of the RAW series as well
(each resized to 512, same frozen encoder) recover the lost detail?

Datasets: those whose raw series are longer than --min-len (default 512, i.e.
every dataset that the resize downsamples). The group is defined from metadata
before any result is seen (lecture week 9).

Feature sets (M = published whole-series embedding, cached by 02_zero_shot.py):
  M            baseline, as in v1
  M+W{k}mean   M next to the mean of the k window embeddings      (same width as 2 x M)
  M+W{k}concat M next to the k window embeddings side by side      (width (k + 1) x M)
Heads: RF200 (the paper's) and ridge (StandardScaler -> block weights ->
RidgeClassifierCV, as in 07_fusion.py). Official train/test splits; the seed
changes the forest only, the encoder is deterministic.

Results: results/multiscale.csv, one row per dataset, seed, features, head.
Report: scripts/15_improve_report.py.

Usage
  python scripts/11_multiscale.py --datasets all
  python scripts/11_multiscale.py --datasets all --windows 2 4 --overlap 0.5
  python scripts/11_multiscale.py --datasets ACSF1,Rock --seeds 0
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (CHECKPOINT_REVISION, RESULTS_DIR, CsvLog, env_info,  # noqa: E402
                        pick_device, require_env)

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from configs.datasets import archive_of, resolve  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import RIDGE_ALPHAS, make_ridge  # noqa: E402
from src.models import cached_embeddings, load_mantis  # noqa: E402
from src.windows import POOLS, cached_window_embeddings, pool_windows  # noqa: E402

HEADS = ["RF200", "ridge"]


def make_head(name: str, sizes: list[int], seed: int, n_jobs: int, weighting: str):
    if name == "ridge":
        return make_ridge(sizes, weighting)
    if name == "RF200":
        return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=n_jobs)
    raise ValueError(name)


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--min-len", type=int, default=512,
                    help="only datasets whose raw max length exceeds this (default 512)")
    ap.add_argument("--windows", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--overlap", type=float, default=0.0, help="window overlap fraction in [0, 1)")
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--weighting", default="equal", choices=["equal", "none"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(RESULTS_DIR / "multiscale.csv", ["dataset", "seed", "features", "classifier"])
    failures = CsvLog(RESULTS_DIR / "multiscale_failures.csv", ["dataset", "seed", "features", "classifier"])

    feature_sets = ["M"] + [f"M+W{k}{p}" for k in args.windows for p in args.pools]
    runs = [(f, h) for f in feature_sets for h in args.heads]

    def pending(name, seed):
        return [(f, h) for f, h in runs if not log.done(dataset=name, seed=seed, features=f, classifier=h)]

    net = None
    for name in resolve(args.datasets):
        if not any(pending(name, s) for s in args.seeds):
            continue
        try:
            data = load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue
        meta, ytr, yte = data["meta"], data["y_train"], data["y_test"]
        # only series that the resize to 512 downsamples
        if meta["max_len"] <= args.min_len:
            continue
        try:
            # published embedding is cached; window embeddings are cached per (k, overlap)
            Mtr, Mte, m_secs, _ = cached_embeddings(name, data, device, batch_size=args.batch_size)
            blocks = {"M": (Mtr, Mte, m_secs)}
            for k in args.windows:
                if net is None:
                    net = load_mantis(device)
                Wtr, Wte, w_secs, _ = cached_window_embeddings(name, k, args.overlap, device,
                                                                args.batch_size, net)
                for p in args.pools:
                    blocks[f"W{k}{p}"] = (pool_windows(Wtr, p), pool_windows(Wte, p), w_secs)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failures.write({"dataset": name, "seed": "", "features": "embed", "classifier": "",
                            "error": f"{type(e).__name__}: {e}"[:500]})
            continue

        for seed in args.seeds:
            for feats, head in pending(name, seed):
                parts = feats.split("+")
                Xtr = np.hstack([blocks[p][0] for p in parts])
                Xte = np.hstack([blocks[p][1] for p in parts])
                sizes = [blocks[p][0].shape[1] for p in parts]
                try:
                    clf = make_head(head, sizes, seed, args.n_jobs, args.weighting)
                    t0 = time.perf_counter()
                    clf.fit(Xtr, ytr)
                    fit_secs = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = clf.predict(Xte)
                    pred_secs = time.perf_counter() - t0
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    failures.write({"dataset": name, "seed": seed, "features": feats, "classifier": head,
                                    "error": f"{type(e).__name__}: {e}"[:500]})
                    continue
                alpha = float(clf[-1].alpha_) if head == "ridge" else np.nan
                embed_secs = sum(blocks[p][2] for p in parts)
                acc = float((pred == yte).mean())
                log.write({
                    "archive": archive_of(name), "dataset": name, "seed": seed,
                    "features": feats, "classifier": head, "overlap": args.overlap,
                    "weighting": args.weighting, "accuracy": acc,
                    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                    "n_features": int(Xtr.shape[1]),
                    "ridge_alpha": alpha,
                    "alpha_at_grid_edge": int(alpha in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1]))
                    if head == "ridge" else "",
                    "embed_seconds": round(embed_secs, 3), "fit_seconds": round(fit_secs, 3),
                    "predict_seconds": round(pred_secs, 3),
                    "n_train": meta["n_train"], "n_test": meta["n_test"],
                    "n_classes": meta["n_classes"], "n_channels": meta["n_channels"],
                    "min_len": meta["min_len"], "max_len": meta["max_len"],
                    "unequal_length": meta["unequal_length"],
                    "sklearn": info.get("sklearn"), "hardware": info["hardware"],
                    "checkpoint_revision": CHECKPOINT_REVISION,
                })
                print(f"{name:28s} seed {seed}  {feats:12s} {head:6s} acc {acc:.4f}")


if __name__ == "__main__":
    main()
