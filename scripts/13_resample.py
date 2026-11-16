"""Stratified train/test resamples (resample 0 is the official split).

Train and test are pooled and re-split with StratifiedShuffleSplit at the
official sizes. Embeddings are re-indexed; MiniRocket and every head are refitted
on each training part. Methods use the features:head format of scripts 07 and 11.
Writes results/resample.csv. Report: scripts/15_report.py.

Usage
  python scripts/13_resample.py --datasets ucr --resamples 10
  python scripts/13_resample.py --datasets all --resamples 10 --min-len 512 \\
      --methods M:RF200,M:ridge,R:ridge,M+W2mean:RF200,M+W2mean:ridge
"""
import argparse
import json
import re
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
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit  # noqa: E402

from configs.datasets import archive_of, resolve  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import make_minirocket_transform, make_ridge, series_stats  # noqa: E402
from src.models import cached_embeddings, load_mantis  # noqa: E402
from src.windows import cached_window_embeddings, pool_windows  # noqa: E402

DEFAULT_METHODS = [("M","RF200"), ("M","ridge"), ("R","ridge"), ("R+M","ridge")]
HEADS =("RF200","ridge")
BLOCKS =("M","R","S")
WINDOW =re.compile(r"^W(\d+)(mean|concat)$")


def valid_block(part: str) -> bool:
    return part in BLOCKS or WINDOW.match(part) is not None


def parse_methods(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return list(DEFAULT_METHODS)
    out = []
    for item in text.split(","):
        feats, clf = item.strip().split(":")
        if clf not in HEADS or not all(valid_block(p) for p in feats.split("+")):
            raise SystemExit(f"unknown method {item!r}; blocks are {', '.join(BLOCKS)} or W<k>mean / W<k>concat, "
                             f"heads are {', '.join(HEADS)}")
        out.append((feats, clf))
    return out


def resample_indices(y_all: np.ndarray, n_train: int, resample: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(y_all)
    if resample == 0:
        return np.arange(n_train), np.arange(n_train, n), 1
    try:
        tr, te = next(StratifiedShuffleSplit(n_splits=1, train_size=n_train, test_size=n - n_train,
                                             random_state=resample).split(np.zeros(n), y_all))
        return np.sort(tr), np.sort(te), 1
    except ValueError:
        tr, te = next(ShuffleSplit(n_splits=1, train_size=n_train, test_size=n - n_train,
                                   random_state=resample).split(np.zeros(n)))
        return np.sort(tr), np.sort(te), 0


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="ucr")
    ap.add_argument("--resamples", type=int, default=10, help="including resample 0, the official split")
    ap.add_argument("--methods", default=None, help="comma-separated features:head pairs")
    ap.add_argument("--min-len", type=int, default=0,
                    help="only datasets whose raw max length exceeds this (512 selects the windowed group)")
    ap.add_argument("--overlap", type=float, default=0.0, help="window overlap, as in 11_multiscale.py")
    ap.add_argument("--seed", type=int, default=0, help="seed of the forest and the MiniRocket kernels")
    ap.add_argument("--weighting", default="equal", choices=["equal", "none"])
    ap.add_argument("--device", default="auto", help="only used if an embedding is not cached yet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    methods = parse_methods(args.methods)

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(RESULTS_DIR / "resample.csv", ["dataset", "resample", "features", "classifier"])
    failures = CsvLog(RESULTS_DIR / "resample_failures.csv", ["dataset", "resample", "features", "classifier"])
    resamples = list(range(args.resamples))
    window_blocks = sorted({p for f, _ in methods for p in f.split("+") if WINDOW.match(p)})

    def pending(name, r):
        return [(f, c) for f, c in methods if not log.done(dataset=name, resample=r, features=f, classifier=c)]

    net = None
    for name in resolve(args.datasets):
        if not any(pending(name, r) for r in resamples):
            continue
        try:
            data = load_dataset(name)
            if data["meta"]["max_len"] <= args.min_len:
                continue
            Mtr, Mte, _, _ = cached_embeddings(name, data, device, batch_size=args.batch_size)
            W_all = {}
            for spec in window_blocks:
                k, how = WINDOW.match(spec).groups()
                if net is None:
                    net = load_mantis(device)
                Wtr, Wte, _, _ = cached_window_embeddings(name, int(k), args.overlap, device,
                                                           args.batch_size, net)
                W_all[spec] = np.vstack([pool_windows(Wtr, how), pool_windows(Wte, how)])
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failures.write({"dataset": name, "resample": "", "features": "load", "classifier": "",
                            "error": f"{type(e).__name__}: {e}"[:500]})
            continue
        meta = data["meta"]
        X_all = np.concatenate([data["X_train"], data["X_test"]])
        y_all = np.concatenate([data["y_train"], data["y_test"]])
        M_all = np.vstack([Mtr, Mte])
        S_all = series_stats(X_all) if any("S" in f.split("+") for f, _ in methods) else None
        n_train = meta["n_train"]

        for r in resamples:
            todo = pending(name, r)
            if not todo:
                continue
            tr, te, stratified = resample_indices(y_all, n_train, r)
            ytr, yte = y_all[tr], y_all[te]
            if len(np.unique(ytr)) < 2:
                print(f"[skip] {name} resample {r}: one class in the training part")
                continue
            # per-series features are re-indexed; MiniRocket is refitted below
            blocks = {"M": (M_all[tr], M_all[te])}
            for spec, W in W_all.items():
                blocks[spec] = (W[tr], W[te])
            if S_all is not None:
                blocks["S"] = (S_all[tr], S_all[te])
            if any("R" in f.split("+") for f, _ in todo):
                try:
                    mr = make_minirocket_transform(args.seed, args.n_jobs).fit(X_all[tr])
                    blocks["R"] = (np.asarray(mr.transform(X_all[tr]), dtype=np.float32),
                                   np.asarray(mr.transform(X_all[te]), dtype=np.float32))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    print(f"[fail] {name} resample {r} MiniRocket: {e}")

            for feats, clf_name in todo:
                parts = feats.split("+")
                if any(p not in blocks for p in parts):
                    continue
                Xtr = np.hstack([blocks[p][0] for p in parts])
                Xte = np.hstack([blocks[p][1] for p in parts])
                sizes = [blocks[p][0].shape[1] for p in parts]
                try:
                    if clf_name == "ridge":
                        clf = make_ridge(sizes, args.weighting)
                    else:
                        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                                     random_state=args.seed, n_jobs=args.n_jobs)
                    t0 = time.perf_counter()
                    clf.fit(Xtr, ytr)
                    pred = clf.predict(Xte)
                    secs = time.perf_counter() - t0
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    failures.write({"dataset": name, "resample": r, "features": feats,
                                    "classifier": clf_name, "error": f"{type(e).__name__}: {e}"[:500]})
                    continue
                acc = float((pred == yte).mean())
                log.write({
                    "archive": archive_of(name), "dataset": name, "resample": r,
                    "stratified": stratified, "features": feats, "classifier": clf_name,
                    "seed": args.seed, "accuracy": acc,
                    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                    "n_features": int(Xtr.shape[1]), "seconds": round(secs, 3),
                    "n_train": len(tr), "n_test": len(te),
                    "n_classes": meta["n_classes"], "n_channels": meta["n_channels"],
                    "max_len": meta["max_len"], "sklearn": info.get("sklearn"),
                    "hardware": info["hardware"], "checkpoint_revision": CHECKPOINT_REVISION,
                })
                print(f"{name:28s} resample {r:2d}  {feats:12s} {clf_name:6s} acc {acc:.4f}")


if __name__ == "__main__":
    main()
