"""Section 4 improvement: fuse frozen Mantis embeddings with MiniRocket features
and raw-series statistics (early fusion: concatenate, then one linear classifier).

Question: do foundation-model embeddings add information that a strong
random-kernel method does not already capture, and does restoring the scale
information that Mantis normalises away fix its failures (e.g. EthanolLevel)?

Feature blocks (see src/features.py):
  M  Mantis-8M, last layer, CLS token (the cached zero-shot embeddings)
  S  9 statistics per channel of the raw series and its first difference
  R  MiniRocket features, fitted on the training set

Methods written to results/fusion.csv (one row per dataset, seed, method):
  features   classifier   role
  M          RF200        zero-shot Mantis as in v1 (should equal zero_shot.csv)
  M+S        RF200        does scale information help Mantis on its own?
  M          ridge        separates the classifier change from the feature change
  S          ridge        statistics alone (the course's "stats" baseline)
  R          ridge        MiniRocket (should match minirocket.csv)
  M+S        ridge
  R+S        ridge
  R+M        ridge        main question: does Mantis add to MiniRocket?
  R+M+S      ridge        full model
  none       dummy        majority-class baseline (lecture week 7 checklist)

Ridge = StandardScaler -> equal block weights -> RidgeClassifierCV (alpha by
leave-one-out CV on train). Same input as every other script: missing values
interpolated, series resized to 512. Official train/test splits; the seed
changes MiniRocket's biases and the random forest. Times are wall-clock.

Usage
  python scripts/07_fusion.py --datasets all
  python scripts/07_fusion.py --datasets tier1 --seeds 0
  python scripts/07_fusion.py --datasets ucr --methods R+M:ridge,R:ridge
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
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from configs.datasets import OTHERS, resolve, ucr128, uea27  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import (RIDGE_ALPHAS, make_minirocket_transform, make_ridge,  # noqa: E402
                          series_stats)
from src.models import cached_embeddings  # noqa: E402

METHODS = [
    ("M", "RF200"), ("M+S", "RF200"),
    ("M", "ridge"), ("S", "ridge"), ("R", "ridge"),
    ("M+S", "ridge"), ("R+S", "ridge"), ("R+M", "ridge"), ("R+M+S", "ridge"),
    ("none", "dummy"),
]


def archive_of(name: str) -> str:
    if name in OTHERS:
        return "other"
    if name in uea27():
        return "uea"
    if name in ucr128():
        return "ucr"
    return "unknown"


def parse_methods(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return list(METHODS)
    out = []
    for item in text.split(","):
        feats, clf = item.strip().split(":")
        if (feats, clf) not in METHODS:
            raise SystemExit(f"unknown method {item!r}; choose from "
                             + ", ".join(f"{f}:{c}" for f, c in METHODS))
        out.append((feats, clf))
    return out


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", default=None,
                    help="comma-separated features:classifier pairs, e.g. R+M:ridge,R:ridge (default: all)")
    ap.add_argument("--weighting", default="equal", choices=["equal", "none"],
                    help="block weighting before the ridge (default equal; 'none' is an ablation)")
    ap.add_argument("--device", default="auto", help="only used if an embedding is not cached yet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    methods = parse_methods(args.methods)
    out = Path(args.out) if args.out else RESULTS_DIR / (
        "fusion.csv" if args.weighting == "equal" else f"fusion_weighting-{args.weighting}.csv")

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(out, ["dataset", "seed", "features", "classifier"])
    failures = CsvLog(RESULTS_DIR / "fusion_failures.csv", ["dataset", "seed", "features", "classifier"])

    def pending(name, seed):
        return [(f, c) for f, c in methods
                if not log.done(dataset=name, seed=seed, features=f, classifier=c)]

    for name in resolve(args.datasets):
        if not any(pending(name, s) for s in args.seeds):
            continue
        try:
            data = load_dataset(name)
            Ztr, Zte, emb_secs, _ = cached_embeddings(name, data, device, batch_size=args.batch_size)
            t0 = time.perf_counter()
            Str, Ste = series_stats(data["X_train"]), series_stats(data["X_test"])
            stat_secs = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            if not failures.done(dataset=name, seed="", features="load", classifier=""):
                failures.write({"dataset": name, "seed": "", "features": "load", "classifier": "",
                                "error": f"{type(e).__name__}: {e}"[:500]})
            continue
        meta, ytr, yte = data["meta"], data["y_train"], data["y_test"]

        for seed in args.seeds:
            todo = pending(name, seed)
            if not todo:
                continue
            blocks = {"M": (Ztr, Zte, emb_secs), "S": (Str, Ste, stat_secs)}
            if any("R" in f.split("+") for f, _ in todo):
                try:
                    t0 = time.perf_counter()
                    mr = make_minirocket_transform(seed, args.n_jobs).fit(data["X_train"])
                    Rtr = np.asarray(mr.transform(data["X_train"]), dtype=np.float32)
                    Rte = np.asarray(mr.transform(data["X_test"]), dtype=np.float32)
                    blocks["R"] = (Rtr, Rte, time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    print(f"[fail] {name} seed {seed} MiniRocket: {e}")

            for feats, clf_name in todo:
                parts = [] if feats == "none" else feats.split("+")
                if any(p not in blocks for p in parts):
                    continue
                try:
                    if parts:
                        Xtr = np.hstack([blocks[p][0] for p in parts])
                        Xte = np.hstack([blocks[p][1] for p in parts])
                    else:
                        Xtr, Xte = np.zeros((len(ytr), 1)), np.zeros((len(yte), 1))
                    sizes = [blocks[p][0].shape[1] for p in parts]
                    if clf_name == "ridge":
                        clf = make_ridge(sizes, args.weighting)
                    elif clf_name == "RF200":
                        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                                     random_state=seed, n_jobs=args.n_jobs)
                    else:
                        clf = DummyClassifier(strategy="most_frequent")
                    t0 = time.perf_counter()
                    clf.fit(Xtr, ytr)
                    fit_secs = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = clf.predict(Xte)
                    pred_secs = time.perf_counter() - t0
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    failures.write({"dataset": name, "seed": seed, "features": feats,
                                    "classifier": clf_name, "error": f"{type(e).__name__}: {e}"[:500]})
                    continue

                alpha = float(clf[-1].alpha_) if clf_name == "ridge" else np.nan
                transform_secs = sum(blocks[p][2] for p in parts)
                acc = float((pred == yte).mean())
                log.write({
                    "archive": archive_of(name), "dataset": name, "seed": seed,
                    "features": feats, "classifier": clf_name, "weighting": args.weighting,
                    "accuracy": acc,
                    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                    "n_features": int(Xtr.shape[1]) if parts else 0,
                    "feature_mb": round(Xtr.shape[0] * Xtr.shape[1] * 4 / 2**20, 3) if parts else 0.0,
                    "ridge_alpha": alpha,
                    "alpha_at_grid_edge": int(alpha in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1]))
                    if clf_name == "ridge" else "",
                    "transform_seconds": round(transform_secs, 3),
                    "fit_seconds": round(fit_secs, 3), "predict_seconds": round(pred_secs, 3),
                    "total_seconds": round(transform_secs + fit_secs + pred_secs, 3),
                    "n_train": meta["n_train"], "n_test": meta["n_test"],
                    "n_classes": meta["n_classes"], "n_channels": meta["n_channels"],
                    "min_len": meta["min_len"], "max_len": meta["max_len"],
                    "unequal_length": meta["unequal_length"],
                    "aeon": info.get("aeon"), "sklearn": info.get("sklearn"),
                    "hardware": info["hardware"], "checkpoint_revision": CHECKPOINT_REVISION,
                })
                print(f"{name:28s} seed {seed}  {feats:6s} {clf_name:6s} acc {acc:.4f}")


if __name__ == "__main__":
    main()
