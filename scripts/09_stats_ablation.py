"""Ablation: which statistics group carries the Mantis + S gain?

Adds one group at a time to M (level, slope, diff, all) with RF200 and ridge.
The M-only baseline comes from results/fusion.csv (same seeds and cache).
Exploratory: added after the fusion results were seen.
Writes results/stats_ablation.csv. Report: scripts/08_fusion_report.py.

Usage
  python scripts/09_stats_ablation.py --datasets all
  python scripts/09_stats_ablation.py --datasets ucr --seeds 0
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
from src.features import RIDGE_ALPHAS, STAT_NAMES, make_ridge, series_stats  # noqa: E402
from src.models import cached_embeddings  # noqa: E402

SUBSETS = {
    "level": ["mean", "std", "min", "max"],
    "slope": ["slope"],
    "diff": ["diff_mean", "diff_std", "diff_min", "diff_max"],
    "all": list(STAT_NAMES),
}
CLASSIFIERS = ["RF200", "ridge"]


def select_stats(S: np.ndarray, n_channels: int, names: list[str]) -> np.ndarray:
    idx = []
    for name in names:
        i = STAT_NAMES.index(name)
        idx.extend(range(i * n_channels, (i + 1) * n_channels))
    return S[:, idx]


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--weighting", default="equal", choices=["equal", "none"])
    ap.add_argument("--device", default="auto", help="only used if an embedding is not cached yet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(RESULTS_DIR / "stats_ablation.csv", ["dataset", "seed", "subset", "classifier"])
    failures = CsvLog(RESULTS_DIR / "stats_ablation_failures.csv",
                      ["dataset", "seed", "subset", "classifier"])
    runs = [(s, c) for s in SUBSETS for c in CLASSIFIERS]

    def pending(name, seed):
        return [(s, c) for s, c in runs if not log.done(dataset=name, seed=seed, subset=s, classifier=c)]

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
            if not failures.done(dataset=name, seed="", subset="load", classifier=""):
                failures.write({"dataset": name, "seed": "", "subset": "load", "classifier": "",
                                "error": f"{type(e).__name__}: {e}"[:500]})
            continue
        meta, ytr, yte = data["meta"], data["y_train"], data["y_test"]
        C = meta["n_channels"]

        for seed in args.seeds:
            for subset, clf_name in pending(name, seed):
                try:
                    Xtr = np.hstack([Ztr, select_stats(Str, C, SUBSETS[subset])])
                    Xte = np.hstack([Zte, select_stats(Ste, C, SUBSETS[subset])])
                    sizes = [Ztr.shape[1], Xtr.shape[1] - Ztr.shape[1]]
                    if clf_name == "ridge":
                        clf = make_ridge(sizes, args.weighting)
                    else:
                        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                                     random_state=seed, n_jobs=args.n_jobs)
                    t0 = time.perf_counter()
                    clf.fit(Xtr, ytr)
                    fit_secs = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = clf.predict(Xte)
                    pred_secs = time.perf_counter() - t0
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    failures.write({"dataset": name, "seed": seed, "subset": subset,
                                    "classifier": clf_name, "error": f"{type(e).__name__}: {e}"[:500]})
                    continue

                alpha = float(clf[-1].alpha_) if clf_name == "ridge" else np.nan
                transform_secs = emb_secs + stat_secs
                acc = float((pred == yte).mean())
                log.write({
                    "archive": archive_of(name), "dataset": name, "seed": seed,
                    "subset": subset, "stat_names": "|".join(SUBSETS[subset]),
                    "features": f"M+{subset}", "classifier": clf_name, "weighting": args.weighting,
                    "accuracy": acc,
                    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                    "n_features": int(Xtr.shape[1]),
                    "feature_mb": round(Xtr.shape[0] * Xtr.shape[1] * 4 / 2**20, 3),
                    "ridge_alpha": alpha,
                    "alpha_at_grid_edge": int(alpha in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1]))
                    if clf_name == "ridge" else "",
                    "transform_seconds": round(transform_secs, 3),
                    "fit_seconds": round(fit_secs, 3), "predict_seconds": round(pred_secs, 3),
                    "total_seconds": round(transform_secs + fit_secs + pred_secs, 3),
                    "n_train": meta["n_train"], "n_test": meta["n_test"],
                    "n_classes": meta["n_classes"], "n_channels": C,
                    "min_len": meta["min_len"], "max_len": meta["max_len"],
                    "unequal_length": meta["unequal_length"],
                    "aeon": info.get("aeon"), "sklearn": info.get("sklearn"),
                    "hardware": info["hardware"], "checkpoint_revision": CHECKPOINT_REVISION,
                })
                print(f"{name:28s} seed {seed}  M+{subset:6s} {clf_name:6s} acc {acc:.4f}")


if __name__ == "__main__":
    main()
