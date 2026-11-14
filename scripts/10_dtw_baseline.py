"""Course baseline (lecture weeks 5 and 7): 1-nearest-neighbour with DTW distance
and a 10% Sakoe-Chiba warping window, on the same preprocessed input as every
other method (missing values interpolated, every series resized to 512).

1NN-DTW has no random component, so there is one run per dataset (seed 0).
Its cost grows with n_train x n_test x n_channels x window x length: datasets
above --max-pairs (n_train x n_test x n_channels) are SKIPPED and listed in
results/dtw_skipped.csv with the reason, so no skip is silent. The report
(scripts/08_fusion_report.py) only compares 1NN-DTW on the datasets it completed.

Usage
  python scripts/10_dtw_baseline.py --datasets all
  python scripts/10_dtw_baseline.py --datasets ucr --max-pairs 500000
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import RESULTS_DIR, CsvLog, env_info, require_env  # noqa: E402

from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from configs.datasets import archive_of, resolve  # noqa: E402
from src.data import load_dataset  # noqa: E402

WINDOW = 0.1  # fraction of the series length (aeon's dtw "window" parameter)


def make_dtw_1nn(n_jobs: int):
    # Signature checked against aeon 1.6.0:
    # KNeighborsTimeSeriesClassifier(distance, distance_params, n_neighbors, weights, n_jobs)
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    return KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw",
                                          distance_params={"window": WINDOW}, n_jobs=n_jobs)


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--max-pairs", type=int, default=2_000_000,
                    help="skip a dataset if n_train * n_test * n_channels exceeds this")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    info = env_info("cpu")
    log = CsvLog(RESULTS_DIR / "dtw.csv", ["dataset"])
    skipped = CsvLog(RESULTS_DIR / "dtw_skipped.csv", ["dataset"])
    for name in resolve(args.datasets):
        if log.done(dataset=name):
            continue
        try:
            data = load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            if not skipped.done(dataset=name):
                skipped.write({"dataset": name, "pairs": "", "max_pairs": args.max_pairs,
                               "reason": f"load failed: {type(e).__name__}: {e}"[:500]})
            continue
        meta = data["meta"]
        pairs = meta["n_train"] * meta["n_test"] * meta["n_channels"]
        if pairs > args.max_pairs:
            print(f"[skip] {name}: {pairs:,} pairs > --max-pairs {args.max_pairs:,}")
            if not skipped.done(dataset=name):
                skipped.write({"dataset": name, "pairs": pairs, "max_pairs": args.max_pairs,
                               "reason": "n_train * n_test * n_channels above --max-pairs"})
            continue
        try:
            t0 = time.perf_counter()
            clf = make_dtw_1nn(args.n_jobs).fit(data["X_train"], data["y_train"])
            pred = clf.predict(data["X_test"])
            secs = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print(f"[fail] {name}: {e}")
            continue
        acc = float((pred == data["y_test"]).mean())
        log.write({"archive": archive_of(name), "dataset": name, "seed": 0, "classifier": "1NN-DTW",
                   "window": WINDOW, "accuracy": acc,
                   "balanced_accuracy": float(balanced_accuracy_score(data["y_test"], pred)),
                   "seconds": round(secs, 2), "pairs": pairs,
                   "n_train": meta["n_train"], "n_test": meta["n_test"],
                   "n_classes": meta["n_classes"], "n_channels": meta["n_channels"],
                   "min_len": meta["min_len"], "max_len": meta["max_len"],
                   "unequal_length": meta["unequal_length"], "input": "resized_512",
                   "aeon": info["aeon"], "hardware": info["hardware"]})
        print(f"{name:32s} acc {acc:.4f}  {secs:8.1f} s  ({pairs:,} pairs)")


if __name__ == "__main__":
    main()
