"""Non-foundation-model baseline: MiniRocket (aeon), on the same preprocessed
input as Mantis (missing values interpolated, every series resized to 512).

Using the identical input removes preprocessing as a confounder; note in the
report that published MiniRocket numbers use native lengths.

Usage
  python scripts/04_minirocket.py --datasets ucr
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import RESULTS_DIR, CsvLog, env_info, require_env  # noqa: E402

from configs.datasets import resolve  # noqa: E402
from src.data import load_dataset  # noqa: E402


def make_minirocket(seed: int, n_jobs: int):
    from aeon.classification.convolution_based import MiniRocketClassifier
    return MiniRocketClassifier(random_state=seed, n_jobs=n_jobs)


def main():
    require_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ucr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    info = env_info("cpu")
    log = CsvLog(RESULTS_DIR / "minirocket.csv", ["dataset", "seed"])
    for name in resolve(args.datasets):
        if all(log.done(dataset=name, seed=s) for s in args.seeds):
            continue
        try:
            data = load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue
        for seed in args.seeds:
            if log.done(dataset=name, seed=seed):
                continue
            try:
                t0 = time.perf_counter()
                clf = make_minirocket(seed, args.n_jobs).fit(data["X_train"], data["y_train"])
                acc = float((clf.predict(data["X_test"]) == data["y_test"]).mean())
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"[fail] {name} seed {seed}: {e}")
                continue
            log.write({"dataset": name, "seed": seed, "classifier": "MiniRocket",
                       "accuracy": acc, "seconds": round(time.perf_counter() - t0, 2),
                       "input": "resized_512", "aeon": info["aeon"], "hardware": info["hardware"]})
            print(f"{name:32s} seed {seed}  acc {acc:.4f}")


if __name__ == "__main__":
    main()
