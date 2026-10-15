"""Download, preprocess and describe datasets. Writes results/dataset_stats.csv.

Run this first: it caches every dataset, shows which ones fail to load (for
example the four "other" datasets), and gives the real train size / length /
class counts used to justify the tier-1 selection and estimate run time.

Usage
  python scripts/00_dataset_stats.py --datasets tier1
  python scripts/00_dataset_stats.py --datasets all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import RESULTS_DIR, require_env  # noqa: E402

import pandas as pd  # noqa: E402

from configs.datasets import resolve  # noqa: E402
from src.data import load_dataset  # noqa: E402


def main():
    require_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="tier1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else RESULTS_DIR / (
        "dataset_stats.csv" if args.datasets == "tier1" else f"dataset_stats_{args.datasets}.csv")

    rows = []
    for name in resolve(args.datasets):
        try:
            meta = load_dataset(name)["meta"]
            rows.append({**meta, "status": "ok"})
            print(f"ok    {name:32s} train {meta['n_train']:6d}  test {meta['n_test']:6d}  "
                  f"len {meta['min_len']}-{meta['max_len']}  classes {meta['n_classes']:3d}  "
                  f"channels {meta['n_channels']:4d}  missing {meta['n_missing_values']}")
        except Exception as e:  # noqa: BLE001
            rows.append({"name": name, "status": f"failed: {type(e).__name__}: {e}"[:300]})
            print(f"FAIL  {name}: {e}")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{(df.status == 'ok').sum()} of {len(df)} loaded; saved {out}")


if __name__ == "__main__":
    main()
