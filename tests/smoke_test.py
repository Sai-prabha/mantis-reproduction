"""Offline smoke test: runs every script end to end on synthetic data with a
randomly initialised Mantis, so no internet connection is needed. It only checks
that the scripts run; the accuracies it prints are meaningless.

  python tests/smoke_test.py
"""
import os
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="mantis_smoke_"))
os.environ["MANTIS_RESULTS_DIR"] = str(tmp / "results")
os.environ["MANTIS_CACHE_DIR"] = str(tmp / "cache")
os.environ["MANTIS_DATA_DIR"] = str(tmp / "data")
os.environ["MANTIS_ALLOW_ANY_ENV"] = "1"  # the smoke test may run on any Python
os.environ["MPLBACKEND"] = "Agg"  # no GUI backend: Tk objects crash when later scripts use threads
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")  # Windows: scripts print non-ASCII (e.g. Δ)

import numpy as np  # noqa: E402

import src.data  # noqa: E402
import src.models  # noqa: E402

rng = np.random.default_rng(0)


def fake_raw(name, split):
    """Synthetic stand-ins with the quirks the real loaders must handle."""
    n = {"train": 40, "test": 30}[split]
    y = rng.integers(0, 3, n)
    if name == "MissingVals":
        X = rng.standard_normal((n, 1, 60)) + y[:, None, None]
        X[rng.random(X.shape) < 0.05] = np.nan
    elif name == "Unequal":
        X = [rng.standard_normal((1, int(rng.integers(30, 90)))) + c for c in y]
    elif name == "Multi":
        X = rng.standard_normal((n, 3, 100)) + y[:, None, None]
    else:
        X = rng.standard_normal((n, 1, 150)) + y[:, None, None]
    return X, np.array([f"class_{c}" for c in y])


_real_load = src.models.load_mantis


def random_mantis(device, pretrained=True, layer=-1, token="cls_token", seed=None):
    return _real_load(device, pretrained=False, layer=layer, token=token, seed=seed)


src.data._load_raw = fake_raw
src.models.load_mantis = random_mantis


def run(script, *args):
    print(f"\n===== {script} {' '.join(args)} =====")
    sys.argv = [script, *args]
    runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")


names = "Plain,MissingVals,Unequal,Multi"
run("00_dataset_stats.py", "--datasets", names, "--out", str(tmp / "results" / "dataset_stats.csv"))
run("01_check_and_benchmark.py")
run("02_zero_shot.py", "--datasets", names, "--device", "cpu")
run("02_zero_shot.py", "--datasets", names, "--device", "cpu")  # resume: should skip everything
run("03_finetune.py", "--datasets", "Plain,MissingVals", "--epochs", "4", "--eval-every", "2",
    "--seeds", "0", "1", "--device", "cpu")
run("03_finetune.py", "--datasets", "Plain", "--epochs", "4", "--eval-every", "2",
    "--seeds", "0", "1", "--device", "cpu")  # resume
run("04_minirocket.py", "--datasets", names, "--seeds", "0")
run("06_summarise.py")
run("07_fusion.py", "--datasets", names, "--seeds", "0", "1", "--device", "cpu")
run("09_stats_ablation.py", "--datasets", names, "--seeds", "0", "1", "--device", "cpu")
run("09_stats_ablation.py", "--datasets", names, "--seeds", "0", "1", "--device", "cpu")  # resume
run("10_dtw_baseline.py", "--datasets", names, "--max-pairs", "2000")  # Multi (3 channels) is skipped
run("10_dtw_baseline.py", "--datasets", names)  # resume: only the skipped one runs
run("08_fusion_report.py")
print(f"\nSmoke test passed. Outputs in {tmp}")
