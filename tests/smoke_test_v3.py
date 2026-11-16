import os
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="mantis_smoke_v3_"))
os.environ["MANTIS_RESULTS_DIR"] = str(tmp / "results")
os.environ["MANTIS_CACHE_DIR"] = str(tmp / "cache")
os.environ["MANTIS_DATA_DIR"] = str(tmp / "data")
os.environ["MANTIS_ALLOW_ANY_ENV"] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import src.data  # noqa: E402
import src.models  # noqa: E402

rng = np.random.default_rng(0)


def fake_raw(name, split):
    n = {"train": 40, "test": 30}[split]
    y = rng.integers(0, 3, n)
    if name == "Long":
        X = rng.standard_normal((n, 1, 1200)) + y[:, None, None]
    elif name == "Wide":
        X = rng.standard_normal((n, 12, 100)) + y[:, None, None]
    elif name == "Unequal":
        X = [rng.standard_normal((1, int(rng.integers(600, 900)))) + c for c in y]
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


names = "Plain,Long,Wide,Unequal"
run("02_zero_shot.py", "--datasets", names, "--device", "cpu", "--seeds", "0")
run("07_fusion.py", "--datasets", names, "--seeds", "0", "--device", "cpu",
    "--methods", "R:ridge")
run("11_multiscale.py", "--datasets", names, "--seeds", "0", "1", "--device", "cpu", "--windows", "2")
run("11_multiscale.py", "--datasets", names, "--seeds", "0", "1", "--device", "cpu", "--windows", "2")
run("12_channel_reduce.py", "--datasets", names, "--seeds", "0", "--device", "cpu",
    "--min-channels", "4", "--k", "3")
run("13_resample.py", "--datasets", names, "--resamples", "3", "--device", "cpu")
run("13_resample.py", "--datasets", names, "--resamples", "3", "--device", "cpu", "--min-len", "512",
    "--methods", "M:RF200,M:ridge,R:ridge,M+W2mean:RF200,M+W2mean:ridge")
run("14_proba_metrics.py", "--datasets", names, "--seeds", "0", "--device", "cpu")
run("15_improve_report.py")
print(f"\nSmoke test passed. Outputs in {tmp}")
