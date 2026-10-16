"""Foundation-model baseline: MOMENT-1-large, zero-shot embeddings + RF200.

Tests v1's claim that Mantis beats MOMENT by 1.53 pp (zero-shot, 159 datasets).

momentfm 0.1.4 pins numpy==1.25.2, transformers==4.33.3 and huggingface-hub==0.24.0,
which clash with the main environment, so this script runs in its own
environment (Python 3.11). It reads the preprocessed data cache written by the main
environment, so run scripts/00_dataset_stats.py --datasets all there first.

Choices described in the report:
  * Same preprocessed input as Mantis (length 512, which MOMENT also expects).
  * Channels embedded one at a time and concatenated, like Mantis. MOMENT's
    own "mean" reduction would average over channels instead.
  * Embedding = mean over patches of the final encoder output (momentfm default).

Usage (from the main environment's shell)
  conda run --no-capture-output -n mantis-moment python scripts/05_moment.py --datasets ucr
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from configs.datasets import resolve  # noqa: E402
from src.common import CACHE_DIR, RESULTS_DIR, CsvLog, env_info, pick_device, require_env  # noqa: E402
from src.data import load_dataset  # noqa: E402

MODEL_ID = "AutonLab/MOMENT-1-large"


def load_moment(device):
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(MODEL_ID, model_kwargs={"task_name": "embedding"})
    model.init()
    return model.to(device).eval()


@torch.no_grad()
def embed_moment(model, X: np.ndarray, device: str, bs: int = 64) -> np.ndarray:
    n, c, L = X.shape
    flat = X.reshape(n * c, 1, L)  # one channel at a time
    out = []
    for i in range(0, len(flat), bs):
        xb = torch.tensor(flat[i:i + bs], dtype=torch.float32, device=device)
        out.append(model(x_enc=xb).embeddings.float().cpu())
    Z = torch.cat(out).numpy()
    return Z.reshape(n, c * Z.shape[-1])


def main():
    require_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ucr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    info = env_info(device)
    log = CsvLog(RESULTS_DIR / "moment.csv", ["dataset", "seed"])
    model = None
    for name in resolve(args.datasets):
        if all(log.done(dataset=name, seed=s) for s in args.seeds):
            continue
        cache = CACHE_DIR / "emb" / "moment1large" / f"{name}.npz"
        try:
            data = load_dataset(name)  # served from the cache
            if cache.exists():
                z = np.load(cache)
                Ztr, Zte, secs = z["Z_train"], z["Z_test"], float(z["seconds"])
            else:
                if model is None:
                    model = load_moment(device)
                t0 = time.perf_counter()
                Ztr = embed_moment(model, data["X_train"], device, args.batch_size)
                Zte = embed_moment(model, data["X_test"], device, args.batch_size)
                secs = time.perf_counter() - t0
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, Z_train=Ztr, Z_test=Zte, seconds=secs)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print(f"[skip] {name}: {e}")
            continue
        for seed in args.seeds:
            if log.done(dataset=name, seed=seed):
                continue
            rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
            rf.fit(Ztr, data["y_train"])
            acc = float((rf.predict(Zte) == data["y_test"]).mean())
            log.write({"dataset": name, "seed": seed, "model": MODEL_ID, "classifier": "RF200",
                       "accuracy": acc, "embed_seconds": round(secs, 2), "device": device,
                       "torch": info["torch"], "hardware": info["hardware"]})
            print(f"{name:32s} seed {seed}  acc {acc:.4f}")


if __name__ == "__main__":
    main()
