"""Zero-shot (frozen encoder) reproduction of Mantis-8M, v1 Figure 3 / Appendix Table 3.

Protocol (v1): last transformer layer, CLS token (256-d per channel; channels
embedded independently and concatenated), random forest with 200 trees and
unlimited depth, no hyperparameter search, 3 seeds, official train/test splits.

The --layer / --token options exist so the Section 4 work (intermediate
layers, combined token) can reuse this script; leave the defaults for Section 3.

Usage
  python scripts/02_zero_shot.py --datasets ucr
  python scripts/02_zero_shot.py --datasets uea --batch-size 1024
  python scripts/02_zero_shot.py --datasets others
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

from configs.datasets import OTHERS, resolve  # noqa: E402
from src.calibration import ece  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.models import cached_embeddings, load_mantis  # noqa: E402


def archive_of(name: str) -> str:
    from configs.datasets import ucr128, uea27
    if name in OTHERS:
        return "other"
    if name in uea27():
        return "uea"
    if name in ucr128():
        return "ucr"
    return "unknown"


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="ucr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--token", default="cls_token", choices=["cls_token", "mean_token", "combined"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(RESULTS_DIR / "zero_shot.csv"))
    args = ap.parse_args()

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(Path(args.out), ["dataset", "seed", "layer", "token"])
    failures = CsvLog(RESULTS_DIR / "zero_shot_failures.csv", ["dataset", "layer", "token"])
    net = None

    for name in resolve(args.datasets):
        if all(log.done(dataset=name, seed=s, layer=args.layer, token=args.token) for s in args.seeds):
            continue
        try:
            data = load_dataset(name)
            if net is None:
                net = load_mantis(device, layer=args.layer, token=args.token)
            Ztr, Zte, emb_secs, from_cache = cached_embeddings(
                name, data, device, args.layer, args.token, args.batch_size, net=net)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            if not failures.done(dataset=name, layer=args.layer, token=args.token):
                failures.write({"dataset": name, "layer": args.layer, "token": args.token,
                                "error": f"{type(e).__name__}: {e}"[:500]})
            continue

        meta = data["meta"]
        for seed in args.seeds:
            if log.done(dataset=name, seed=seed, layer=args.layer, token=args.token):
                continue
            t0 = time.perf_counter()
            rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
            rf.fit(Ztr, data["y_train"])
            proba = np.zeros((len(Zte), meta["n_classes"]))
            proba[:, rf.classes_] = rf.predict_proba(Zte)
            acc = float((proba.argmax(1) == data["y_test"]).mean())
            log.write({
                "archive": archive_of(name), "dataset": name, "seed": seed,
                "layer": args.layer, "token": args.token, "classifier": "RF200",
                "accuracy": acc,
                "ece_rf": ece(proba, data["y_test"]),  # not a v1 claim; logged for Section 4
                "n_train": meta["n_train"], "n_test": meta["n_test"],
                "n_classes": meta["n_classes"], "n_channels": meta["n_channels"],
                "min_len": meta["min_len"], "max_len": meta["max_len"],
                "unequal_length": meta["unequal_length"], "n_missing_values": meta["n_missing_values"],
                "embed_seconds": round(emb_secs, 2), "embeddings_from_cache": int(from_cache),
                "rf_seconds": round(time.perf_counter() - t0, 2),
                "device": device, "torch": info["torch"], "mantis": info["mantis"],
                "hardware": info["hardware"], "checkpoint_revision": CHECKPOINT_REVISION or "latest",
            })
            print(f"{name:32s} seed {seed}  acc {acc:.4f}")


if __name__ == "__main__":
    main()
