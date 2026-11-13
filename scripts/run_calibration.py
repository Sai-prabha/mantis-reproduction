"""Calibration of frozen Mantis-8M probabilities.

The reproduction measures ECE after fine-tuning. This script keeps the encoder
frozen and calibrates the random forest instead. The forest is fit on 80% of
the official training split. Temperature scaling and isotonic regression (IRM)
are fit on the remaining 20%, then scored on the official test split.

Temperature scaling does not change the predicted class. IRM can. The quantity
under test is ECE, not accuracy.

  python scripts/run_calibration.py --datasets study
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inference_paths", Path(__file__).resolve().parents[1] / "src" / "paths.py"
)
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)
PROJECT, _ROOT = _paths.prepare()

from src.common import CHECKPOINT_REVISION, CsvLog, pick_device, require_env  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from embeddings import Encoder  # noqa: E402
from study import archive_of, select  # noqa: E402

OUT_DIR = PROJECT / "results"
CSV_PATH = OUT_DIR / "calibration.csv"
KEY = ["dataset", "seed"]


def split_train_val(y: np.ndarray, seed: int, frac: float = 0.2):
    """Same 80/20 rule as scripts/03_finetune.py, so the split is comparable."""
    index = np.arange(len(y))
    n_val = max(int(round(frac * len(y))), 1)
    try:
        train, val = train_test_split(index, test_size=n_val, stratify=y, random_state=seed)
        stratified = 1
    except ValueError:
        train, val = train_test_split(index, test_size=n_val, random_state=seed)
        stratified = 0
    return np.sort(train), np.sort(val), stratified


def forest_logits(forest, features: np.ndarray, n_classes: int) -> np.ndarray:
    """Log probabilities, with a column for every class the dataset contains."""
    probabilities = np.clip(forest.predict_proba(features), 1e-12, 1.0)
    logits = np.full((len(features), n_classes), np.log(1e-12), dtype=np.float64)
    logits[:, forest.classes_] = np.log(probabilities)
    return logits


def main():
    require_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from src.calibration import IRM, calibration_report, softmax

    device = pick_device(args.device)
    names = select(args.datasets)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = CsvLog(CSV_PATH, KEY)
    failures = CsvLog(OUT_DIR / "calibration_failures.csv", ["dataset"])
    encoder = Encoder(device, args.batch_size)
    print(f"device {device}  datasets {len(names)}")

    for name in names:
        if all(log.done(dataset=name, seed=seed) for seed in args.seeds):
            continue
        try:
            from src.data import load_dataset

            data = load_dataset(name)
            Z_train, Z_test, _seconds, from_cache = encoder.base(name, data)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            if not failures.done(dataset=name):
                failures.write({"dataset": name, "error": f"{type(exc).__name__}: {exc}"[:500]})
            continue

        meta = data["meta"]
        n_classes = int(meta["n_classes"])
        y_test = data["y_test"]
        print(f"{name:32s} embed {'cache' if from_cache else 'computed'}")
        for seed in args.seeds:
            if log.done(dataset=name, seed=seed):
                continue
            train_index, val_index, stratified = split_train_val(data["y_train"], seed)
            forest = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=seed, n_jobs=1
            )
            forest.fit(Z_train[train_index], data["y_train"][train_index])
            logits_val = forest_logits(forest, Z_train[val_index], n_classes)
            logits_test = forest_logits(forest, Z_test, n_classes)
            y_val = data["y_train"][val_index]
            report = calibration_report(logits_val, y_val, logits_test, y_test)
            temperature = report["temperature"]
            irm_proba = IRM().fit(softmax(logits_val), y_val).transform(softmax(logits_test))
            row = {
                "archive": archive_of(name),
                "dataset": name,
                "seed": seed,
                "n_channels": int(meta["n_channels"]),
                "min_len": int(meta["min_len"]),
                "n_train_fit": int(len(train_index)),
                "n_val": int(len(val_index)),
                "n_test": int(len(y_test)),
                "n_classes": n_classes,
                "stratified_val": stratified,
                "acc_raw": float((logits_test.argmax(1) == y_test).mean()),
                "acc_ts": float(((logits_test / temperature).argmax(1) == y_test).mean()),
                "acc_irm": float((irm_proba.argmax(1) == y_test).mean()),
                **report,
                "device": device,
                "checkpoint_revision": CHECKPOINT_REVISION or "latest",
            }
            log.write(row)
            print(f"{name:32s} seed {seed}  ece {report['ece_raw']:.4f} -> "
                  f"ts {report['ece_ts']:.4f}  irm {report['ece_irm']:.4f}")


if __name__ == "__main__":
    main()
