"""Inference-only views on frozen Mantis-8M.

The checkpoint and the random forest (200 trees, unlimited depth) stay as in
the reproduction. What changes is the array the forest reads.

  frozen_rf        last-layer CLS embedding, one block per channel
  channel_pool     mean of those blocks
  cross_channel    the same blocks, plus embeddings of the per-series mean,
                   standard deviation, and leading cross-channel component
  segment          the same blocks, plus embeddings of the two temporal halves
                   when every raw series is at least 1024 steps
  channel_shuffle  the forest scored after the test channels are permuted

Run from this directory, in the mantis environment:

  python scripts/run_views.py --datasets study
  python scripts/run_views.py --datasets ECG200 --methods frozen_rf,channel_pool
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inference_paths", Path(__file__).resolve().parents[1] / "src" / "paths.py"
)
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)
PROJECT, _ROOT = _paths.prepare()

from src.common import CHECKPOINT_REVISION, CsvLog, env_info, pick_device, require_env, save_json  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from embeddings import Encoder  # noqa: E402
from study import METHODS, archive_of, select  # noqa: E402
from views import (  # noqa: E402
    LONG_SERIES,
    channel_pool,
    cross_channel_views,
    shuffle_channels,
    split_series,
)

OUT_DIR = PROJECT / "results"
CSV_PATH = OUT_DIR / "inference_views.csv"
KEY = ["dataset", "method", "seed"]


def score_forest(Z_train, y_train, Z_test, y_test, seed: int, n_classes: int):
    """Fit the published forest and score the official test split."""
    from src.calibration import ece

    forest = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=seed, n_jobs=1
    )
    started = time.perf_counter()
    forest.fit(Z_train, y_train)
    proba = np.zeros((len(Z_test), n_classes), dtype=np.float64)
    proba[:, forest.classes_] = forest.predict_proba(Z_test)
    accuracy = float((proba.argmax(1) == y_test).mean())
    return accuracy, float(ece(proba, y_test)), time.perf_counter() - started


def temporal_halves(name: str):
    """Resize each temporal half to 512, using the reproduction's resampler."""
    from src.data import _fill_missing, _load_raw, _resize_one

    def one_split(split: str):
        raw, _labels = _load_raw(name, split)
        series = list(raw) if isinstance(raw, list) else [raw[i] for i in range(len(raw))]
        lefts, rights = [], []
        for item in series:
            values = _fill_missing(np.atleast_2d(np.asarray(item, dtype=np.float64)))
            left, right = split_series(values)
            lefts.append(_resize_one(left))
            rights.append(_resize_one(right))
        return np.stack(lefts).astype(np.float32), np.stack(rights).astype(np.float32)

    left_train, right_train = one_split("train")
    left_test, right_test = one_split("test")
    return left_train, right_train, left_test, right_test


def main():
    require_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="study")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}. Choose from {', '.join(METHODS)}.")
    names = select(args.datasets)
    device = pick_device(args.device)
    info = env_info(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json({"env": info, "datasets": names, "methods": methods, "seeds": args.seeds,
               "checkpoint_revision": CHECKPOINT_REVISION}, OUT_DIR / "run_meta.json")
    log = CsvLog(CSV_PATH, KEY)
    failures = CsvLog(OUT_DIR / "failures.csv", ["dataset", "method"])
    encoder = Encoder(device, args.batch_size)
    seeds = list(args.seeds)

    def still_pending(dataset: str) -> bool:
        return any(
            not log.done(dataset=dataset, method=method, seed=seed)
            for method in methods for seed in seeds
        )

    pending_names = [dataset for dataset in names if still_pending(dataset)]
    print(
        f"device {device}  datasets {len(names)}  methods {', '.join(methods)}\n"
        f"skipping {len(names) - len(pending_names)} finished datasets"
        + (f"; next is {pending_names[0]}" if pending_names else ""),
        flush=True,
    )

    for name in names:
        if name not in pending_names:
            continue
        print(f"{name}: loading", flush=True)
        try:
            from src.data import load_dataset

            data = load_dataset(name)
            Z_train, Z_test, base_seconds, from_cache = encoder.base(name, data)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _fail(failures, name, "load", exc)
            continue

        meta = data["meta"]
        n_channels = int(meta["n_channels"])
        n_classes = int(meta["n_classes"])
        y_train, y_test = data["y_train"], data["y_test"]
        context = {
            "archive": archive_of(name),
            "dataset": name,
            "n_train": int(meta["n_train"]),
            "n_test": int(meta["n_test"]),
            "n_classes": n_classes,
            "n_channels": n_channels,
            "min_len": int(meta["min_len"]),
            "max_len": int(meta["max_len"]),
            "base_dim": int(Z_train.shape[1]),
            "base_embed_seconds": round(float(base_seconds), 2),
            "embeddings_from_cache": int(from_cache),
            "device": device,
            "checkpoint_revision": CHECKPOINT_REVISION or "latest",
        }
        print(f"{name:32s} channels {n_channels:<4d} length {context['min_len']}-{context['max_len']}  "
              f"embed {'cache' if from_cache else 'computed'}", flush=True)
        pending = [method for method in methods
                   if any(not log.done(dataset=name, method=method, seed=seed) for seed in seeds)]

        if "frozen_rf" in pending:
            _score_method(
                log, failures, name, "frozen_rf", seeds, Z_train, y_train, Z_test, y_test,
                n_classes, context, applied=1, extra_seconds=0.0,
                feature_dim=context["base_dim"],
                note="published protocol: last layer, CLS token, RF200",
            )
        if "channel_pool" in pending:
            try:
                pooled_train = channel_pool(Z_train, n_channels)
                pooled_test = channel_pool(Z_test, n_channels)
            except Exception as exc:  # noqa: BLE001
                _fail(failures, name, "channel_pool", exc)
            else:
                _score_method(
                    log, failures, name, "channel_pool", seeds, pooled_train, y_train,
                    pooled_test, y_test, n_classes, context, applied=1, extra_seconds=0.0,
                    feature_dim=int(pooled_train.shape[1]),
                    note="mean of per-channel CLS blocks",
                )
        if "cross_channel" in pending:
            if n_channels < 2:
                print(f"{name:32s} cross_channel skipped (univariate)")
            else:
                _run_cross_channel(
                    encoder, log, failures, name, data, seeds, Z_train, Z_test,
                    y_train, y_test, n_classes, context,
                )
        if "segment" in pending:
            _run_segment(
                encoder, log, failures, name, seeds, Z_train, Z_test, y_train, y_test,
                n_classes, context,
            )
        if "channel_shuffle" in pending:
            if n_channels < 2:
                print(f"{name:32s} channel_shuffle skipped (univariate)")
            else:
                _run_shuffle(
                    log, failures, name, seeds, Z_train, Z_test, y_train, y_test,
                    n_classes, n_channels, context,
                )


def _run_cross_channel(encoder, log, failures, name, data, seeds, Z_train, Z_test,
                       y_train, y_test, n_classes, context):
    try:
        def build():
            views_train = cross_channel_views(data["X_train"])
            views_test = cross_channel_views(data["X_test"])
            embedded_train, seconds_train = encoder.embed(views_train)
            embedded_test, seconds_test = encoder.embed(views_test)
            return embedded_train, embedded_test, seconds_train + seconds_test

        extra_train, extra_test, seconds, _cached = encoder.extra("cross-channel", name, build)
        features_train = np.concatenate([Z_train, extra_train], axis=1)
        features_test = np.concatenate([Z_test, extra_test], axis=1)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _fail(failures, name, "cross_channel", exc)
        return
    _score_method(
        log, failures, name, "cross_channel", seeds, features_train, y_train,
        features_test, y_test, n_classes, context, applied=1, extra_seconds=seconds,
        feature_dim=int(features_train.shape[1]),
        note="CLS blocks plus mean, std, and leading component",
    )


def _run_segment(encoder, log, failures, name, seeds, Z_train, Z_test, y_train, y_test,
                 n_classes, context):
    long_enough = context["min_len"] >= LONG_SERIES
    if not long_enough:
        _score_method(
            log, failures, name, "segment", seeds, Z_train, y_train, Z_test, y_test,
            n_classes, context, applied=0, extra_seconds=0.0,
            feature_dim=context["base_dim"],
            note=f"raw length below {LONG_SERIES}; halves not added",
        )
        return
    try:
        def build():
            left_train, right_train, left_test, right_test = temporal_halves(name)
            if len(left_train) != len(y_train) or len(left_test) != len(y_test):
                raise RuntimeError(
                    f"{name}: half split has {len(left_train)}/{len(left_test)} rows, "
                    f"preprocessed data has {len(y_train)}/{len(y_test)}"
                )
            left_z_train, t1 = encoder.embed(left_train)
            right_z_train, t2 = encoder.embed(right_train)
            left_z_test, t3 = encoder.embed(left_test)
            right_z_test, t4 = encoder.embed(right_test)
            return (
                np.concatenate([left_z_train, right_z_train], axis=1),
                np.concatenate([left_z_test, right_z_test], axis=1),
                t1 + t2 + t3 + t4,
            )

        extra_train, extra_test, seconds, _cached = encoder.extra("segment", name, build)
        features_train = np.concatenate([Z_train, extra_train], axis=1)
        features_test = np.concatenate([Z_test, extra_test], axis=1)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _fail(failures, name, "segment", exc)
        return
    _score_method(
        log, failures, name, "segment", seeds, features_train, y_train, features_test,
        y_test, n_classes, context, applied=1, extra_seconds=seconds,
        feature_dim=int(features_train.shape[1]),
        note="CLS blocks plus embeddings of the two temporal halves",
    )


def _run_shuffle(log, failures, name, seeds, Z_train, Z_test, y_train, y_test,
                 n_classes, n_channels, context):
    try:
        shuffled = shuffle_channels(Z_test, n_channels, seed=0)
    except Exception as exc:  # noqa: BLE001
        _fail(failures, name, "channel_shuffle", exc)
        return
    for seed in seeds:
        if log.done(dataset=name, method="channel_shuffle", seed=seed):
            continue
        try:
            forest = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=seed, n_jobs=1
            )
            started = time.perf_counter()
            forest.fit(Z_train, y_train)
            from src.calibration import ece

            def probabilities(features):
                proba = np.zeros((len(features), n_classes), dtype=np.float64)
                proba[:, forest.classes_] = forest.predict_proba(features)
                return proba

            plain = probabilities(Z_test)
            mixed = probabilities(shuffled)
            elapsed = time.perf_counter() - started
            accuracy = float((mixed.argmax(1) == y_test).mean())
            plain_accuracy = float((plain.argmax(1) == y_test).mean())
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _fail(failures, name, "channel_shuffle", exc)
            return
        _write(
            log, context, "channel_shuffle", seed, accuracy, float(ece(mixed, y_test)),
            elapsed, applied=1, extra_seconds=0.0, feature_dim=context["base_dim"],
            note="test channel blocks permuted with seed 0; forest trained on the true order",
            accuracy_unshuffled=plain_accuracy,
        )
        print(f"{name:32s} {'channel_shuffle':16s} seed {seed}  acc {accuracy:.4f}  "
              f"unshuffled {plain_accuracy:.4f}")


def _score_method(log, failures, name, method, seeds, Z_train, y_train, Z_test, y_test,
                  n_classes, context, applied, extra_seconds, feature_dim, note):
    for seed in seeds:
        if log.done(dataset=name, method=method, seed=seed):
            continue
        try:
            accuracy, calibration, elapsed = score_forest(
                Z_train, y_train, Z_test, y_test, seed, n_classes
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _fail(failures, name, method, exc)
            return
        _write(
            log, context, method, seed, accuracy, calibration, elapsed,
            applied=applied, extra_seconds=extra_seconds, feature_dim=feature_dim, note=note,
        )
        print(f"{name:32s} {method:16s} seed {seed}  acc {accuracy:.4f}")


def _write(log, context, method, seed, accuracy, calibration, rf_seconds, applied,
           extra_seconds, feature_dim, note, accuracy_unshuffled=""):
    log.write({
        **context,
        "method": method,
        "seed": seed,
        "accuracy": accuracy,
        "ece": calibration,
        "accuracy_unshuffled": accuracy_unshuffled,
        "feature_dim": feature_dim,
        "applied": int(applied),
        "extra_embed_seconds": round(float(extra_seconds), 2),
        "rf_seconds": round(float(rf_seconds), 2),
        "note": note,
    })


def _fail(failures, name, method, exc):
    if failures.done(dataset=name, method=method):
        return
    failures.write({
        "dataset": name,
        "method": method,
        "error": f"{type(exc).__name__}: {exc}"[:500],
    })
    print(f"[fail] {name} {method}: {exc}")


if __name__ == "__main__":
    main()
