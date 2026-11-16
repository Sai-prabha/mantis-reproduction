"""Probability metrics: accuracy, balanced accuracy, macro F1, AUROC, log loss, ECE.

Ridge gives no probabilities, so the heads are RF200 (on M) and logistic
regression (scaler, block weights, C = 1; on M, R and R+M).
Writes results/proba.csv. Report: scripts/15_report.py.

Usage
  python scripts/14_proba_metrics.py --datasets all
  python scripts/14_proba_metrics.py --datasets ucr --methods M:RF200,M:logreg
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
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,  # noqa: E402
                             roc_auc_score)

from configs.datasets import archive_of, resolve  # noqa: E402
from src.calibration import ece  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import make_logreg, make_minirocket_transform  # noqa: E402
from src.models import cached_embeddings  # noqa: E402

METHODS = [("M", "RF200"), ("M", "logreg"), ("R", "logreg"), ("R+M", "logreg")]


def parse_methods(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return list(METHODS)
    out = []
    for item in text.split(","):
        feats, clf = item.strip().split(":")
        if (feats, clf) not in METHODS:
            raise SystemExit(f"unknown method {item!r}; choose from "
                             + ", ".join(f"{f}:{c}" for f, c in METHODS))
        out.append((feats, clf))
    return out


def full_proba(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.zeros((len(X), n_classes), dtype=np.float64)
    p[:, np.asarray(clf.classes_, dtype=int)] = clf.predict_proba(X)
    # float32 rows miss 1 by ~1e-7, which log_loss flags; renormalise in float64
    return p / p.sum(axis=1, keepdims=True)


def proba_metrics(proba: np.ndarray, y: np.ndarray) -> dict:
    n_classes = proba.shape[1]
    pred = proba.argmax(1)
    out = {"accuracy": float((pred == y).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "f1_macro": float(f1_score(y, pred, average="macro")),
           "log_loss": float(log_loss(y, proba, labels=list(range(n_classes)))),
           "ece": float(ece(proba, y))}
    try:
        if n_classes == 2:
            out["auroc"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            out["auroc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                               labels=list(range(n_classes))))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", default=None)
    ap.add_argument("--weighting", default="equal", choices=["equal", "none"])
    ap.add_argument("--device", default="auto", help="only used if an embedding is not cached yet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    methods = parse_methods(args.methods)

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(RESULTS_DIR / "proba.csv", ["dataset", "seed", "features", "classifier"])
    failures = CsvLog(RESULTS_DIR / "proba_failures.csv", ["dataset", "seed", "features", "classifier"])

    def pending(name, seed):
        return [(f, c) for f, c in methods if not log.done(dataset=name, seed=seed, features=f, classifier=c)]

    for name in resolve(args.datasets):
        if not any(pending(name, s) for s in args.seeds):
            continue
        try:
            data = load_dataset(name)
            Mtr, Mte, _, _ = cached_embeddings(name, data, device, batch_size=args.batch_size)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failures.write({"dataset": name, "seed": "", "features": "load", "classifier": "",
                            "error": f"{type(e).__name__}: {e}"[:500]})
            continue
        meta, ytr, yte = data["meta"], data["y_train"], data["y_test"]
        n_classes = meta["n_classes"]

        for seed in args.seeds:
            todo = pending(name, seed)
            if not todo:
                continue
            blocks = {"M": (Mtr, Mte)}
            if any("R" in f.split("+") for f, _ in todo):
                try:
                    mr = make_minirocket_transform(seed, args.n_jobs).fit(data["X_train"])
                    blocks["R"] = (np.asarray(mr.transform(data["X_train"]), dtype=np.float32),
                                   np.asarray(mr.transform(data["X_test"]), dtype=np.float32))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    print(f"[fail] {name} seed {seed} MiniRocket: {e}")

            for feats, clf_name in todo:
                parts = feats.split("+")
                if any(p not in blocks for p in parts):
                    continue
                Xtr = np.hstack([blocks[p][0] for p in parts])
                Xte = np.hstack([blocks[p][1] for p in parts])
                sizes = [blocks[p][0].shape[1] for p in parts]
                try:
                    if clf_name == "logreg":
                        clf = make_logreg(sizes, args.weighting)
                    else:
                        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                                     random_state=seed, n_jobs=args.n_jobs)
                    t0 = time.perf_counter()
                    clf.fit(Xtr, ytr)
                    secs = time.perf_counter() - t0
                    metrics = proba_metrics(full_proba(clf, Xte, n_classes), yte)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    failures.write({"dataset": name, "seed": seed, "features": feats,
                                    "classifier": clf_name, "error": f"{type(e).__name__}: {e}"[:500]})
                    continue
                log.write({
                    "archive": archive_of(name), "dataset": name, "seed": seed,
                    "features": feats, "classifier": clf_name, **metrics,
                    "n_features": int(Xtr.shape[1]), "fit_seconds": round(secs, 3),
                    "n_train": meta["n_train"], "n_test": meta["n_test"],
                    "n_classes": n_classes, "n_channels": meta["n_channels"],
                    "max_len": meta["max_len"], "sklearn": info.get("sklearn"),
                    "hardware": info["hardware"], "checkpoint_revision": CHECKPOINT_REVISION,
                })
                print(f"{name:28s} seed {seed}  {feats:5s} {clf_name:6s} "
                      f"acc {metrics['accuracy']:.4f}  auroc {metrics['auroc']:.4f}  "
                      f"logloss {metrics['log_loss']:.3f}  ece {metrics['ece']:.3f}")


if __name__ == "__main__":
    main()
