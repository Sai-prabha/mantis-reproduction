"""Data loading and preprocessing.

Preprocessing choices (described in Section 2 of the report):
  * Raw archive data is loaded (load_equal_length=False, load_no_missing=False).
    aeon 0.11 to 1.2 default both to True and return the archive's pre-padded or
    pre-imputed copies; aeon 1.6 defaults to False. Setting them explicitly gives
    the same data whatever the aeon version.
  * Missing values: linear interpolation within each channel, then edge fill.
    v1 does not describe missing-value handling, so this is our assumption.
  * Every series is resized to length 512 with linear interpolation
    (torch F.interpolate, align_corners=False), as v1 states. Unequal-length
    series are resized individually.
  * No extra normalisation: Mantis instance-normalises inside its forward pass.

Sensitivity check (Section 3): setting AEON_EQUAL_LENGTH = True (or
`02_zero_shot.py --aeon-equal-length`) loads the archive's pre-equalised "_eq"
files instead, which is what aeon 0.11 to 1.2 returned by default. It only
changes the unequal-length datasets and uses a separate cache.
"""
import inspect
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from configs.datasets import INSECT_SUBSAMPLE
from src.common import CACHE_DIR, DATA_DIR

SEQ_LEN = 512
AEON_EQUAL_LENGTH = False  # sensitivity check only; see module docstring


def _load_raw(name: str, split: str):
    from aeon.datasets import load_classification
    kwargs = {"split": split, "extract_path": str(DATA_DIR)}
    params = inspect.signature(load_classification).parameters
    if "load_equal_length" in params:
        kwargs["load_equal_length"] = AEON_EQUAL_LENGTH
    if "load_no_missing" in params:
        kwargs["load_no_missing"] = False
    X, y = load_classification(name, **kwargs)
    return X, np.asarray(y)


def _fill_missing(x: np.ndarray) -> np.ndarray:
    """x: (channels, length). Linear interpolation of NaNs per channel."""
    x = np.array(x, dtype=np.float64)
    for c in range(x.shape[0]):
        row = x[c]
        bad = ~np.isfinite(row)
        if not bad.any():
            continue
        if bad.all():
            x[c] = 0.0
            continue
        idx = np.arange(row.size)
        x[c, bad] = np.interp(idx[bad], idx[~bad], row[~bad])
    return x


def _resize_one(x: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    t = torch.tensor(x, dtype=torch.float32)[None]  # (1, c, L)
    if t.shape[-1] == 1:  # a single point cannot be interpolated linearly
        t = t.repeat(1, 1, 2)
    return F.interpolate(t, size=length, mode="linear", align_corners=False)[0].numpy()


def preprocess(X, length: int = SEQ_LEN) -> tuple[np.ndarray, dict]:
    """Returns (n, channels, length) float32 array plus facts about the raw data."""
    series = list(X) if isinstance(X, list) else [X[i] for i in range(len(X))]
    lengths = [np.asarray(s).shape[-1] for s in series]
    n_missing = int(sum(np.sum(~np.isfinite(np.asarray(s, dtype=float))) for s in series))
    out = np.stack([_resize_one(_fill_missing(np.atleast_2d(s)), length) for s in series])
    facts = {
        "n_channels": int(out.shape[1]),
        "min_len": int(min(lengths)),
        "max_len": int(max(lengths)),
        "unequal_length": int(min(lengths) != max(lengths)),
        "n_missing_values": n_missing,
    }
    return out.astype(np.float32), facts


def _stratified_subsample(X, y, n, seed):
    from sklearn.model_selection import train_test_split
    if len(y) <= n:
        return X, y
    idx = np.arange(len(y))
    keep, _ = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    keep = np.sort(keep)
    X = [X[i] for i in keep] if isinstance(X, list) else X[keep]
    return X, y[keep]


def load_dataset(name: str, length: int = SEQ_LEN, use_cache: bool = True) -> dict:
    """Load train/test, preprocess, encode labels. Cached as .npz after the first call."""
    suffix = "_aeoneq" if AEON_EQUAL_LENGTH else ""
    cache = CACHE_DIR / "data" / f"{name}_len{length}{suffix}.npz"
    if use_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            files = set(z.files)
            if "meta_json" in files:
                out = {k: z[k] for k in ("X_train", "y_train", "X_test", "y_test")}
                out["meta"] = json.loads(str(z["meta_json"]))
                return out
        # Old cache format: 'meta' was a pickled object array. numpy 2 pickles
        # reference numpy._core, which numpy 1.x (the MOMENT env) cannot load.
        try:
            z = np.load(cache, allow_pickle=True)
            out = {k: z[k] for k in ("X_train", "y_train", "X_test", "y_test")}
            out["meta"] = z["meta"].item()
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"{cache.name} is in the old cache format, which this numpy version cannot read. "
                "Convert it once from the main environment: "
                "`conda activate mantis && python scripts/00_dataset_stats.py --datasets all`") from e
        _save_cache(cache, out)  # migrate to the portable format
        return out

    Xtr, ytr = _load_raw(name, "train")
    Xte, yte = _load_raw(name, "test")
    if name == INSECT_SUBSAMPLE["name"]:
        Xtr, ytr = _stratified_subsample(Xtr, ytr, INSECT_SUBSAMPLE["n_train"], INSECT_SUBSAMPLE["seed"])
        Xte, yte = _stratified_subsample(Xte, yte, INSECT_SUBSAMPLE["n_test"], INSECT_SUBSAMPLE["seed"])

    Xtr, facts_tr = preprocess(Xtr, length)
    Xte, facts_te = preprocess(Xte, length)
    enc = LabelEncoder().fit(np.concatenate([ytr, yte]))
    meta = {
        "name": name,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(enc.classes_)),
        "n_channels": facts_tr["n_channels"],
        "min_len": min(facts_tr["min_len"], facts_te["min_len"]),
        "max_len": max(facts_tr["max_len"], facts_te["max_len"]),
        "unequal_length": int(facts_tr["unequal_length"] or facts_te["unequal_length"]
                              or facts_tr["min_len"] != facts_te["min_len"]),
        "n_missing_values": facts_tr["n_missing_values"] + facts_te["n_missing_values"],
    }
    out = {"X_train": Xtr, "y_train": enc.transform(ytr), "X_test": Xte,
           "y_test": enc.transform(yte), "meta": meta}
    _save_cache(cache, out)
    return out


def _save_cache(cache, out: dict) -> None:
    """Portable cache: plain numeric arrays plus metadata as a JSON string (no
    pickled objects), so numpy 1.x and 2.x environments can both read it."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X_train=out["X_train"], y_train=out["y_train"],
                        X_test=out["X_test"], y_test=out["y_test"],
                        meta_json=np.array(json.dumps(out["meta"])))
