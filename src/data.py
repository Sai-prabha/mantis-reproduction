"""Data loading and preprocessing.

Preprocessing choices (state these in Section 2 of the report):
  * Raw archive data is loaded on purpose (load_equal_length=False,
    load_no_missing=False). aeon 0.11 to 1.2 default both to True, silently
    returning the archive's pre-padded / pre-imputed copies; aeon 1.6 defaults
    to False. Setting them explicitly makes the result version-independent.
  * Missing values: linear interpolation within each channel, then edge fill.
    v1 does not describe missing-value handling, so this is our assumption.
  * Every series is resized to length 512 with linear interpolation
    (torch F.interpolate, align_corners=False), as v1 states. Unequal-length
    series are resized individually.
  * No extra normalisation: Mantis instance-normalises inside its forward pass.
"""
import inspect

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from configs.datasets import INSECT_SUBSAMPLE
from src.common import CACHE_DIR, DATA_DIR

SEQ_LEN = 512


def _load_raw(name: str, split: str):
    from aeon.datasets import load_classification
    kwargs = {"split": split, "extract_path": str(DATA_DIR)}
    params = inspect.signature(load_classification).parameters
    if "load_equal_length" in params:
        kwargs["load_equal_length"] = False
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
    cache = CACHE_DIR / "data" / f"{name}_len{length}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: (z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]) for k in z.files}

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
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{k: (np.array(v, dtype=object) if k == "meta" else v)
                                  for k, v in out.items()})
    return out
