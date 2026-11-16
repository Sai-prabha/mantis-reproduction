import numpy as np

from configs.datasets import INSECT_SUBSAMPLE
from src.common import CACHE_DIR, CHECKPOINT_REVISION
from src.data import SEQ_LEN, _fill_missing, _load_raw, _resize_one, _stratified_subsample

POOLS = ("mean", "concat")


def window_bounds(length: int, n_windows: int, overlap: float = 0.0) -> list[tuple[int, int]]:
    if n_windows < 1:
        raise ValueError("n_windows must be at least 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if n_windows == 1:
        return [(0, length)]
    width = int(np.ceil(length / n_windows * (1.0 + overlap)))
    width = max(2, min(width, length))
    starts = np.linspace(0, length - width, n_windows).round().astype(int)
    return [(int(s), int(s) + width) for s in starts]


def split_windows(x: np.ndarray, n_windows: int, overlap: float = 0.0,
                  length: int = SEQ_LEN) -> list[np.ndarray]:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return [_resize_one(x[:, s:e], length) for s, e in window_bounds(x.shape[-1], n_windows, overlap)]


def raw_windows(name: str, split: str, n_windows: int, overlap: float = 0.0,
                length: int = SEQ_LEN) -> list[np.ndarray]:
    X, y = _load_raw(name, split)
    if name == INSECT_SUBSAMPLE["name"]:
        key = "n_train" if split == "train" else "n_test"
        X, y = _stratified_subsample(X, y, INSECT_SUBSAMPLE[key], INSECT_SUBSAMPLE["seed"])
    series = list(X) if isinstance(X, list) else [X[i] for i in range(len(X))]
    per_window = [[] for _ in range(n_windows)]
    for s in series:
        for w, piece in enumerate(split_windows(_fill_missing(np.atleast_2d(s)), n_windows, overlap, length)):
            per_window[w].append(piece)
    return [np.stack(p).astype(np.float32) for p in per_window]


def pool_windows(Z: list[np.ndarray], how: str) -> np.ndarray:
    if how == "mean":
        return np.mean(np.stack(Z, axis=0), axis=0).astype(np.float32)
    if how == "concat":
        return np.hstack(Z).astype(np.float32)
    raise ValueError(f"unknown pooling {how!r}; choose from {POOLS}")


def cached_window_embeddings(name: str, n_windows: int, overlap: float, device: str,
                             batch_size: int = 256, net=None):
    """Embeddings of every window for train and test, cached per
    (checkpoint, n_windows, overlap). Returns (Z_train_list, Z_test_list, seconds, from_cache)."""
    from src.models import embed, load_mantis
    tag = f"mantis8m_rev-{CHECKPOINT_REVISION or 'latest'}_win{n_windows}_ov{overlap:g}"
    path = CACHE_DIR / "emb" / "windows" / tag / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        Ztr = [z[f"Z_train_w{i}"] for i in range(n_windows)]
        Zte = [z[f"Z_test_w{i}"] for i in range(n_windows)]
        return Ztr, Zte, float(z["seconds"]), True
    if net is None:
        net = load_mantis(device)
    Wtr = raw_windows(name, "train", n_windows, overlap)
    Wte = raw_windows(name, "test", n_windows, overlap)
    secs = 0.0
    Ztr, Zte = [], []
    for w in range(n_windows):
        z, t = embed(net, Wtr[w], device, batch_size)
        Ztr.append(z)
        secs += t
        z, t = embed(net, Wte[w], device, batch_size)
        Zte.append(z)
        secs += t
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"Z_train_w{i}": Ztr[i] for i in range(n_windows)}
    arrays.update({f"Z_test_w{i}": Zte[i] for i in range(n_windows)})
    np.savez(path, seconds=secs, **arrays)
    return Ztr, Zte, secs, False
