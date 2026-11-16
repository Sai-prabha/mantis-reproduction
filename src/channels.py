import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

METHODS = ("pca", "random", "summary", "magnitude")


def _to_table(X: np.ndarray) -> np.ndarray:
    n, C, T = X.shape
    return np.asarray(X, dtype=np.float64).transpose(0, 2, 1).reshape(n * T, C)


def _from_table(table: np.ndarray, n: int, T: int) -> np.ndarray:
    k = table.shape[1]
    return table.reshape(n, T, k).transpose(0, 2, 1).astype(np.float32)


def fit_pca_channels(X_train: np.ndarray, k: int, seed: int = 0):
    C = X_train.shape[1]
    k = min(k, C)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed))
    pipe.fit(_to_table(X_train))
    return pipe


def apply_pca_channels(pipe, X: np.ndarray) -> np.ndarray:
    n, _, T = X.shape
    return _from_table(pipe.transform(_to_table(X)), n, T)


def random_channel_indices(C: int, k: int, seed: int = 0) -> np.ndarray:
    k = min(k, C)
    return np.sort(np.random.default_rng(seed).choice(C, size=k, replace=False))


def summary_channels(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mag = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    return np.concatenate([mag, X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)],
                          axis=1).astype(np.float32)


def magnitude_channel(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.sqrt((X ** 2).sum(axis=1, keepdims=True)).astype(np.float32)


def reduce_channels(method: str, X_train: np.ndarray, X_test: np.ndarray, k: int,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if method == "pca":
        pipe = fit_pca_channels(X_train, k, seed)
        return apply_pca_channels(pipe, X_train), apply_pca_channels(pipe, X_test)
    if method == "random":
        idx = random_channel_indices(X_train.shape[1], k, seed)
        return X_train[:, idx, :], X_test[:, idx, :]
    if method == "summary":
        return summary_channels(X_train), summary_channels(X_test)
    if method == "magnitude":
        return magnitude_channel(X_train), magnitude_channel(X_test)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
