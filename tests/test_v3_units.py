import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.channels import (apply_pca_channels, fit_pca_channels, magnitude_channel,  # noqa: E402
                          random_channel_indices, reduce_channels, summary_channels)
from src.stats_utils import comparison_table, holm, mean_ranks  # noqa: E402
from src.windows import pool_windows, split_windows, window_bounds  # noqa: E402

rng = np.random.default_rng(0)


def test_window_bounds():
    assert window_bounds(100, 1) == [(0, 100)]
    assert window_bounds(100, 2) == [(0, 50), (50, 100)]
    assert window_bounds(100, 4) == [(0, 25), (25, 50), (50, 75), (75, 100)]
    b = window_bounds(100, 2, overlap=0.5)
    assert b[0][0] == 0 and b[-1][1] == 100 and b[0][1] - b[0][0] == 75
    b = window_bounds(7, 4)
    assert len(b) == 4 and all(e - s >= 2 for s, e in b)


def test_split_windows_shapes():
    x = rng.standard_normal((3, 1000))
    parts = split_windows(x, 4, length=512)
    assert len(parts) == 4 and all(p.shape == (3, 512) for p in parts)
    step = np.zeros((1, 1000))
    step[:, 500:] = 1.0
    parts = split_windows(step, 4, length=512)
    assert parts[0].max() == 0.0 and parts[3].min() == 1.0


def test_pool_windows():
    Z = [np.full((5, 4), i, dtype=np.float32) for i in range(3)]
    assert np.allclose(pool_windows(Z, "mean"), 1.0)
    assert pool_windows(Z, "concat").shape == (5, 12)


def test_channels():
    Xtr = rng.standard_normal((20, 10, 50))
    Xte = rng.standard_normal((8, 10, 50))
    pipe = fit_pca_channels(Xtr, 3)
    assert apply_pca_channels(pipe, Xte).shape == (8, 3, 50)
    assert fit_pca_channels(Xtr, 99).named_steps["pca"].n_components_ == 10
    idx = random_channel_indices(10, 4, seed=1)
    assert len(idx) == 4 and len(set(idx)) == 4 and np.all(np.diff(idx) > 0)
    assert summary_channels(Xte).shape == (8, 3, 50)
    assert magnitude_channel(Xte).shape == (8, 1, 50)
    one = np.ones((2, 4, 5))
    assert np.allclose(magnitude_channel(one), 2.0)
    for m in ("pca", "random", "summary", "magnitude"):
        a, b = reduce_channels(m, Xtr, Xte, 3)
        assert a.shape[0] == 20 and b.shape[0] == 8 and a.shape[1:] == b.shape[1:]
    scaler = pipe.named_steps["standardscaler"]
    assert np.allclose(scaler.mean_, Xtr.transpose(0, 2, 1).reshape(-1, 10).mean(0))


def test_stats():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    acc = pd.DataFrame({"A": [0.9, 0.8, 0.7, 0.95, 0.85], "B": [0.8, 0.7, 0.6, 0.9, 0.8],
                        "C": [0.9, 0.6, 0.7, 0.8, 0.7]}, index=list("abcde"))
    r = mean_ranks(acc)
    assert r["A"] < r["B"] and r["A"] < r["C"]
    t = comparison_table(acc, [("A", "B", "q1"), ("A", "C", "q2")])
    assert list(t.wins) == [5, 3] and list(t.ties) == [0, 2] and t.loc[0, "n"] == 5
    assert float(t.loc[0, "p_holm"]) >= float(t.loc[0, "p"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all unit checks passed")
