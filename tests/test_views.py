"""Checks that do not need the checkpoint or a dataset download.

  python tests/test_views.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "inference_paths", Path(__file__).resolve().parents[1] / "src" / "paths.py"
)
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)
_paths.prepare()

from stats import format_p, format_pp, markdown_table, paired_summary  # noqa: E402
from study import select  # noqa: E402
from views import channel_pool, cross_channel_views, shuffle_channels, split_series  # noqa: E402


def test_cross_channel_tracks_the_shared_signal():
    rng = np.random.default_rng(0)
    length = 40
    signal = np.linspace(0.0, 1.0, length, dtype=np.float64)
    series = np.zeros((2, 3, length), dtype=np.float64)
    series[:, 1, :] = signal
    series[:, 2, :] = 0.5 * signal
    series[:, 0, :] = 0.01 * rng.normal(size=(2, length))
    views = cross_channel_views(series.astype(np.float32))
    assert views.shape == (2, 3, length)
    assert np.allclose(views[:, 0], series.mean(axis=1), atol=1e-5)
    assert np.allclose(views[:, 1], series.std(axis=1), atol=1e-5)
    correlation = np.corrcoef(views[0, 2], signal)[0, 1]
    assert correlation > 0.99
    again = cross_channel_views(series.astype(np.float32))
    assert np.allclose(views, again)


def test_cross_channel_rejects_univariate():
    try:
        cross_channel_views(np.zeros((4, 1, 16), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("univariate input should be rejected")


def test_pool_is_the_identity_for_one_channel():
    embeddings = np.arange(12, dtype=float).reshape(3, 4)
    assert np.allclose(channel_pool(embeddings, 1), embeddings)


def test_pool_averages_channel_blocks():
    embeddings = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 2.0, 2.0]])
    pooled = channel_pool(embeddings, 2)
    assert np.allclose(pooled[0], [2.0, 3.0])
    assert np.allclose(pooled[1], [1.0, 1.0])


def test_shuffle_keeps_the_same_numbers_in_a_new_order():
    embeddings = np.arange(20, dtype=float).reshape(2, 10)
    shuffled = shuffle_channels(embeddings, 2, seed=0)
    assert shuffled.shape == embeddings.shape
    assert not np.allclose(shuffled, embeddings)
    for row in range(2):
        assert np.allclose(np.sort(shuffled[row]), np.sort(embeddings[row]))


def test_split_series_lengths():
    even_left, even_right = split_series(np.zeros((3, 10)))
    odd_left, odd_right = split_series(np.zeros((3, 9)))
    assert even_left.shape == (3, 5) and even_right.shape == (3, 5)
    assert odd_left.shape == (3, 4) and odd_right.shape == (3, 5)


def test_paired_summary_detects_a_consistent_gap():
    baseline = np.linspace(0.4, 0.8, 15)
    other = baseline + 0.02
    summary = paired_summary(baseline, other)
    assert summary["wins"] == 15 and summary["ties"] == 0 and summary["losses"] == 0
    assert summary["p"] < 0.05
    assert format_pp(summary["mean_delta"]) == "+2.00 pp"
    ties = paired_summary(baseline, baseline.copy())
    assert ties["note"] == "all ties" and ties["p"] is None
    assert format_p(0.0004) == "<0.001"
    table = markdown_table(["Method", "Δ"], [["pool", "+1.00 pp"]])
    assert table.splitlines()[0].startswith("| Method")


def test_study_lists_cover_both_kinds_of_series():
    multi = select("multi")
    long = select("long")
    controls = select("controls")
    study = select("study")
    assert "Epilepsy" in multi and "ECG200" not in multi
    assert "HandOutlines" in long and "Chinatown" not in long
    assert controls[0] == "Chinatown" and "ECG200" in controls
    assert set(multi).issubset(study) and set(controls).issubset(study)
    assert select("ECG200") == ["ECG200"]


def test_headline_table_on_a_tiny_frame():
    import pandas as pd

    from evaluate import _headline, _narrative, _seed_means

    rows = []
    specs = [
        ("Short", 1, 24, 50, "frozen_rf", 0.80, 0, ""),
        ("Short", 1, 24, 50, "channel_pool", 0.80, 1, ""),
        ("Short", 1, 24, 50, "segment", 0.80, 0, ""),
        ("Long", 3, 2000, 40, "frozen_rf", 0.70, 1, ""),
        ("Long", 3, 2000, 40, "channel_pool", 0.60, 1, ""),
        ("Long", 3, 2000, 40, "cross_channel", 0.75, 1, ""),
        ("Long", 3, 2000, 40, "segment", 0.72, 1, ""),
        ("Long", 3, 2000, 40, "channel_shuffle", 0.40, 1, 0.70),
    ]
    for dataset, channels, length, n_train, method, accuracy, applied, unshuffled in specs:
        for seed in (0, 1):
            rows.append({
                "dataset": dataset, "method": method, "seed": seed, "accuracy": accuracy,
                "ece": 0.1, "accuracy_unshuffled": unshuffled, "extra_embed_seconds": 1.0,
                "n_channels": channels, "min_len": length, "max_len": length,
                "n_train": n_train, "n_test": 30, "applied": applied, "feature_dim": 256,
                "archive": "ucr", "note": "",
            })
    means = _seed_means(pd.DataFrame(rows))
    text = _headline(means)
    assert "Channel pool, univariate" in text
    assert "Cross-channel views, multivariate" in text
    assert "Temporal halves, raw length at least 1024" in text
    narrative = _narrative(means)
    assert "univariate" in narrative and "multivariate" in narrative


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
