"""Which datasets each inference experiment uses.

`study` is the default. It is the union of three groups:

* multivariate series, where the encoder never mixes channels
* long univariate and multivariate series (raw length at least 1024), where
  resizing onto 512 throws away time
* a fixed list of short univariate series, used as a check that the unchanged
  code path still matches the frozen encoder
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import reproduction_root

ROOT = reproduction_root()
STATS_PATH = ROOT / "results" / "dataset_stats_all.csv"

# Short univariate series already in the UCR-128 list. Lengths are well below
# 512, so temporal halves must not be added.
CONTROLS = [
    "Chinatown",
    "ItalyPowerDemand",
    "ECG200",
    "CBF",
    "TwoLeadECG",
    "GunPoint",
    "SwedishLeaf",
    "Wafer",
    "Adiac",
    "Coffee",
    "FiftyWords",
    "Beef",
]

METHODS = (
    "frozen_rf",
    "channel_pool",
    "cross_channel",
    "segment",
    "channel_shuffle",
)


def archive_of(name: str) -> str:
    from configs.datasets import OTHERS, ucr128, uea27

    if name in OTHERS:
        return "other"
    if name in uea27():
        return "uea"
    if name in ucr128():
        return "ucr"
    return "unknown"


def load_stats(path: Path = STATS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/00_dataset_stats.py --datasets all first."
        )
    stats = pd.read_csv(path)
    ok = stats[stats["status"].astype(str).eq("ok")].copy()
    ok["name"] = ok["name"].astype(str)
    return ok


def select(selection: str, path: Path = STATS_PATH) -> list[str]:
    """Dataset names for a CLI argument.

    Accepts study, multi, long, controls, and every selection the reproduction
    scripts already accept (ucr, uea, tier1, a comma-separated list, ...).
    """
    key = selection.strip().lower()
    if key in {"study", "multi", "long", "controls"}:
        stats = load_stats(path)
        multi = stats.loc[stats["n_channels"] > 1, "name"].tolist()
        long = stats.loc[stats["min_len"] >= 1024, "name"].tolist()
        present = set(stats["name"])
        controls = [name for name in CONTROLS if name in present]
        if key == "multi":
            return multi
        if key == "long":
            return long
        if key == "controls":
            return controls
        # Stable order: multivariate, then long series not already included, then controls.
        chosen = list(dict.fromkeys(multi + long + controls))
        return chosen

    from configs.datasets import resolve

    return resolve(selection.strip())
