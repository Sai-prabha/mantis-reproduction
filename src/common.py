"""Shared helpers: device choice, seeding, environment info, crash-safe CSV logging."""
import os

# Must be set before torch is imported: lets unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import csv
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MANTIS_DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("MANTIS_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("MANTIS_RESULTS_DIR", ROOT / "results"))

CHECKPOINT = "paris-noah/Mantis-8M"
# Pinned Hugging Face commit of paris-noah/Mantis-8M (resolved 15 Oct 2026), so
# every group member and every rerun uses byte-identical weights.
CHECKPOINT_REVISION = os.environ.get("MANTIS_REVISION", "bc7d5ab40c02133386a28e2c127f35c17c86901d")


def require_env() -> None:
    """Stop if the scripts are run outside the project's Python 3.11 environment.
    Mixing environments (e.g. conda base) silently changes package versions
    between result rows. Override with MANTIS_ALLOW_ANY_ENV=1."""
    import sys
    if sys.version_info[:2] != (3, 11) and not os.environ.get("MANTIS_ALLOW_ANY_ENV"):
        sys.exit(f"Wrong Python environment: Python {platform.python_version()} at {sys.executable}.\n"
                 "Run `conda activate mantis` (Python 3.11) and try again. "
                 "Set MANTIS_ALLOW_ANY_ENV=1 only if you really mean to override.")


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def sync(device: str) -> None:
    """Wait for queued GPU work so that it is included in the timing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def hardware_string() -> str:
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return f"{platform.system()} {platform.release()} | {chip}"


def env_info(device: str) -> dict:
    import sklearn
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "device": device,
        "hardware": hardware_string(),
    }
    for pkg in ("mantis", "aeon"):
        try:
            from importlib.metadata import version
            info[pkg] = version("mantis-tsfm" if pkg == "mantis" else pkg)
        except Exception:
            info[pkg] = "not installed"
    return info


class CsvLog:
    """Append-only CSV log. Rows are flushed immediately, so an interrupted
    overnight run loses at most the row in progress. `done()` lets scripts
    skip work that is already recorded, so a rerun resumes where it stopped."""

    def __init__(self, path: Path, key_fields: list[str]):
        self.path = Path(path)
        self.key_fields = key_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = set()
        self._fields = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._fields = reader.fieldnames
                for row in reader:
                    self._keys.add(self._key(row))

    def _key(self, row: dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in self.key_fields)

    def done(self, **key) -> bool:
        return self._key(key) in self._keys

    def write(self, row: dict) -> None:
        new_file = self._fields is None
        if new_file:
            self._fields = list(row.keys())
        extra = [k for k in row if k not in self._fields]
        if extra:
            raise ValueError(f"{self.path.name}: unexpected columns {extra}; "
                             "delete or rename the old CSV if the schema changed")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fields})
        self._keys.add(self._key(row))


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
