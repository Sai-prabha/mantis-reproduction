"""Find this project and the reproduction checkout without shadowing its packages."""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def reproduction_root() -> Path:
    """Checkout that contains the published scripts and src.common.

    Locally this project sits inside that checkout. On its own branch the
    published files are not present, so the project root is returned.
    """
    project = project_root()
    for parent in [project, *project.parents]:
        if (parent / "scripts" / "02_zero_shot.py").exists() and (parent / "src" / "common.py").exists():
            return parent
    return project


def prepare() -> tuple[Path, Path]:
    """Put the reproduction first on sys.path, then this project's modules."""
    project = project_root()
    repo = reproduction_root()
    entries = [repo, project / "scripts", project / "src", project / "config"]
    for entry in reversed(entries):
        text = str(entry)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return project, repo
