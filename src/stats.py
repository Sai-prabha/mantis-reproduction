"""Paired comparison helpers.

Datasets are the unit of comparison, as in the week-7 evaluation lab.
Mean accuracy is reported for orientation. Wins, ties, losses, the median
gap, and a two-sided Wilcoxon signed-rank test carry the conclusion.
The direction is not fixed after looking at the numbers, so the test is
two-sided. Zero gaps are dropped before ranking (zero_method="wilcox").
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

TOL = 1e-6


def win_tie_loss(baseline: np.ndarray, other: np.ndarray, tol: float = TOL) -> tuple[int, int, int]:
    """Wins, ties and losses of `other` against `baseline`."""
    gap = np.asarray(other, dtype=float) - np.asarray(baseline, dtype=float)
    wins = int(np.sum(gap > tol))
    losses = int(np.sum(gap < -tol))
    ties = int(gap.size - wins - losses)
    return wins, ties, losses


def paired_summary(baseline: np.ndarray, other: np.ndarray, tol: float = TOL) -> dict:
    """Effect size and a two-sided Wilcoxon p-value on paired dataset scores."""
    baseline = np.asarray(baseline, dtype=float)
    other = np.asarray(other, dtype=float)
    if baseline.shape != other.shape:
        raise ValueError(f"shape mismatch: {baseline.shape} vs {other.shape}")
    gap = other - baseline
    wins, ties, losses = win_tie_loss(baseline, other, tol)
    informative = np.abs(gap) > tol
    p_value = None
    note = "all ties" if not np.any(informative) else ""
    if np.any(informative):
        try:
            p_value = float(
                wilcoxon(
                    other[informative],
                    baseline[informative],
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )
        except ValueError:
            note = "wilcoxon undefined"
    return {
        "n": int(gap.size),
        "n_informative": int(np.sum(informative)),
        "mean_baseline": float(np.mean(baseline)) if gap.size else float("nan"),
        "mean_other": float(np.mean(other)) if gap.size else float("nan"),
        "mean_delta": float(np.mean(gap)) if gap.size else float("nan"),
        "median_delta": float(np.median(gap)) if gap.size else float("nan"),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p": p_value,
        "note": note,
    }


def format_p(p_value) -> str:
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def format_pp(delta: float) -> str:
    """Percentage points, with an explicit sign."""
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return ""
    return f"{100.0 * delta:+.2f} pp"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """GitHub-style table. Numeric-looking columns are right aligned."""
    body = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def looks_numeric(column: int) -> bool:
        values = [row[column] for row in body if row[column] != ""]
        if not values:
            return False
        cleaned = [value.replace("+", "").replace(" pp", "").replace("<", "") for value in values]
        return all(_is_number(value) for value in cleaned)

    def line(cells: list[str], fill: str | None = None) -> str:
        padded = []
        for i, cell in enumerate(cells):
            if fill is None:
                text = cell.rjust(widths[i]) if looks_numeric(i) else cell.ljust(widths[i])
            else:
                text = fill * widths[i]
            padded.append(text)
        return "| " + " | ".join(padded) + " |"

    align = []
    for i in range(len(headers)):
        if looks_numeric(i):
            align.append("-" * (widths[i] - 1) + ":")
        else:
            align.append("-" * widths[i])
    lines = [line(headers), "| " + " | ".join(align) + " |", *[line(row) for row in body]]
    return "\n".join(lines)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
