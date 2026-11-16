"""Report figure for the multi-scale experiment (results/multiscale.csv, script 11).

Paired scatter, one point per dataset: plain Mantis against Mantis with two
window embeddings, same head (ridge by default). Points above the diagonal are
datasets the windows improve. Marker colour and shape show the raw series
length, so the figure also reads in greyscale print.

Outputs: results/figures/fig_windows_scatter.pdf and .png

Usage
  python scripts/16_report_figures.py
  python scripts/16_report_figures.py --head RF200 --windowed M+W4mean
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = [("raw length 513 to 1,000", lambda L: L <= 1000, "#2a78d6", "o"),
          ("raw length above 1,000", lambda L: L > 1000, "#eb6834", "^")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", default="ridge", choices=["ridge", "RF200"])
    ap.add_argument("--windowed", default="M+W2mean")
    ap.add_argument("--labels", type=int, default=4, help="name this many datasets with the largest gaps")
    args = ap.parse_args()

    df = pd.read_csv(RESULTS_DIR / "multiscale.csv")
    df = df[df.classifier == args.head]
    acc = df.pivot_table(index="dataset", columns="features", values="accuracy", aggfunc="mean")
    length = df.groupby("dataset").max_len.first()
    x, y = acc["M"], acc[args.windowed]
    gain = y - x
    wins, ties, losses = int((gain > 0).sum()), int((gain == 0).sum()), int((gain < 0).sum())

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(3.45, 3.2))
    lo = min(x.min(), y.min()) - 0.03
    ax.plot([lo, 1.01], [lo, 1.01], color=MUTED, lw=1, zorder=1)
    for off in (0.05, -0.05):
        ax.plot([lo, 1.01], [lo + off, 1.01 + off], color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    for name, rule, color, marker in GROUPS:
        m = rule(length.reindex(acc.index))
        ax.scatter(x[m], y[m], s=26, c=color, marker=marker, edgecolors="#fcfcfb", linewidths=0.8,
                   zorder=3, label=f"{name} ({int(m.sum())})")
    picks = list(gain.sort_values().index[-(args.labels - 1):]) + [gain.idxmin()]
    placed = {"Mallat": (-14, 6, "right"), "PigArtPressure": (-14, 0, "right"),
              "EthanolLevel": (28, -38, "left"), "SelfRegulationSCP1": (-6, -40, "right")}
    for name in picks:
        dx, dy, ha = placed.get(name, (-14, 6, "right") if gain[name] > 0 else (10, -14, "left"))
        ax.annotate(f"{name} {100 * gain[name]:+.1f}", (x[name], y[name]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5, color=INK, ha=ha, va="center",
                    bbox={"boxstyle": "square,pad=0.15", "fc": "#fcfcfb", "ec": "none"},
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkB": 3})
    k = args.windowed.replace("M+W", "").replace("mean", "").replace("concat", "")
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Plain Mantis accuracy")
    ax.set_ylabel(f"Mantis + {k} windows accuracy")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_windows_scatter.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out / 'fig_windows_scatter.pdf'}  ({len(acc)} datasets, W/T/L {wins}/{ties}/{losses}, "
          f"mean gain {100 * gain.mean():+.2f} pp)")


if __name__ == "__main__":
    main()
