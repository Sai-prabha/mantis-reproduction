"""Tables and figures for the inference experiments.

Reads results/ here and the reproduction's results/zero_shot.csv.
Writes markdown tables and pdf/png figures. Datasets are the unit of
comparison: wins, ties, losses, median gap, and a two-sided Wilcoxon test.
Mean accuracy is reported for orientation only.

  python scripts/evaluate.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inference_paths", Path(__file__).resolve().parents[1] / "src" / "paths.py"
)
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)
PROJECT, ROOT = _paths.prepare()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stats import format_p, format_pp, markdown_table, paired_summary  # noqa: E402

OUT_DIR = PROJECT / "results"
TABLES = OUT_DIR / "tables"
FIGURES = OUT_DIR / "figures"
VIEWS_CSV = OUT_DIR / "inference_views.csv"
CAL_CSV = OUT_DIR / "calibration.csv"
BOTTLENECK_JSON = OUT_DIR / "bottleneck.json"
ZERO_SHOT_CSV = ROOT / "results" / "zero_shot.csv"

# Same blue / orange as results/figures/zs_scatter_by_type, so the report reads as one document.
UNI = "#2a78d6"
MULTI = "#eb6834"
GREEN = "#1baf7a"
INK = "#3d3d3a"
GUIDE = "#8a8980"
METHOD_LABEL = {
    "frozen_rf": "Frozen encoder + RF200",
    "channel_pool": "Channel pool + RF200",
    "cross_channel": "Cross-channel views + RF200",
    "segment": "Temporal halves + RF200",
    "channel_shuffle": "Shuffled test channels",
}
# Short names for the critical-difference diagram. The table keeps METHOD_LABEL.
CD_LABEL = {
    "frozen_rf": "Frozen encoder",
    "channel_pool": "Channel pool",
    "cross_channel": "Cross-channel",
}
SCATTER_Y = {
    "channel_pool": "Channel-pool accuracy",
    "cross_channel": "Cross-channel accuracy",
    "segment": "Temporal-halves accuracy",
}
FIGURE_ORDER = (
    "scatter_channel_pool.pdf",
    "scatter_cross_channel.pdf",
    "scatter_segment.pdf",
    "delta_vs_channels.pdf",
    "delta_vs_train_size.pdf",
    "delta_vs_length.pdf",
    "tradeoff_accuracy_vs_time.pdf",
    "cd_multivariate.pdf",
    "calibration_ece.pdf",
    "bottleneck_channels.pdf",
)
CAPTIONS = {
    "scatter_channel_pool.pdf": (
        "Channel pooling against the frozen encoder. Each point is one dataset, averaged over seeds. "
        "The solid line is equal accuracy and the dotted lines are ±5 percentage points. "
        "Univariate points lie on the line: with one channel the average leaves the embedding unchanged. "
        "Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points."
    ),
    "scatter_cross_channel.pdf": (
        "Cross-channel views against the frozen encoder, multivariate datasets only. "
        "Each point is one dataset, averaged over seeds. The solid line is equal accuracy and the "
        "dotted lines are ±5 percentage points. Points above the line are gains. "
        "Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points."
    ),
    "scatter_segment.pdf": (
        "Temporal halves against the frozen encoder, restricted to series of raw length at least 1024, "
        "where the halves are actually embedded. Each point is one dataset, averaged over seeds. "
        "The solid line is equal accuracy and the dotted lines are ±5 percentage points. "
        "Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points."
    ),
    "delta_vs_channels.pdf": (
        "Accuracy change from cross-channel views against the number of channels. "
        "The change is this method minus the frozen encoder, in percentage points, averaged over seeds. "
        "The horizontal line is no change. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points."
    ),
    "delta_vs_train_size.pdf": (
        "Accuracy change from cross-channel views against the number of training series. "
        "The change is this method minus the frozen encoder, in percentage points, averaged over seeds. "
        "The horizontal line is no change. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points."
    ),
    "delta_vs_length.pdf": (
        "Accuracy change from temporal halves against raw length, on series of length at least 1024. "
        "The change is this method minus the frozen encoder, in percentage points, averaged over seeds. "
        "The horizontal line is no change. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points."
    ),
    "tradeoff_accuracy_vs_time.pdf": (
        "Accuracy change against the extra time spent embedding the added series. "
        "The change is the method minus the frozen encoder, in percentage points, averaged over seeds. "
        "The horizontal line is no change. Cross-channel views add three series on every multivariate dataset. "
        "Temporal halves add two embeddings, and only on long series."
    ),
    "cd_multivariate.pdf": (
        "Average ranks of the frozen encoder, channel pooling, and cross-channel views on the multivariate "
        "datasets. Rank 1 is best. A bar joins methods that the Wilcoxon test with Holm correction does not "
        "separate at α = 0.05. The classifier is RF200 in every case."
    ),
    "calibration_ece.pdf": (
        "Mean expected calibration error of the frozen forest, with 10 equal-width bins, before and after "
        "post-hoc calibration. The forest is fit on 80% of the official training split. Temperature scaling "
        "and isotonic regression are fit on the remaining 20% and scored on the official test split. "
        "A lower bar is a better-calibrated probability."
    ),
    "bottleneck_channels.pdf": (
        "Forward time of frozen Mantis-8M against the number of channels. Each point is one synthetic batch "
        "of 128 series of length 512, timed on CPU. Time grows with the channel count because each channel "
        "is embedded on its own."
    ),
}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    _report_style()
    parts = ["# Inference experiments on frozen Mantis-8M", ""]
    if not VIEWS_CSV.exists():
        parts.append(
            "No inference results yet. From the repository root, with the mantis "
            "environment active:"
        )
        parts.append("")
        parts.append("```")
        parts.append("python scripts/run_views.py --datasets study")
        parts.append("python scripts/evaluate.py")
        parts.append("```")
        _write(TABLES / "headline.md", "\n".join(parts) + "\n")
        print(f"wrote {TABLES / 'headline.md'}")
        return

    raw = pd.read_csv(VIEWS_CSV)
    means = _seed_means(raw)
    parts.append(_orientation_note())
    parts.append("")
    parts.append("## Against the frozen encoder")
    parts.append("")
    parts.append(_headline(means))
    parts.append("")
    parts.append(_reproduction(means))
    parts.append("")
    parts.append("## Per dataset")
    parts.append("")
    parts.append(_per_dataset(means))
    parts.append("")
    parts.append("## Largest gaps")
    parts.append("")
    parts.append(_largest_gaps(means))
    parts.append("")
    _figures(means)
    parts.append("## Multivariate comparison")
    parts.append("")
    parts.append(_critical_difference(means))
    parts.append("")
    parts.append("## Calibration of the frozen forest")
    parts.append("")
    parts.append(_calibration())
    parts.append("")
    parts.append("## Forward-pass bottlenecks")
    parts.append("")
    parts.append(_bottleneck())
    parts.append("")
    parts.append("## Figures")
    parts.append("")
    parts.append("Captions below are for the report. The figures themselves carry the axis labels only.")
    parts.append("")
    parts.extend(_figure_list())
    parts.append("")
    parts.append("## Reading for the report")
    parts.append("")
    parts.append(_narrative(means))
    text = "\n".join(parts).rstrip() + "\n"
    _write(TABLES / "headline.md", text)
    print(f"wrote {TABLES / 'headline.md'}")


def _seed_means(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in ("accuracy", "ece", "accuracy_unshuffled", "extra_embed_seconds",
                   "n_channels", "min_len", "max_len", "n_train", "n_test", "applied", "feature_dim"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(["dataset", "method"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        ece=("ece", "mean"),
        accuracy_unshuffled=("accuracy_unshuffled", "mean"),
        extra_embed_seconds=("extra_embed_seconds", "mean"),
        n_seeds=("seed", "nunique"),
        n_channels=("n_channels", "first"),
        min_len=("min_len", "first"),
        max_len=("max_len", "first"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        applied=("applied", "max"),
        feature_dim=("feature_dim", "first"),
        archive=("archive", "first"),
        note=("note", "first"),
    )
    grouped["series"] = np.where(grouped["n_channels"] > 1, "multivariate", "univariate")
    return grouped


def _orientation_note() -> str:
    return (
        "Each number is the mean over the seeds recorded for that dataset. "
        "Mean accuracy is for orientation: accuracies on different datasets are not "
        "on a common scale (Demšar, 2006). Comparisons use wins/ties/losses, the median "
        "gap, and a two-sided Wilcoxon signed-rank test on datasets. Zero gaps are "
        "dropped before ranking. α = 0.05."
    )


def _headline(means: pd.DataFrame) -> str:
    specs = [
        ("channel_pool", "univariate", "Channel pool, univariate (reproduction check)",
         lambda frame: frame["n_channels"] <= 1),
        ("channel_pool", "multivariate", "Channel pool, multivariate",
         lambda frame: frame["n_channels"] > 1),
        ("cross_channel", "multivariate", "Cross-channel views, multivariate",
         lambda frame: frame["n_channels"] > 1),
        ("segment", "long", "Temporal halves, raw length at least 1024",
         lambda frame: frame["applied"] >= 1),
        ("segment", "control", "Temporal halves not added (short series)",
         lambda frame: frame["applied"] < 1),
    ]
    rows = []
    for method, _key, label, predicate in specs:
        summary, _merged = _against_frozen(means, method, predicate)
        if summary is None:
            continue
        rows.append([
            label,
            str(summary["n"]),
            f"{summary['mean_baseline']:.4f}",
            f"{summary['mean_other']:.4f}",
            format_pp(summary["mean_delta"]),
            format_pp(summary["median_delta"]),
            f"{summary['wins']}/{summary['ties']}/{summary['losses']}",
            format_p(summary["p"]) or summary["note"],
        ])
    if not rows:
        return "The frozen encoder rows are not in the CSV yet, so the paired table is empty."
    headers = ["Comparison", "Datasets", "Frozen", "This method", "Mean Δ", "Median Δ", "W/T/L", "Wilcoxon p"]
    return markdown_table(headers, rows)


def _against_frozen(means, method, predicate):
    frozen = means[means["method"] == "frozen_rf"][["dataset", "accuracy"]].rename(
        columns={"accuracy": "baseline"}
    )
    other = means[means["method"] == method].copy()
    if other.empty or frozen.empty:
        return None, None
    other = other[predicate(other)]
    merged = other.merge(frozen, on="dataset", how="inner")
    if merged.empty:
        return None, None
    summary = paired_summary(merged["baseline"].to_numpy(), merged["accuracy"].to_numpy())
    return summary, merged


def _reproduction(means: pd.DataFrame) -> str:
    frozen = means[means["method"] == "frozen_rf"]
    if frozen.empty or not ZERO_SHOT_CSV.exists():
        return "The reproduction CSV `results/zero_shot.csv` is not available for a side-by-side check."
    published = pd.read_csv(ZERO_SHOT_CSV)
    published = published[(published["layer"] == -1) & (published["token"] == "cls_token")]
    published = published.groupby("dataset", as_index=False)["accuracy"].mean()
    merged = frozen.merge(published, on="dataset", suffixes=("_here", "_repo"))
    if merged.empty:
        return "No shared datasets between this run and `results/zero_shot.csv`."
    summary = paired_summary(merged["accuracy_repo"].to_numpy(), merged["accuracy_here"].to_numpy())
    gap = np.max(np.abs(merged["accuracy_here"] - merged["accuracy_repo"]))
    headers = ["Check", "Datasets", "Repo zero-shot", "Refit here", "Mean Δ", "Median Δ", "W/T/L", "Wilcoxon p"]
    row = [[
        "Frozen RF refit against results/zero_shot.csv",
        str(summary["n"]),
        f"{summary['mean_baseline']:.4f}",
        f"{summary['mean_other']:.4f}",
        format_pp(summary["mean_delta"]),
        format_pp(summary["median_delta"]),
        f"{summary['wins']}/{summary['ties']}/{summary['losses']}",
        format_p(summary["p"]) or summary["note"],
    ]]
    note = (
        f"\n\nLargest absolute gap on a dataset: {gap:.4f}. "
        "The repository run used `n_jobs=-1`. This folder refits the same forest with one thread, "
        "so a gap of a few tenths of a percentage point is thread noise in the forest, not a new encoder."
    )
    return markdown_table(headers, row) + note


def _per_dataset(means: pd.DataFrame) -> str:
    meta = means.sort_values("method").groupby("dataset", as_index=False).first()
    wide = means.pivot(index="dataset", columns="method", values="accuracy")
    table = meta[["dataset", "series", "n_channels", "min_len", "n_train", "n_test"]].merge(
        wide.reset_index(), on="dataset", how="left"
    )
    table = table.sort_values(["n_channels", "min_len", "dataset"])
    headers = ["Dataset", "Series", "Channels", "Min length", "Train", "Test"]
    methods = [name for name in METHOD_LABEL if name in table.columns]
    headers.extend(METHOD_LABEL[name] for name in methods)
    rows = []
    for _, record in table.iterrows():
        row = [
            str(record["dataset"]),
            str(record["series"]),
            str(int(record["n_channels"])),
            str(int(record["min_len"])),
            str(int(record["n_train"])),
            str(int(record["n_test"])),
        ]
        for name in methods:
            value = record[name]
            row.append("" if pd.isna(value) else f"{value:.4f}")
        rows.append(row)
    _write(TABLES / "by_dataset.md", markdown_table(headers, rows) + "\n")
    return f"Accuracies for every dataset are in `tables/by_dataset.md` ({len(rows)} rows)."


def _largest_gaps(means: pd.DataFrame) -> str:
    blocks = []
    for method, title, predicate in (
        ("cross_channel", "Cross-channel views", lambda frame: frame["n_channels"] > 1),
        ("segment", "Temporal halves, where they were added", lambda frame: frame["applied"] >= 1),
        ("channel_pool", "Channel pool, multivariate", lambda frame: frame["n_channels"] > 1),
    ):
        summary, merged = _against_frozen(means, method, predicate)
        if merged is None or merged.empty:
            continue
        merged = merged.copy()
        merged["delta"] = merged["accuracy"] - merged["baseline"]
        merged["abs_delta"] = merged["delta"].abs()
        top = merged.nlargest(min(8, len(merged)), "abs_delta")
        rows = [[
            str(record.dataset),
            str(int(record.n_channels)),
            str(int(record.min_len)),
            str(int(record.n_train)),
            f"{record.baseline:.4f}",
            f"{record.accuracy:.4f}",
            format_pp(record.delta),
        ] for record in top.itertuples(index=False)]
        headers = ["Dataset", "Channels", "Min length", "Train", "Frozen", "This method", "Δ"]
        blocks.append(f"### {title}\n\n" + markdown_table(headers, rows))
        if summary is not None:
            blocks.append("")
    if not blocks:
        return "No paired gaps to list yet."
    return "\n".join(blocks)


def _report_style():
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _panel(ax):
    ax.set_facecolor("white")
    ax.grid(color="#e6e5e0", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.6, length=2.5, color="#55554f", labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    for spine in ax.spines.values():
        spine.set_color("#55554f")
        spine.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _figures(means: pd.DataFrame):
    _scatter(means, "channel_pool", lambda frame: frame["n_channels"] > 0,
             FIGURES / "scatter_channel_pool.pdf")
    _scatter(means, "cross_channel", lambda frame: frame["n_channels"] > 1,
             FIGURES / "scatter_cross_channel.pdf")
    _scatter(means, "segment", lambda frame: frame["applied"] >= 1,
             FIGURES / "scatter_segment.pdf")
    _delta_scatter(
        means, "cross_channel", lambda frame: frame["n_channels"] > 1,
        "n_channels", "Channels", True, FIGURES / "delta_vs_channels.pdf",
    )
    _delta_scatter(
        means, "cross_channel", lambda frame: frame["n_channels"] > 1,
        "n_train", "Training series", True, FIGURES / "delta_vs_train_size.pdf",
    )
    _delta_scatter(
        means, "segment", lambda frame: frame["applied"] >= 1,
        "min_len", "Raw length", True, FIGURES / "delta_vs_length.pdf",
    )
    _tradeoff(means)


def _scatter(means, method, predicate, path: Path):
    _summary, merged = _against_frozen(means, method, predicate)
    if merged is None or len(merged) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.45, 3.45))
    fig.patch.set_facecolor("white")
    present = []
    for label, color in (("Univariate", UNI), ("Multivariate", MULTI)):
        group = merged[merged["series"] == label.lower()]
        if group.empty:
            continue
        present.append(label)
        ax.scatter(
            group["baseline"], group["accuracy"], s=16, c=color, label=f"{label} ({len(group)})",
            edgecolor="white", linewidth=0.45, zorder=3,
        )
    both = np.concatenate([merged["baseline"].to_numpy(), merged["accuracy"].to_numpy()])
    lo, hi = _padded_limits(both, pad=0.04)
    ax.plot([lo, hi], [lo, hi], lw=0.8, color="#55554f", zorder=1)
    for shift in (-0.05, 0.05):
        ax.plot([lo, hi], [lo + shift, hi + shift], lw=0.6, ls=(0, (3, 3)), color="#a3a29a", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Frozen encoder accuracy")
    ax.set_ylabel(SCATTER_Y[method])
    ax.set_aspect("equal", adjustable="box")
    _panel(ax)
    if len(present) > 1:
        corner = _quiet_corner(merged["baseline"].to_numpy(), merged["accuracy"].to_numpy(), lo, hi)
        ax.legend(frameon=False, loc=corner, handletextpad=0.25, borderaxespad=0.15)
    gap = np.abs(merged["accuracy"].to_numpy() - merged["baseline"].to_numpy())
    _leader_labels(
        ax, merged["baseline"].to_numpy(), merged["accuracy"].to_numpy(),
        merged["dataset"].to_numpy(), gap, k=4, min_abs=0.05,
    )
    fig.tight_layout(pad=0.35)
    _save(fig, path)


def _delta_scatter(means, method, predicate, xcolumn, xlabel, logx, path):
    _summary, merged = _against_frozen(means, method, predicate)
    if merged is None or len(merged) < 2:
        return
    delta_pp = 100.0 * (merged["accuracy"] - merged["baseline"])
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.patch.set_facecolor("white")
    kinds = [("Univariate", "univariate", UNI), ("Multivariate", "multivariate", MULTI)]
    present = []
    for label, key, color in kinds:
        group = merged[merged["series"] == key]
        if group.empty:
            continue
        present.append((label, group))
        ax.scatter(
            group[xcolumn], 100.0 * (group["accuracy"] - group["baseline"]),
            s=16, c=color, label=f"{label} ({len(group)})",
            edgecolor="white", linewidth=0.45, zorder=3,
        )
    ax.axhline(0.0, color="#55554f", lw=0.7, zorder=1)
    if logx and (merged[xcolumn] > 0).all():
        ax.set_xscale("log")
    x_lo, x_hi = _padded_limits(merged[xcolumn].to_numpy(), pad=0.04, log=bool(logx))
    y_lo, y_hi = _padded_limits(delta_pp.to_numpy(), pad=0.05)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy change (pp)")
    _panel(ax)
    if len(present) > 1:
        corner = _quiet_corner(merged[xcolumn].to_numpy(), delta_pp.to_numpy(), x_lo, x_hi, y_lo, y_hi)
        ax.legend(frameon=False, loc=corner, handletextpad=0.25, borderaxespad=0.15)
    _leader_labels(
        ax, merged[xcolumn].to_numpy(), delta_pp.to_numpy(),
        merged["dataset"].to_numpy(), np.abs(delta_pp.to_numpy()), k=4, min_abs=5.0,
    )
    fig.tight_layout(pad=0.35)
    _save(fig, path)


def _tradeoff(means: pd.DataFrame):
    frames = []
    for method, predicate, label, color in (
        ("cross_channel", lambda frame: frame["n_channels"] > 1, "Cross-channel", UNI),
        ("segment", lambda frame: frame["applied"] >= 1, "Temporal halves", MULTI),
    ):
        _summary, merged = _against_frozen(means, method, predicate)
        if merged is None or merged.empty:
            continue
        copied = merged.copy()
        copied["delta_pp"] = 100.0 * (copied["accuracy"] - copied["baseline"])
        copied["label"] = label
        copied["color"] = color
        frames.append(copied)
    if not frames:
        return
    plot_df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.patch.set_facecolor("white")
    for label, group in plot_df.groupby("label", sort=False):
        ax.scatter(
            group["extra_embed_seconds"], group["delta_pp"], s=16, c=group["color"].iloc[0],
            label=f"{label} ({len(group)})", edgecolor="white", linewidth=0.45, zorder=3,
        )
    ax.axhline(0.0, color="#55554f", lw=0.7, zorder=1)
    x_lo, x_hi = _padded_limits(plot_df["extra_embed_seconds"].to_numpy(), pad=0.04)
    x_lo = min(x_lo, 0.0)
    y_lo, y_hi = _padded_limits(plot_df["delta_pp"].to_numpy(), pad=0.05)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Extra embedding time (s)")
    ax.set_ylabel("Accuracy change (pp)")
    _panel(ax)
    corner = _quiet_corner(
        plot_df["extra_embed_seconds"].to_numpy(), plot_df["delta_pp"].to_numpy(),
        x_lo, x_hi, y_lo, y_hi,
    )
    ax.legend(frameon=False, loc=corner, handletextpad=0.25, borderaxespad=0.15)
    fig.tight_layout(pad=0.35)
    _save(fig, FIGURES / "tradeoff_accuracy_vs_time.pdf")


def _critical_difference(means: pd.DataFrame):
    methods = ["frozen_rf", "channel_pool", "cross_channel"]
    wide = means.pivot(index="dataset", columns="method", values="accuracy")
    meta = means[means["n_channels"] > 1][["dataset", "n_channels"]].drop_duplicates()
    wide = wide.join(meta.set_index("dataset"), how="inner")
    for method in methods:
        if method not in wide.columns:
            return "Critical difference diagram needs frozen_rf, channel_pool, and cross_channel."
    wide = wide.dropna(subset=methods)
    if len(wide) < 5:
        return "Fewer than 5 multivariate datasets have all three methods."
    scores = wide[methods].to_numpy()
    labels = [CD_LABEL[method] for method in methods]
    note = _draw_cd(scores, labels, FIGURES / "cd_multivariate.pdf")
    pairwise = []
    for left, right in (("frozen_rf", "channel_pool"), ("frozen_rf", "cross_channel"), ("channel_pool", "cross_channel")):
        summary = paired_summary(wide[left].to_numpy(), wide[right].to_numpy())
        pairwise.append([
            f"{METHOD_LABEL[right]} vs {METHOD_LABEL[left]}",
            str(summary["n"]),
            format_pp(summary["mean_delta"]),
            format_pp(summary["median_delta"]),
            f"{summary['wins']}/{summary['ties']}/{summary['losses']}",
            format_p(summary["p"]) or summary["note"],
        ])
    table = markdown_table(
        ["Comparison on multivariate datasets", "Datasets", "Mean Δ", "Median Δ", "W/T/L", "Wilcoxon p"],
        pairwise,
    )
    text = note + "\n\n" + table
    _write(TABLES / "cd_multivariate.md", text + "\n")
    return text


def _draw_cd(scores: np.ndarray, labels: list[str], path: Path) -> str:
    try:
        from aeon.visualisation import plot_critical_difference
    except Exception as exc:  # noqa: BLE001
        return f"Critical difference diagram was not drawn ({type(exc).__name__}: {exc})."
    plt.figure()
    try:
        result = plot_critical_difference(
            scores, labels, lower_better=False, test="wilcoxon", correction="holm", alpha=0.05
        )
    except TypeError:
        result = plot_critical_difference(scores, labels, lower_better=False, test="wilcoxon", alpha=0.05)
    fig = result[0] if isinstance(result, tuple) else result
    if not hasattr(fig, "savefig"):
        fig = plt.gcf()
    _restyle_cd(fig)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close("all")
    return (
        "Critical difference diagram on the multivariate datasets that have all three methods. "
        "α = 0.05, Wilcoxon with Holm correction when the installed aeon accepts that argument. "
        f"Saved to `{path.name}`."
    )


def _calibration() -> str:
    if not CAL_CSV.exists():
        return "Not run. `python scripts/run_calibration.py --datasets study`"
    frame = pd.read_csv(CAL_CSV)
    for column in ("ece_raw", "ece_ts", "ece_irm", "acc_raw", "acc_ts", "acc_irm", "n_channels"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    per_dataset = frame.groupby("dataset", as_index=False).agg(
        ece_raw=("ece_raw", "mean"),
        ece_ts=("ece_ts", "mean"),
        ece_irm=("ece_irm", "mean"),
        acc_raw=("acc_raw", "mean"),
        acc_ts=("acc_ts", "mean"),
        acc_irm=("acc_irm", "mean"),
        n_channels=("n_channels", "first"),
    )
    per_dataset["series"] = np.where(per_dataset["n_channels"] > 1, "multivariate", "univariate")
    rows = []
    for label, group in (("all", per_dataset), ("univariate", per_dataset[per_dataset["series"] == "univariate"]),
                         ("multivariate", per_dataset[per_dataset["series"] == "multivariate"])):
        if group.empty:
            continue
        ts = paired_summary(group["ece_raw"].to_numpy(), group["ece_ts"].to_numpy())
        irm = paired_summary(group["ece_raw"].to_numpy(), group["ece_irm"].to_numpy())
        rows.append([
            label,
            str(len(group)),
            f"{group['ece_raw'].mean():.4f}",
            f"{group['ece_ts'].mean():.4f}",
            format_pp(ts["mean_delta"]),
            format_p(ts["p"]) or ts["note"],
            f"{group['ece_irm'].mean():.4f}",
            format_pp(irm["mean_delta"]),
            format_p(irm["p"]) or irm["note"],
            f"{group['acc_raw'].mean():.4f}",
            f"{group['acc_irm'].mean():.4f}",
        ])
    headers = ["Group", "Datasets", "ECE raw", "ECE TS", "TS Δ", "TS p", "ECE IRM", "IRM Δ", "IRM p", "Acc raw", "Acc IRM"]
    ts_gap = float(np.max(np.abs(per_dataset["acc_ts"] - per_dataset["acc_raw"])))
    _calibration_figure(per_dataset)
    note = (
        "\n\nΔ is calibrated ECE minus raw ECE, so a negative value is a lower error. "
        f"Temperature scaling changed argmax accuracy by at most {ts_gap:.4f}. "
        "The forest was fit on 80% of the training split, so these accuracies are not the "
        "full-training numbers in the view table."
    )
    return markdown_table(headers, rows) + note


def _calibration_figure(per_dataset: pd.DataFrame):
    groups = ["univariate", "multivariate"]
    present = [group for group in groups if (per_dataset["series"] == group).any()]
    if not present:
        return
    raw, ts, irm = [], [], []
    for group in present:
        block = per_dataset[per_dataset["series"] == group]
        raw.append(block["ece_raw"].mean())
        ts.append(block["ece_ts"].mean())
        irm.append(block["ece_irm"].mean())
    index = np.arange(len(present))
    width = 0.24
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    fig.patch.set_facecolor("white")
    ax.bar(index - width, raw, width, label="Uncalibrated", color=UNI)
    ax.bar(index, ts, width, label="Temperature scaling", color=GREEN)
    ax.bar(index + width, irm, width, label="Isotonic regression", color=MULTI)
    ax.set_xticks(index, [group.capitalize() for group in present])
    ax.set_ylabel("Mean ECE")
    _panel(ax)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.15)
    fig.tight_layout(pad=0.35)
    _save(fig, FIGURES / "calibration_ece.pdf")


def _bottleneck() -> str:
    if not BOTTLENECK_JSON.exists():
        return "Not run. `python scripts/run_bottleneck.py`"
    report = json.loads(BOTTLENECK_JSON.read_text(encoding="utf-8"))
    lines = []
    params = report.get("params") or {}
    if params:
        total = params.get("total", 0)
        lines.append(f"Total parameters: {total:,}." if isinstance(total, int) else f"Total parameters: {total}.")
        rows = [[str(block["index"]), f"{block['params']:,}"] for block in params.get("blocks", [])]
        if rows:
            lines.append("")
            lines.append(markdown_table(["Transformer block", "Parameters"], rows))
        later = params.get("params_in_blocks_after_the_third")
        if later is not None and total:
            lines.append("")
            lines.append(
                f"Blocks after the third hold {later:,} parameters "
                f"({100.0 * later / total:.1f}% of the network)."
            )
    early = report.get("early_exit") or []
    if early:
        lines.append("")
        lines.append(markdown_table(
            ["Forward pass", "Seconds"],
            [[item["request"], f"{item['seconds']:.3f}"] for item in early],
        ))
    ratio = report.get("early_exit_time_ratio_layer0_over_last")
    if ratio is not None:
        lines.append("")
        lines.append(f"First-layer time divided by full forward time: {ratio:.2f}.")
    channels = report.get("scale_channels") or []
    if channels:
        lines.append("")
        lines.append(markdown_table(
            ["Channels", "Seconds for 128 series", "Seconds per channel"],
            [[str(item["channels"]), f"{item['seconds']:.3f}", f"{item['seconds_per_channel']:.3f}"]
             for item in channels],
        ))
        _channel_time_figure(channels)
    resize = report.get("resize") or []
    if resize:
        lines.append("")
        lines.append(markdown_table(
            ["Raw length", "Seconds to resize 128 series onto 512"],
            [[str(item["raw_length"]), f"{item['seconds']:.3f}"] for item in resize],
        ))
    notes = report.get("notes") or []
    if notes:
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) if lines else "The bottleneck file has no measurements."


def _channel_time_figure(channels: list[dict]):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    fig.patch.set_facecolor("white")
    xs = [item["channels"] for item in channels]
    ys = [item["seconds"] for item in channels]
    ax.plot(xs, ys, marker="o", ms=4.5, color=MULTI, lw=1.1, markerfacecolor=MULTI, markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel("Channels")
    ax.set_ylabel("Time (s)")
    _panel(ax)
    fig.tight_layout(pad=0.35)
    _save(fig, FIGURES / "bottleneck_channels.pdf")


def _narrative(means: pd.DataFrame) -> str:
    sentences = [
        "The encoder weights are the released Mantis-8M checkpoint. The classifier is the same "
        "200-tree forest used in the reproduction. The experiments change the array that forest reads.",
    ]
    pool_uni, merged_uni = _against_frozen(means, "channel_pool", lambda frame: frame["n_channels"] <= 1)
    if pool_uni is not None:
        gap = float(np.max(np.abs(merged_uni["accuracy"] - merged_uni["baseline"])))
        sentences.append(
            f"On {pool_uni['n']} univariate datasets, averaging the single channel block leaves the "
            f"embedding unchanged. The largest absolute accuracy gap against the frozen forest is {gap:.4f} "
            f"(mean {format_pp(pool_uni['mean_delta'])}, {pool_uni['wins']}/{pool_uni['ties']}/{pool_uni['losses']} "
            f"wins/ties/losses)."
        )
    pool_multi, _merged = _against_frozen(means, "channel_pool", lambda frame: frame["n_channels"] > 1)
    if pool_multi is not None:
        sentences.append(
            f"On {pool_multi['n']} multivariate datasets, that same average changes accuracy by "
            f"{format_pp(pool_multi['mean_delta'])} on average "
            f"(median {format_pp(pool_multi['median_delta'])}, "
            f"{pool_multi['wins']}/{pool_multi['ties']}/{pool_multi['losses']}, "
            f"p {format_p(pool_multi['p']) or pool_multi['note']}). "
            "A loss here means the forest was using channel identity, which the average throws away."
        )
    cross, _merged = _against_frozen(means, "cross_channel", lambda frame: frame["n_channels"] > 1)
    if cross is not None:
        sentences.append(
            f"Adding the per-series mean, standard deviation, and leading component on those "
            f"multivariate datasets changes accuracy by {format_pp(cross['mean_delta'])} on average "
            f"(median {format_pp(cross['median_delta'])}, "
            f"{cross['wins']}/{cross['ties']}/{cross['losses']}, "
            f"p {format_p(cross['p']) or cross['note']}). "
            "The extra forward pass is three series, whatever the channel count."
        )
    halves, _merged = _against_frozen(means, "segment", lambda frame: frame["applied"] >= 1)
    if halves is not None:
        sentences.append(
            f"On {halves['n']} datasets whose raw series are at least 1024 steps, embedding each temporal "
            f"half changes accuracy by {format_pp(halves['mean_delta'])} on average "
            f"(median {format_pp(halves['median_delta'])}, "
            f"{halves['wins']}/{halves['ties']}/{halves['losses']}, "
            f"p {format_p(halves['p']) or halves['note']})."
        )
    control, control_merged = _against_frozen(means, "segment", lambda frame: frame["applied"] < 1)
    if control is not None:
        gap = float(np.max(np.abs(control_merged["accuracy"] - control_merged["baseline"])))
        sentences.append(
            f"On {control['n']} shorter datasets the halves are not added. The largest absolute gap "
            f"against the frozen forest is {gap:.4f}, which is the check that this path reproduces the "
            "published inference."
        )
    shuffle = means[means["method"] == "channel_shuffle"].dropna(subset=["accuracy_unshuffled"])
    if not shuffle.empty:
        summary = paired_summary(shuffle["accuracy_unshuffled"].to_numpy(), shuffle["accuracy"].to_numpy())
        sentences.append(
            f"Permuting test channels, with the forest trained on the true order, changes accuracy by "
            f"{format_pp(summary['mean_delta'])} on {summary['n']} multivariate datasets "
            f"(median {format_pp(summary['median_delta'])}, "
            f"{summary['wins']}/{summary['ties']}/{summary['losses']}, "
            f"p {format_p(summary['p']) or summary['note']}). "
            "A negative gap means the channel index carries class information the average would delete."
        )
    sentences.append(
        "Use the scatter plots for the paired view, the delta plots for where a gain concentrates, "
        "and the time plot before describing a method as cheap. A non-significant p-value with a "
        "consistent median gap is a small sample, not evidence that the two methods are equal."
    )
    return " ".join(sentences)


def _figure_list() -> list[str]:
    lines = []
    for name in FIGURE_ORDER:
        path = FIGURES / name
        if not path.exists():
            continue
        caption = CAPTIONS.get(name, "")
        lines.append(f"- `figures/{name}`: {caption}" if caption else f"- `figures/{name}`")
    if not lines:
        lines.append("No figures were written. That happens when a comparison has fewer than two datasets.")
    return lines


def _padded_limits(values, pad=0.12, log=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    if log:
        values = values[values > 0]
        logged = np.log10(values)
        lo, hi = float(logged.min()), float(logged.max())
        span = max(hi - lo, 0.12)
        return 10 ** (lo - pad * span), 10 ** (hi + pad * span)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def _quiet_corner(x, y, x_lo, x_hi, y_lo=None, y_hi=None):
    """Legend corner with the fewest points. Upper left wins a tie."""
    y_lo = x_lo if y_lo is None else y_lo
    y_hi = x_hi if y_hi is None else y_hi
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Log axes: split on the geometric mean when the range covers more than a factor of four.
    if x_lo > 0 and x_hi / max(x_lo, 1e-12) > 4:
        x_mid = (x_lo * x_hi) ** 0.5
    else:
        x_mid = (x_lo + x_hi) / 2.0
    y_mid = (y_lo + y_hi) / 2.0
    counts = {
        "upper left": int(((x < x_mid) & (y >= y_mid)).sum()),
        "upper right": int(((x >= x_mid) & (y >= y_mid)).sum()),
        "lower left": int(((x < x_mid) & (y < y_mid)).sum()),
        "lower right": int(((x >= x_mid) & (y < y_mid)).sum()),
    }
    return min(counts, key=lambda name: (counts[name], 0 if name == "upper left" else 1))


# Short offsets only. A name is omitted when every one of these would leave the
# axes or send its leader across another marker.
_LEADER_OFFSETS = (
    (18, 9), (26, 0), (18, -9), (30, 12), (30, -12),
    (-18, 9), (-26, 0), (-18, -9), (-30, 12), (-30, -12),
    (12, 16), (-12, 16), (12, -16), (-12, -16),
    (38, 5), (-38, 5), (38, -10), (-38, -10),
)


def _leader_labels(ax, x, y, labels, score, k=4, min_abs=0.0):
    """Name the largest gaps, stacked in the same vertical order as the points.

    On an accuracy scatter the threshold matches the ±5 pp guides, so a point
    between the guides is not named. Wins are stacked on the left and losses
    on the right. On a difference plot each sign is stacked on the right.
    A name is omitted when that order would put the leader across another marker.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels).astype(str)
    ranked = [int(i) for i in np.argsort(score)[::-1] if np.isfinite(score[i]) and score[i] >= min_abs]
    ranked = ranked[:k]
    if not ranked:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = ax.transData.transform(np.column_stack([x, y]))
    ax_box = _box_tuple(ax.get_window_extent(renderer))
    occupied: list[tuple] = []
    legend = ax.get_legend()
    if legend is not None:
        occupied.append((_box_tuple(legend.get_window_extent(renderer).expanded(1.04, 1.15)), None, None))
    accuracy_plot = float(np.nanmax(x)) <= 1.05 and float(np.nanmax(y)) <= 1.05
    if accuracy_plot:
        groups = (
            ([i for i in ranked if y[i] >= x[i]], -1),
            ([i for i in ranked if y[i] < x[i]], 1),
        )
    else:
        groups = (
            ([i for i in ranked if y[i] >= 0], 1),
            ([i for i in ranked if y[i] < 0], 1),
        )
    missed = []
    for indices, side in groups:
        missed.extend((index, -side) for index in _place_column(
            ax, fig, indices, x, y, labels, points, occupied, ax_box, side,
        ))
    # A name that cannot sit on its own side still belongs in the same top-to-bottom order.
    for side in (1, -1):
        retry = [index for index, other in missed if other == side]
        if retry:
            _place_column(ax, fig, retry, x, y, labels, points, occupied, ax_box, side)


def _place_column(ax, fig, indices, x, y, labels, points, occupied, ax_box, side):
    """Stack one sign from top to bottom, every name on the same side.

    Returns the indices that could not be placed without leaving the axes or
    crossing another marker.
    """
    ceiling = ax_box[3]
    ha = "left" if side > 0 else "right"
    skipped = []
    for index in sorted(indices, key=lambda i: y[i], reverse=True):
        origin = (float(points[index, 0]), float(points[index, 1]))
        chosen = None
        for dist in (22, 32, 42, 16):
            dx = side * dist
            for dy in (0, 8, -8, 16, -16, 24, -24):
                probe = _annotate_leader(ax, labels[index], x[index], y[index], dx, dy, ha)
                fig.canvas.draw()
                box = _box_tuple(probe.get_window_extent(fig.canvas.get_renderer()))
                probe.remove()
                if box[3] > ceiling - 1:
                    continue
                end = (box[0], (box[1] + box[3]) / 2.0) if dx >= 0 else (box[2], (box[1] + box[3]) / 2.0)
                if _label_cost(box, ax_box, points, index, occupied, dx, dy, origin, end) >= _LABEL_REJECT:
                    continue
                chosen = (dx, dy, box, end)
                break
            if chosen is not None:
                break
        if chosen is None:
            skipped.append(index)
            continue
        dx, dy, box, end = chosen
        _annotate_leader(ax, labels[index], x[index], y[index], dx, dy, ha)
        occupied.append((box, origin, end))
        ceiling = box[1] - 3
    return skipped


def _annotate_leader(ax, text, x, y, dx, dy, ha):
    ann = ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6,
        color=INK,
        ha=ha,
        va="center",
        arrowprops={"arrowstyle": "-", "color": GUIDE, "lw": 0.45, "shrinkA": 0, "shrinkB": 2.5},
        annotation_clip=False,
        zorder=5,
    )
    ann.set_clip_on(False)
    return ann


def _box_tuple(box) -> tuple[float, float, float, float]:
    return (float(box.x0), float(box.y0), float(box.x1), float(box.y1))


_LABEL_REJECT = 500.0


def _label_cost(box, ax_box, points, index, occupied, dx, dy, origin, end) -> float:
    """Lower is better. Crossing a marker, or leaving the axes, is rejected."""
    cost = 0.12 * float(np.hypot(dx, dy))
    if box[0] < ax_box[0] + 1 or box[2] > ax_box[2] - 1 or box[1] < ax_box[1] + 1 or box[3] > ax_box[3] - 1:
        return _LABEL_REJECT
    for j, (px, py) in enumerate(points):
        if j == index:
            if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                cost += 40.0
            continue
        if box[0] - 3 <= px <= box[2] + 3 and box[1] - 3 <= py <= box[3] + 3:
            return _LABEL_REJECT
        if _line_hits_point(px, py, origin, end):
            return _LABEL_REJECT
    for other_box, other_origin, other_end in occupied:
        if _boxes_overlap(box, other_box, pad_x=3, pad_y=4):
            return _LABEL_REJECT
        if other_origin is not None and _segment_hits_box(other_origin, other_end, box, pad=3):
            return _LABEL_REJECT
        if _segment_hits_box(origin, end, other_box, pad=3):
            return _LABEL_REJECT
    return cost


def _line_hits_point(px, py, origin, end) -> bool:
    """True when the leader, past its own marker, passes over another point."""
    if _point_segment_distance(px, py, origin, end) > 8:
        return False
    if np.hypot(px - origin[0], py - origin[1]) < 9:
        return False
    return True


def _boxes_overlap(a, b, pad_x=0.0, pad_y=0.0) -> bool:
    return not (a[2] + pad_x < b[0] or b[2] + pad_x < a[0] or a[3] + pad_y < b[1] or b[3] + pad_y < a[1])


def _segment_hits_box(start, end, box, pad=0.0) -> bool:
    for step in range(2, 14):
        weight = step / 14.0
        px = start[0] + (end[0] - start[0]) * weight
        py = start[1] + (end[1] - start[1]) * weight
        if box[0] - pad <= px <= box[2] + pad and box[1] - pad <= py <= box[3] + pad:
            return True
    return False


def _point_segment_distance(px, py, start, end) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-9:
        return float(np.hypot(px - start[0], py - start[1]))
    weight = max(0.0, min(1.0, ((px - start[0]) * vx + (py - start[1]) * vy) / length_sq))
    return float(np.hypot(px - (start[0] + weight * vx), py - (start[1] + weight * vy)))


def _restyle_cd(fig):
    from matplotlib.text import Text

    fig.patch.set_facecolor("white")
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")
        ax.set_facecolor("white")
    for text in fig.findobj(match=Text):
        text.set_fontsize(7)
        text.set_color(INK)
    fig.set_size_inches(6.6, 1.85)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
