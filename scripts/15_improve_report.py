"""Tables and figures for the v3 improvements (scripts 11 to 14), following the
evaluation checklist of lecture weeks 7 to 9:
  * per-dataset score table and mean ranks (not mean accuracy) per experiment
  * planned pairwise comparisons, fixed BEFORE the results are seen, two-sided
    Wilcoxon signed-rank over datasets, Holm-corrected within each family
  * Friedman test and a critical difference diagram where 3+ methods are compared
  * cost next to accuracy (embedding time, feature size)
  * resampling: mean +- std over resamples per dataset, then the same paired tests
  * probability metrics: AUROC, log loss, ECE, not only accuracy
Each section runs only if its CSV exists, so partial runs still produce tables.

Outputs: results/tables/v3_*.md, results/figures/v3_*.pdf, results/v3_summary.json.

Usage
  python scripts/15_improve_report.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402
from src.stats_utils import (cd_diagram, comparison_table, friedman, mean_ranks,  # noqa: E402
                             pfmt, to_md)

TABLES, FIGS = RESULTS_DIR / "tables", RESULTS_DIR / "figures"
summary: dict = {}

PLANNED_MULTISCALE: list[tuple[str, str, str]] = [
    ("M+W2mean (RF200)", "M (RF200)", "Do two windows help the paper's head?"),
    ("M+W2mean (ridge)", "M (ridge)", "Do two windows help a ridge head?"),
    ("M+W2concat (RF200)", "M+W2mean (RF200)", "Concatenate or average the windows?"),
    ("M+W4mean (RF200)", "M+W2mean (RF200)", "Do four windows beat two?"),
    ("M+W2mean (ridge)", "MiniRocket (ridge)", "Does windowed Mantis close the gap to MiniRocket?"),
]

PLANNED_CHANNELS = [
    ("pca8 (ridge)", "full (ridge)", "PCA to 8 channels vs all channels"),
    ("random8 (ridge)", "full (ridge)", "Random 8 channels vs all channels"),
    ("summary (ridge)", "full (ridge)", "Three summary channels vs all channels"),
    ("pca8 (ridge)", "random8 (ridge)", "Is PCA better than a random subset?"),
]

PLANNED_RESAMPLE = [
    ("R+M (ridge)", "R (ridge)", "Does Mantis add to MiniRocket, over resamples?"),
    ("M (ridge)", "M (RF200)", "Ridge vs RF head, over resamples"),
    ("M+W2mean (ridge)", "M (ridge)", "Do two windows help a ridge head, over resamples?"),
    ("M+W2mean (RF200)", "M (RF200)", "Do two windows help the paper's head, over resamples?"),
    ("M+W2mean (ridge)", "R (ridge)", "Does windowed Mantis match MiniRocket, over resamples?"),
]

PLANNED_PROBA_LOGLOSS = [
    ("M (logreg)", "M (RF200)", "Log loss: logistic vs the paper's RF on Mantis"),
    ("R+M (logreg)", "R (logreg)", "Log loss: does Mantis improve MiniRocket's probabilities?"),
]
PLANNED_PROBA_AUROC = [
    ("M (logreg)", "M (RF200)", "AUROC: logistic vs the paper's RF on Mantis"),
    ("R+M (logreg)", "R (logreg)", "AUROC: does Mantis improve MiniRocket's ranking?"),
]


def label(features: pd.Series, classifier: pd.Series) -> pd.Series:
    return features.astype(str) + " (" + classifier.astype(str) + ")"


def pivot(df: pd.DataFrame, value: str, method_col: str = "method") -> pd.DataFrame:
    return df.pivot_table(index="dataset", columns=method_col, values=value, aggfunc="mean")


def minirocket_rows(datasets) -> pd.Series | None:
    path = RESULTS_DIR / "fusion.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path)
    f = f[(f.features == "R") & (f.classifier == "ridge")]
    s = f.groupby("dataset").accuracy.mean().reindex(datasets)
    return s if s.notna().any() else None


def overview(acc: pd.DataFrame, cost: pd.DataFrame | None, name: str) -> pd.DataFrame:
    table = pd.DataFrame({"method": acc.columns, "mean_rank": mean_ranks(acc).values,
                          "mean_accuracy": acc.mean().values, "datasets": acc.notna().sum().values})
    if cost is not None:
        table = table.merge(cost, on="method", how="left")
    table = table.sort_values("mean_rank")
    fr = friedman(acc.dropna())
    print(f"\nFriedman chi2 = {fr['chi2']:.2f}, p = {pfmt(fr['p'])} (k = {fr['k']}, N = {fr['n']})")
    summary[f"{name}_friedman"] = fr
    to_md(table, TABLES / f"v3_{name}_overview")
    summary[f"{name}_overview"] = table.to_dict(orient="records")
    return table


def section_multiscale():
    path = RESULTS_DIR / "multiscale.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["method"] = label(df.features, df.classifier)
    acc = pivot(df, "accuracy")
    mr = minirocket_rows(acc.index)
    if mr is not None:
        acc["MiniRocket (ridge)"] = mr
    # MiniRocket comes from fusion.csv; only datasets with every method are compared
    acc = acc.dropna()
    print(f"\n=== Multi-scale: {len(acc)} datasets with every method ===")
    cost = df.groupby("method")[["embed_seconds", "n_features"]].agg(
        embed_minutes=("embed_seconds", lambda s: s.sum() / 60 / max(df.seed.nunique(), 1)),
        median_features=("n_features", "median")).reset_index()
    overview(acc, cost, "multiscale")
    to_md(acc.reset_index(), TABLES / "v3_multiscale_per_dataset")
    if PLANNED_MULTISCALE:
        t = comparison_table(acc, PLANNED_MULTISCALE)
        to_md(t, TABLES / "v3_multiscale_planned", floatfmt=".2f")
        summary["multiscale_planned"] = t.to_dict(orient="records")
    else:
        print("[note] PLANNED_MULTISCALE is empty: no paired tests for the multi-scale experiment.")
    cd_cols = [c for c in acc.columns if c.endswith("(ridge)")]
    cd_diagram(acc[cd_cols], FIGS / "v3_multiscale_cd.pdf")
    meta = df.groupby("dataset")[["max_len", "n_train", "n_classes"]].first()
    best_win = [c for c in acc.columns if c.startswith("M+W") and c.endswith("(ridge)")]
    if best_win and "M (ridge)" in acc:
        gain = acc[best_win].max(axis=1) - acc["M (ridge)"]
        t = pd.concat([acc[["M (ridge)"]], acc[best_win].max(axis=1).rename("best_windowed (ridge)"),
                       gain.rename("delta"), meta.loc[acc.index]], axis=1)
        t = t.reindex(gain.abs().sort_values(ascending=False).index).head(12).reset_index()
        to_md(t, TABLES / "v3_multiscale_largest_changes")


def section_channels():
    path = RESULTS_DIR / "channel_reduce.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["method"] = label(df.method, df.classifier)
    acc = pivot(df, "accuracy").dropna()
    print(f"\n=== Channel reduction: {len(acc)} datasets with every method ===")
    secs = df[df.dataset.isin(acc.index)].groupby(["method", "dataset"]).embed_seconds.mean().groupby("method").sum()
    mb = df[df.dataset.isin(acc.index)].groupby("method").feature_mb.max()
    cost = pd.DataFrame({"method": secs.index, "embed_seconds_total": secs.values,
                         "speedup_vs_full": (secs["full (ridge)"] / secs).values if "full (ridge)" in secs else np.nan,
                         "max_feature_mb": mb.reindex(secs.index).values})
    overview(acc, cost, "channels")
    to_md(acc.reset_index(), TABLES / "v3_channels_per_dataset")
    t = comparison_table(acc, PLANNED_CHANNELS)
    if not t.empty:
        to_md(t, TABLES / "v3_channels_planned", floatfmt=".2f")
        summary["channels_planned"] = t.to_dict(orient="records")
    cd_diagram(acc[[c for c in acc.columns if c.endswith("(ridge)")]], FIGS / "v3_channels_cd.pdf")
    per = df.groupby(["dataset", "method"])[["accuracy", "embed_seconds", "channels_used", "channels_in"]].mean()
    to_md(per.reset_index(), TABLES / "v3_channels_accuracy_vs_time", floatfmt=".3f")


def section_resample():
    path = RESULTS_DIR / "resample.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["method"] = label(df.features, df.classifier)
    n_res = df["resample"].nunique()
    mean_all = pivot(df, "accuracy").dropna(axis=1, how="all")
    cols = list(mean_all.columns)
    std_all = df.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="std")[cols]
    official_all = df[df["resample"] == 0].pivot_table(index="dataset", columns="method", values="accuracy")[cols]
    mean = mean_all.dropna()
    if len(mean) < 2:
        print("\n[note] fewer than 2 datasets have every resample method; skipping the resampling tables.")
        return
    std, official = std_all.loc[mean.index], official_all.loc[mean.index]
    print(f"\n=== Resampling: {len(mean)} datasets with every method, {n_res} resamples ===")
    table = pd.DataFrame({"method": mean.columns, "mean_rank": mean_ranks(mean).values,
                          "mean_accuracy": mean.mean().values,
                          "accuracy_official_split": official[mean.columns].mean().values,
                          "mean_std_over_resamples": std[mean.columns].mean().values,
                          "resamples": n_res}).sort_values("mean_rank")
    to_md(table, TABLES / "v3_resample_overview")
    summary["resample_overview"] = table.to_dict(orient="records")
    t = comparison_table(mean_all, PLANNED_RESAMPLE)
    if not t.empty:
        # each pair uses every dataset that has both methods, so n can differ between rows
        # practical check: datasets where the gap is below one resample std
        for i, (a, b, _) in enumerate([p for p in PLANNED_RESAMPLE if p[0] in mean_all and p[1] in mean_all]):
            gap = (mean_all[a] - mean_all[b]).abs()
            noise = np.sqrt((std_all[a] ** 2 + std_all[b] ** 2) / 2)
            both_present = mean_all[a].notna() & mean_all[b].notna()
            t.loc[i, "datasets_gap_below_1std"] = int(((gap < noise) & both_present).sum())
        to_md(t, TABLES / "v3_resample_planned", floatfmt=".2f")
        summary["resample_planned"] = t.to_dict(orient="records")
    both = pd.concat([mean_all.add_suffix(" mean"), std_all.add_suffix(" std")], axis=1).sort_index(axis=1)
    to_md(both.reset_index(), TABLES / "v3_resample_per_dataset")
    cd_diagram(mean, FIGS / "v3_resample_cd.pdf")


def section_proba():
    path = RESULTS_DIR / "proba.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["method"] = label(df.features, df.classifier)
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "auroc", "log_loss", "ece"]
    per = {m: pivot(df, m) for m in metrics}
    keep = per["accuracy"].dropna().index
    print(f"\n=== Probability metrics: {len(keep)} datasets with every method ===")
    table = pd.DataFrame({"method": per["accuracy"].columns})
    for m in metrics:
        table[f"mean_{m}"] = per[m].loc[keep].mean().values
    table["mean_rank_accuracy"] = mean_ranks(per["accuracy"].loc[keep]).values
    table["mean_rank_log_loss"] = per["log_loss"].loc[keep].rank(axis=1, ascending=True, method="average").mean().values
    table = table.sort_values("mean_rank_accuracy")
    to_md(table, TABLES / "v3_proba_overview")
    summary["proba_overview"] = table.to_dict(orient="records")
    t = comparison_table(per["log_loss"].loc[keep], PLANNED_PROBA_LOGLOSS, higher_is_better=False)
    if not t.empty:
        t = t.rename(columns={"mean_diff_pp": "mean_diff_x100", "gain_>1pp": "better_>0.01", "loss_>1pp": "worse_>0.01"})
        to_md(t, TABLES / "v3_proba_planned_logloss", floatfmt=".2f")
        summary["proba_planned_logloss"] = t.to_dict(orient="records")
    t = comparison_table(per["auroc"].loc[keep], PLANNED_PROBA_AUROC)
    if not t.empty:
        to_md(t, TABLES / "v3_proba_planned_auroc", floatfmt=".2f")
        summary["proba_planned_auroc"] = t.to_dict(orient="records")
    to_md(per["log_loss"].loc[keep].reset_index(), TABLES / "v3_proba_log_loss_per_dataset")
    to_md(per["auroc"].loc[keep].reset_index(), TABLES / "v3_proba_auroc_per_dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    section_multiscale()
    section_channels()
    section_resample()
    section_proba()
    out = RESULTS_DIR / "v3_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
