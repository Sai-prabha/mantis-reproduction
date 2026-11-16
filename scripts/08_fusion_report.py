"""Tables and figures for the Section 4 fusion experiment (results/fusion.csv).

Follows the evaluation checklist of lecture weeks 7 to 9:
  * per-dataset score table for every method (mean over seeds)
  * ranks: average rank over datasets is the headline, not average accuracy
  * Friedman test across all methods, then a critical difference (CD) diagram
    (aeon: one-sided Wilcoxon tests, Holm correction) for the key methods
  * planned pairwise comparisons, fixed before the results were seen,
    two-sided Wilcoxon signed-rank tests over datasets with a Holm correction
  * post hoc comparisons (added after seeing results), reported separately, own Holm family
  * if results/stats_ablation.csv exists: each statistics subset vs Mantis alone (09)
  * if results/dtw.csv exists: the overview again with 1NN-DTW on the datasets it completed (10)
  * a one-row multiple comparison matrix (MCM): full model vs every other method
  * practical significance: datasets with a gain or loss above 1 point
  * groups defined in advance from metadata (length, channels, classes, train size)
  * cost: run time and feature-matrix size; accuracy-vs-runtime figure
  * sanity checks: our MiniRocket and Mantis (RF) rows against 04 and 02

Only datasets on which every method finished are compared; the rest are listed.

Usage
  python scripts/08_fusion_report.py
  python scripts/08_fusion_report.py --csv results/fusion_weighting-none.csv --tag none
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402

TABLES, FIGS = RESULTS_DIR / "tables", RESULTS_DIR / "figures"

LABELS = {
    ("none", "dummy"): "Majority class",
    ("M", "RF200"): "Mantis (RF)",
    ("M+S", "RF200"): "Mantis+Stats (RF)",
    ("M", "ridge"): "Mantis (ridge)",
    ("S", "ridge"): "Stats (ridge)",
    ("M+S", "ridge"): "Mantis+Stats (ridge)",
    ("R", "ridge"): "MiniRocket",
    ("R+S", "ridge"): "MiniRocket+Stats",
    ("R+M", "ridge"): "MiniRocket+Mantis",
    ("R+M+S", "ridge"): "MiniRocket+Mantis+Stats",
}
TARGET = "MiniRocket+Mantis+Stats"

PLANNED = [
    ("Mantis+Stats (RF)", "Mantis (RF)", "Does scale information help Mantis?"),
    ("MiniRocket+Mantis", "MiniRocket", "Does Mantis add to MiniRocket?"),
    ("MiniRocket+Mantis+Stats", "MiniRocket+Mantis", "Do statistics add on top?"),
    ("MiniRocket+Mantis+Stats", "MiniRocket", "Full model vs strongest baseline"),
    ("Mantis (ridge)", "Mantis (RF)", "Control: classifier change only"),
]

POST_HOC = [
    ("Mantis+Stats (ridge)", "Mantis (RF)", "Best Mantis-only setup vs the paper's protocol"),
    ("Mantis+Stats (ridge)", "Mantis (ridge)", "Do statistics still help with a ridge head?"),
]

ABLATION_SUBSETS = {"level": "mean, std, min, max", "slope": "slope",
                    "diff": "diff mean, std, min, max", "all": "all nine"}

CD_METHODS = ["Majority class", "Mantis (RF)", "Mantis+Stats (RF)", "MiniRocket",
              "MiniRocket+Mantis", "MiniRocket+Mantis+Stats"]

GROUPS = {
    "all": lambda m: m.index == m.index,
    "univariate": lambda m: m.n_channels == 1,
    "multivariate": lambda m: m.n_channels > 1,
    "length <= 512 (not downsampled)": lambda m: m.max_len <= 512,
    "length > 1000 (downsampled 2x+)": lambda m: m.max_len > 1000,
    "more than 10 classes": lambda m: m.n_classes > 10,
    "at most 50 training cases": lambda m: m.n_train <= 50,
    "unequal length": lambda m: m.unequal_length == 1,
}
PRACTICAL = 0.01

summary: dict = {}


def to_md(df: pd.DataFrame, name: str, floatfmt: str = ".4f", index: bool = False) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    text = df.to_markdown(index=index, floatfmt=floatfmt)
    (TABLES / f"{name}.md").write_text(text, encoding="utf-8")
    print(f"\n### {name}\n{text}")


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def compare(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    out = {"n": int(len(d)), "mean_diff_pp": 100 * float(d.mean()),
           "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
           f"gain_>{int(PRACTICAL*100)}pp": int((d > PRACTICAL).sum()),
           f"loss_>{int(PRACTICAL*100)}pp": int((d < -PRACTICAL).sum())}
    out["p"] = float(wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() >= 1 else 1.0
    return out


def pfmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def load(csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv)
    df["method"] = [LABELS.get((f, c), f"{f} ({c})") for f, c in zip(df.features, df.classifier)]
    acc = df.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    bal = df.pivot_table(index="dataset", columns="method", values="balanced_accuracy", aggfunc="mean")
    meta = df.groupby("dataset")[["archive", "n_train", "n_test", "n_classes", "n_channels",
                                  "max_len", "unequal_length"]].first()
    order = [m for m in LABELS.values() if m in acc.columns] + \
            [m for m in acc.columns if m not in LABELS.values()]
    return df, acc[order], bal[order], meta


def completeness(acc: pd.DataFrame) -> pd.DataFrame:
    complete = acc.dropna()
    missing = acc[acc.isna().any(axis=1)]
    rows = [{"dataset": d, "missing_methods": ", ".join(c for c in acc.columns if pd.isna(r[c]))}
            for d, r in missing.iterrows()]
    summary["datasets_compared"] = len(complete)
    summary["datasets_excluded"] = rows
    if rows:
        to_md(pd.DataFrame(rows), "fusion_excluded_datasets")
    print(f"\n{len(complete)} datasets have every method; {len(rows)} excluded.")
    return complete


def overview(df, acc, bal):
    ranks = acc.rank(axis=1, ascending=False, method="average")
    bal_ranks = bal.loc[acc.index].rank(axis=1, ascending=False, method="average")
    cost = df[df.dataset.isin(acc.index)].groupby(["method", "dataset"])[
        ["total_seconds", "feature_mb"]].mean().groupby("method")
    seeds = df[df.dataset.isin(acc.index)].groupby("method").seed.nunique()
    table = pd.DataFrame({
        "method": acc.columns,
        "mean_rank": ranks.mean().values,
        "mean_accuracy": acc.mean().values,
        "mean_balanced_accuracy": bal.loc[acc.index].mean().values,
        "mean_rank_balanced": bal_ranks.mean().values,
        "total_minutes": (cost.total_seconds.sum() / 60).reindex(acc.columns).values,
        "median_feature_mb": cost.feature_mb.median().reindex(acc.columns).values,
        "max_feature_mb": cost.feature_mb.max().reindex(acc.columns).values,
        "seeds": seeds.reindex(acc.columns).values,
    }).sort_values("mean_rank")
    to_md(table, "fusion_overview")
    summary["overview"] = table.to_dict(orient="records")

    alpha = df[(df.classifier == "ridge") & df.dataset.isin(acc.index)]
    edge = alpha.groupby("method").alpha_at_grid_edge.mean()
    summary["ridge_alpha_at_grid_edge_fraction"] = edge.to_dict()
    if (edge > 0.2).any():
        print("\n[warning] ridge alpha at the edge of the grid in more than 20% of runs for: "
              + ", ".join(edge[edge > 0.2].index) + ". Consider a wider grid (src/features.py).")

    stat, p = friedmanchisquare(*[acc[c].values for c in acc.columns])
    summary["friedman"] = {"chi2": float(stat), "p": float(p), "k": acc.shape[1], "n": acc.shape[0]}
    print(f"\nFriedman chi2 = {stat:.2f}, p = {pfmt(p)} (k = {acc.shape[1]}, N = {acc.shape[0]})")

    TABLES.mkdir(parents=True, exist_ok=True)
    acc.round(4).to_csv(TABLES / "fusion_per_dataset.csv")
    to_md(acc.reset_index(), "fusion_per_dataset")
    return ranks, table


def comparison_family(acc, pairs, name, key):
    rows = []
    for a, b, question in pairs:
        if a in acc and b in acc:
            rows.append({"comparison": f"{a} vs {b}", "question": question, **compare(acc[a], acc[b])})
    if not rows:
        return
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
        r["p"], r["p_holm"] = pfmt(r["p"]), pfmt(r["p_holm"])
    to_md(pd.DataFrame(rows), name, floatfmt=".2f")
    summary[key] = rows


def planned(acc):
    comparison_family(acc, PLANNED, "fusion_planned_comparisons", "planned")


def post_hoc(acc):
    print("\n[note] The next table was added after the results were seen; the planned list above "
          "was fixed beforehand. Treat it as exploratory.")
    comparison_family(acc, POST_HOC, "fusion_posthoc_comparisons", "post_hoc")


def stats_ablation(df):
    """Mantis + one statistics subset vs Mantis alone (results/stats_ablation.csv against the
    M rows of fusion.csv), per classifier. Exploratory: one Holm family for this table."""
    path = RESULTS_DIR / "stats_ablation.csv"
    if not path.exists():
        return
    ab = pd.read_csv(path)
    base = df[df.features == "M"].pivot_table(index="dataset", columns="classifier", values="accuracy")
    rows = []
    for clf in ["RF200", "ridge"]:
        sub = ab[ab.classifier == clf].pivot_table(index="dataset", columns="subset", values="accuracy")
        if clf not in base or sub.empty:
            continue
        sub = sub.dropna().join(base[clf].rename("M"), how="inner")
        for s in ABLATION_SUBSETS:
            if s in sub:
                rows.append({"classifier": clf, "subset": s, "statistics": ABLATION_SUBSETS[s],
                             "mean_accuracy_M": float(sub["M"].mean()),
                             "mean_accuracy": float(sub[s].mean()), **compare(sub[s], sub["M"])})
    if not rows:
        return
    print("\n[note] Statistics ablation: added after the fusion results were seen; exploratory.")
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p"], r["p_holm"] = pfmt(r["p"]), pfmt(q)
    to_md(pd.DataFrame(rows), "fusion_stats_ablation")
    summary["stats_ablation"] = rows


def dtw_overview(df, acc, bal):
    """Overview again, with 1NN-DTW (results/dtw.csv), on the datasets DTW completed."""
    path = RESULTS_DIR / "dtw.csv"
    if not path.exists():
        return
    d = pd.read_csv(path).set_index("dataset")
    skip = RESULTS_DIR / "dtw_skipped.csv"
    n_skipped = len(pd.read_csv(skip)) if skip.exists() else 0
    idx = acc.index.intersection(d.index)
    print(f"\n1NN-DTW finished on {len(idx)} of the {len(acc)} compared datasets; "
          f"{n_skipped} datasets skipped (results/dtw_skipped.csv).")
    if len(idx) < 2:
        return
    a = acc.loc[idx].assign(**{"1NN-DTW": d.accuracy.reindex(idx)})
    b = bal.loc[idx].assign(**{"1NN-DTW": d.balanced_accuracy.reindex(idx)})
    secs = df[df.dataset.isin(idx)].groupby(["method", "dataset"]).total_seconds.mean().groupby("method").sum()
    secs["1NN-DTW"] = d.seconds.reindex(idx).sum()
    table = pd.DataFrame({
        "method": a.columns,
        "mean_rank": a.rank(axis=1, ascending=False, method="average").mean().values,
        "mean_accuracy": a.mean().values,
        "mean_balanced_accuracy": b.mean().values,
        "total_minutes": (secs.reindex(a.columns) / 60).values,
    }).sort_values("mean_rank")
    to_md(table, "fusion_overview_with_dtw")
    summary["overview_with_dtw"] = {"n_datasets": int(len(idx)), "dtw_skipped": n_skipped,
                                    "table": table.to_dict(orient="records")}


def mcm_row(acc):
    if TARGET not in acc:
        return
    rows = [{"vs": c, **compare(acc[TARGET], acc[c])} for c in acc.columns if c != TARGET]
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = pfmt(q)
        r["p"] = pfmt(r["p"])
    to_md(pd.DataFrame(rows), "fusion_mcm_row", floatfmt=".2f")
    summary["mcm_row"] = rows


def groups(acc, meta):
    keys = [m for m in ["Mantis (RF)", "Mantis+Stats (RF)", "MiniRocket", "MiniRocket+Mantis", TARGET]
            if m in acc]
    m = meta.loc[acc.index]
    rows = []
    for g, rule in GROUPS.items():
        idx = acc.index[np.asarray(rule(m))]
        if len(idx) == 0:
            continue
        row = {"group": g, "n": len(idx), **acc.loc[idx, keys].mean().to_dict()}
        if TARGET in acc and "MiniRocket" in acc:
            c = compare(acc.loc[idx, TARGET], acc.loc[idx, "MiniRocket"])
            row["full_minus_MR_pp"] = c["mean_diff_pp"]
            row["W/T/L vs MR"] = f"{c['wins']}/{c['ties']}/{c['losses']}"
        if {"Mantis+Stats (RF)", "Mantis (RF)"} <= set(acc):
            row["stats_gain_for_Mantis_pp"] = 100 * float(
                (acc.loc[idx, "Mantis+Stats (RF)"] - acc.loc[idx, "Mantis (RF)"]).mean())
        rows.append(row)
    to_md(pd.DataFrame(rows), "fusion_groups")
    summary["groups"] = rows


def largest_changes(acc, meta, n=10):
    pairs = [("Mantis+Stats (RF)", "Mantis (RF)", "stats_for_mantis"),
             (TARGET, "MiniRocket", "full_vs_minirocket")]
    for a, b, tag in pairs:
        if a not in acc or b not in acc:
            continue
        d = (acc[a] - acc[b]).rename("delta")
        t = pd.concat([acc[[b, a]], d, meta.loc[acc.index, ["n_train", "max_len", "n_classes",
                                                             "n_channels"]]], axis=1)
        t = t.reindex(d.abs().sort_values(ascending=False).index).head(n)
        to_md(t.reset_index(), f"fusion_largest_changes_{tag}")


def sanity(acc):
    checks = {}
    mr = RESULTS_DIR / "minirocket.csv"
    if "MiniRocket" in acc and mr.exists():
        ref = pd.read_csv(mr).groupby("dataset").accuracy.mean()
        d = (acc["MiniRocket"] - ref).dropna()
        checks["MiniRocket_vs_04"] = {"n": len(d), "mean_abs_diff": float(d.abs().mean()),
                                      "max_abs_diff": float(d.abs().max()) if len(d) else None}
    zs = RESULTS_DIR / "zero_shot.csv"
    if "Mantis (RF)" in acc and zs.exists():
        z = pd.read_csv(zs)
        z = z[(z.layer == -1) & (z.token == "cls_token")].groupby("dataset").accuracy.mean()
        d = (acc["Mantis (RF)"] - z).dropna()
        checks["MantisRF_vs_02"] = {"n": len(d), "mean_abs_diff": float(d.abs().mean()),
                                    "max_abs_diff": float(d.abs().max()) if len(d) else None}
    summary["sanity_checks"] = checks
    print("\nSanity checks (should be close to 0):\n" + json.dumps(checks, indent=2))


def cd_diagram(acc, tag):
    cols = [c for c in CD_METHODS if c in acc]
    if len(cols) < 3:
        return
    try:
        from aeon.visualisation import plot_critical_difference
        res = plot_critical_difference(acc[cols].values, cols, test="wilcoxon", correction="holm")
        fig = res[0] if isinstance(res, tuple) else res
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / f"fusion_cd{tag}.pdf", bbox_inches="tight")
        print(f"wrote {FIGS / f'fusion_cd{tag}.pdf'}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip] CD diagram: {type(e).__name__}: {e}")


def runtime_plot(table, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[skip] runtime plot: {e}")
        return
    t = table.dropna(subset=["total_minutes"])
    t = t[t.total_minutes > 0]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(t.total_minutes, t.mean_rank, s=25)
    for _, r in t.iterrows():
        ax.annotate(r["method"], (r.total_minutes, r.mean_rank), fontsize=7,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Total run time over all datasets (minutes, log scale)")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.grid(alpha=0.3)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / f"fusion_rank_vs_runtime{tag}.pdf")
    plt.close(fig)
    print(f"wrote {FIGS / f'fusion_rank_vs_runtime{tag}.pdf'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(RESULTS_DIR / "fusion.csv"))
    ap.add_argument("--tag", default="", help="suffix for figure names and the summary file")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    csv = Path(args.csv)
    if not csv.exists():
        sys.exit(f"{csv} not found; run scripts/07_fusion.py first")

    df, acc_all, bal, meta = load(csv)
    acc = completeness(acc_all)
    if len(acc) < 2:
        sys.exit("fewer than 2 complete datasets; nothing to compare")
    _, table = overview(df, acc, bal)
    planned(acc)
    post_hoc(acc)
    stats_ablation(df)
    dtw_overview(df, acc, bal)
    mcm_row(acc)
    groups(acc, meta)
    largest_changes(acc, meta)
    sanity(acc)
    cd_diagram(acc, tag)
    runtime_plot(table, tag)
    out = RESULTS_DIR / f"fusion_summary{tag}.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
