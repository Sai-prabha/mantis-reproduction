"""Turn results/*.csv into the report's tables and figures.

Writes markdown tables to results/tables/, figures to results/figures/ and a
machine-readable results/summary.json. Every number in Sections 3 and 5 (README
results table) should come from here, never typed by hand.

Paper targets are read from targets/*.csv (transcribed from v1's appendix; see
targets/README.md). Missing target files are skipped, not faked.

Usage
  python scripts/06_summarise.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.calibration import ece, softmax  # noqa: E402
from src.common import RESULTS_DIR, ROOT  # noqa: E402

TABLES, FIGS = RESULTS_DIR / "tables", RESULTS_DIR / "figures"
TARGETS = ROOT / "targets"
CROSS_CHECK_UCR128 = 0.7921  # original Mantis, RF, UCR-128 (arXiv 2602.17868, Fig. 9b / Table 1)
PAPER_FULL_MINUS_SCRATCH = {"last": 0.0508, "best": 0.0392}  # v1, Figure 7 (151 datasets)
summary: dict = {}


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else None


def to_md(df: pd.DataFrame, name: str, floatfmt: str = ".4f") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt))
    print(f"\n### {name}\n{df.to_markdown(index=False, floatfmt=floatfmt)}")


def paired_test(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    out = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else np.nan,
           "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}
    if len(d) >= 6 and (d != 0).any():
        out["wilcoxon_p"] = float(wilcoxon(d[d != 0]).pvalue)
    return out


def per_dataset(df: pd.DataFrame, col: str = "accuracy") -> pd.DataFrame:
    return df.groupby("dataset")[col].agg(["mean", "std", "count"]).reset_index()


# ------------------------------------------------------------ zero-shot --

def zero_shot():
    zs = read(RESULTS_DIR / "zero_shot.csv")
    if zs is None:
        print("no zero_shot.csv yet")
        return None
    zs = zs[(zs.layer == -1) & (zs.token == "cls_token")]  # v1 protocol only
    ours = per_dataset(zs).rename(columns={"mean": "ours", "std": "ours_std", "count": "seeds"})
    ours = ours.merge(zs.drop_duplicates("dataset")[["dataset", "archive", "n_train", "n_test",
                      "n_classes", "n_channels", "min_len", "max_len", "unequal_length",
                      "n_missing_values"]], on="dataset")

    arch = ours.groupby("archive")["ours"].agg(["count", "mean"]).reset_index()
    to_md(arch.rename(columns={"count": "datasets", "mean": "mean accuracy (ours)"}), "zs_archive_means")
    ucr = ours[ours.archive == "ucr"]
    summary["zero_shot"] = {"ucr_mean_ours": float(ucr.ours.mean()), "ucr_n": int(len(ucr)),
                            "ucr_cross_check_2602.17868": CROSS_CHECK_UCR128,
                            "seed_std_median": float(ours.ours_std.median())}

    tgt = read(TARGETS / "v1_table3_zeroshot.csv")
    if tgt is not None and tgt["mantis"].notna().any():
        m = ours.merge(tgt[["dataset", "mantis"]].rename(columns={"mantis": "paper"}), on="dataset")
        m = m.dropna(subset=["paper"])
        m["delta"] = m["ours"] - m["paper"]
        m["within_2std"] = (m["delta"].abs() <= 2 * m["ours_std"].fillna(0) + 1e-9)
        summary["zero_shot"]["vs_paper"] = paired_test(m["ours"], m["paper"])
        summary["zero_shot"]["vs_paper"]["share_within_2std"] = float(m["within_2std"].mean())
        by_arch = m.groupby("archive")[["paper", "ours", "delta"]].mean().reset_index()
        to_md(by_arch, "zs_paper_vs_ours_by_archive")
        worst = m.reindex(m["delta"].abs().sort_values(ascending=False).index).head(12)
        to_md(worst[["dataset", "archive", "paper", "ours", "ours_std", "delta", "n_train",
                     "max_len", "n_classes", "unequal_length", "n_missing_values"]], "zs_largest_gaps")
        scatter(m["paper"], m["ours"], "Zero-shot accuracy (RF200, last layer, CLS)",
                "paper (v1 Table 3)", "ours", "zs_scatter.png")
    return ours


# ------------------------------------------------------------ baselines --

def baselines(ours):
    if ours is None:
        return
    table = ours[["dataset", "archive", "ours"]].rename(columns={"ours": "Mantis-8M"})
    for name, file in [("MiniRocket", "minirocket.csv"), ("MOMENT-1-large", "moment.csv")]:
        df = read(RESULTS_DIR / file)
        if df is not None:
            table = table.merge(per_dataset(df)[["dataset", "mean"]].rename(columns={"mean": name}),
                                on="dataset", how="left")
    cols = [c for c in table.columns if c not in ("dataset", "archive")]
    to_md(table.groupby("archive")[cols].mean().reset_index(), "baselines_archive_means")
    summary["baselines"] = {c: paired_test(table["Mantis-8M"], table[c])
                            for c in cols if c != "Mantis-8M"}
    tgt = read(TARGETS / "v1_table3_zeroshot.csv")
    if tgt is not None and {"mantis", "moment"} <= set(tgt.columns) and "MOMENT-1-large" in table:
        m = table.merge(tgt, on="dataset").dropna(subset=["mantis", "moment", "MOMENT-1-large"])
        summary["baselines"]["mantis_minus_moment"] = {
            "paper": float((m["mantis"] - m["moment"]).mean()),
            "ours": float((m["Mantis-8M"] - m["MOMENT-1-large"]).mean()), "n": int(len(m))}


# ----------------------------------------------------------- fine-tuning --

def finetune():
    ft = read(RESULTS_DIR / "finetune.csv")
    if ft is None:
        print("no finetune.csv yet")
        return
    final = ft[ft.is_final == 1]
    regimes = [r for r in ["rf", "head", "full", "scratch", "full_pkg_defaults"] if r in set(final.regime)]

    # accuracy per regime (last epoch), averaged over seeds then datasets
    rows = []
    for r in regimes:
        d = final[final.regime == r].groupby("dataset")[["acc_last", "acc_val_selected",
                                                         "acc_oracle_best_test"]].mean()
        rows.append({"regime": r, "datasets": len(d), **d.mean(numeric_only=True).to_dict()})
    to_md(pd.DataFrame(rows), "ft_regimes")

    wide = final.groupby(["dataset", "regime"])[["acc_last", "acc_oracle_best_test"]].mean().unstack()
    if {"full", "scratch"} <= set(regimes):
        gap_last = wide["acc_last"]["full"] - wide["acc_last"]["scratch"]
        gap_best = wide["acc_oracle_best_test"]["full"] - wide["acc_oracle_best_test"]["scratch"]
        summary["full_minus_scratch"] = {
            "ours_last": float(gap_last.mean()), "paper_last": PAPER_FULL_MINUS_SCRATCH["last"],
            "ours_best_oracle": float(gap_best.mean()), "paper_best": PAPER_FULL_MINUS_SCRATCH["best"],
            "test_last": paired_test(wide["acc_last"]["full"], wide["acc_last"]["scratch"])}
    if "full" in regimes:
        f = final[final.regime == "full"]
        summary["best_epoch_effect_full"] = {
            "last": float(f.acc_last.mean()), "val_selected": float(f.acc_val_selected.astype(float).mean()),
            "oracle_test_selected": float(f.acc_oracle_best_test.astype(float).mean())}
        sel = ft[(ft.lr_selection_run == 1) & (ft.is_final == 1)]
        summary["selected_lr_counts"] = {r: sel[sel.regime == r].lr.value_counts().to_dict() for r in regimes}

        tgt = read(TARGETS / "v1_table4_finetune.csv")
        if tgt is not None and tgt["mantis"].notna().any():
            ours_ft = f.groupby("dataset").acc_last.agg(["mean", "std"]).reset_index()
            m = ours_ft.merge(tgt[["dataset", "mantis"]], on="dataset").dropna()
            m["delta"] = m["mean"] - m["mantis"]
            summary["finetune_vs_paper"] = paired_test(m["mean"], m["mantis"])
            to_md(m.rename(columns={"mean": "ours", "std": "ours_std", "mantis": "paper"}), "ft_full_vs_paper")
            scatter(m["mantis"], m["mean"], "Full fine-tuning accuracy", "paper (v1 Table 4)", "ours",
                    "ft_scatter.png")

    # calibration
    cal = final.groupby("regime")[["ece_raw", "ece_ts", "ece_irm", "nll_raw", "nll_ts", "temperature"]].mean()
    to_md(cal.reset_index(), "calibration")
    if "full" in regimes:
        f = final[final.regime == "full"].copy()
        f["ts_gain"] = f.ece_raw - f.ece_ts
        small = f.n_val <= 10
        summary["ts_gain_by_val_size"] = {"n_val<=10": float(f[small].ts_gain.mean()),
                                          "n_val>10": float(f[~small].ts_gain.mean())}
        reliability(final)


# ----------------------------------------------------------------- plots --

def scatter(x, y, title, xl, yl, fname):
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.scatter(x, y, s=10, alpha=0.7)
    ax.plot([0, 1], [0, 1], lw=0.8, color="grey")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xl, ylabel=yl, title=title)
    ax.title.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=300)
    plt.close(fig)


def reliability(final, n_plots=3, n_bins=10):
    """Reliability diagrams (raw vs temperature-scaled) for the full regime, seed 0,
    on the tier-1 datasets with the largest test sets."""
    import matplotlib.pyplot as plt
    from src.calibration import fit_temperature
    f = final[(final.regime == "full") & (final.seed == final.seed.min())]
    picks = f.sort_values("n_test", ascending=False).head(n_plots)
    if picks.empty:
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, picks.iterrows()):
        path = RESULTS_DIR / "finetune_history" / f"{r.dataset}__full__s{int(r.seed)}__lr{float(r.lr):g}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        t = fit_temperature(z["logits_val"], z["y_val"])
        for label, p in [("raw", softmax(z["logits_test"])), (f"TS (T={t:.2f})", softmax(z["logits_test"] / t))]:
            conf, correct = p.max(1), (p.argmax(1) == z["y_test"])
            edges = np.linspace(0, 1, n_bins + 1)
            idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
            xs = [conf[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
            ys = [correct[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
            ax.plot(xs, ys, marker="o", ms=3, label=f"{label}, ECE {ece(p, z['y_test']):.3f}")
        ax.plot([0, 1], [0, 1], lw=0.8, color="grey")
        ax.set(title=r.dataset, xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "reliability_full.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------- headline table --

README_START, README_END = "<!-- RESULTS-TABLE:START -->", "<!-- RESULTS-TABLE:END -->"


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "pending"
    return f"{x * 100:+.2f} pp" if pct else f"{x:.4f}"


def headline():
    """Paper-vs-ours table for the README (Section 5 requirement). Means are
    taken over the datasets present in BOTH our results and the paper's table."""
    rows = []
    zs = read(RESULTS_DIR / "zero_shot.csv")
    t3 = read(TARGETS / "v1_table3_zeroshot.csv")
    if zs is not None:
        zs = zs[(zs.layer == -1) & (zs.token == "cls_token")]
        ours = zs.groupby(["dataset", "archive"]).accuracy.mean().reset_index()
    for arch, label in [("ucr", "Zero-shot, UCR"), ("uea", "Zero-shot, UEA"), ("other", "Zero-shot, other 4")]:
        paper = mine = p = None
        n = 0
        if zs is not None and t3 is not None:
            m = ours[ours.archive == arch].merge(t3[["dataset", "mantis"]], on="dataset").dropna()
            n = len(m)
            if n:
                paper, mine = m.mantis.mean(), m.accuracy.mean()
                p = paired_test(m.accuracy, m.mantis).get("wilcoxon_p")
        rows.append({"Setting": f"{label} (RF200, last layer, CLS)", "Datasets": n, "Paper (v1)": fmt(paper),
                     "Ours": fmt(mine), "Δ": fmt(None if paper is None else mine - paper, pct=True),
                     "Wilcoxon p": "" if p is None else f"{p:.3f}"})
    if zs is not None:
        u = ours[ours.archive == "ucr"]
        rows.append({"Setting": "Zero-shot, UCR-128 cross-check (arXiv 2602.17868)", "Datasets": len(u),
                     "Paper (v1)": f"{CROSS_CHECK_UCR128:.4f}", "Ours": fmt(u.accuracy.mean() if len(u) else None),
                     "Δ": fmt(u.accuracy.mean() - CROSS_CHECK_UCR128 if len(u) else None, pct=True),
                     "Wilcoxon p": ""})

    ft = read(RESULTS_DIR / "finetune.csv")
    t4 = read(TARGETS / "v1_table4_finetune.csv")
    if ft is not None:
        final = ft[ft.is_final == 1]
        full = final[final.regime == "full"].groupby("dataset").acc_last.mean().reset_index()
        paper = mine = p = None
        n = 0
        if t4 is not None and len(full):
            m = full.merge(t4[["dataset", "mantis"]], on="dataset").dropna()
            n = len(m)
            if n:
                paper, mine = m.mantis.mean(), m.acc_last.mean()
                p = paired_test(m.acc_last, m.mantis).get("wilcoxon_p")
        rows.append({"Setting": "Full fine-tuning (tier 1)", "Datasets": n, "Paper (v1)": fmt(paper),
                     "Ours": fmt(mine), "Δ": fmt(None if paper is None else mine - paper, pct=True),
                     "Wilcoxon p": "" if p is None else f"{p:.3f}"})
        g = summary.get("full_minus_scratch", {})
        for key, pkey, label in [("ours_last", "paper_last", "last epoch"),
                                 ("ours_best_oracle", "paper_best", "best epoch")]:
            rows.append({"Setting": f"Full minus scratch, {label}", "Datasets": g.get("test_last", {}).get("n", 0),
                         "Paper (v1)": fmt(PAPER_FULL_MINUS_SCRATCH[label.split()[0]], pct=True),
                         "Ours": fmt(g.get(key), pct=True), "Δ": "", "Wilcoxon p": ""})
    mm = summary.get("baselines", {}).get("mantis_minus_moment")
    if mm:
        rows.append({"Setting": "Zero-shot, Mantis minus MOMENT", "Datasets": mm["n"],
                     "Paper (v1)": fmt(mm["paper"], pct=True), "Ours": fmt(mm["ours"], pct=True),
                     "Δ": "", "Wilcoxon p": ""})
    if not rows:
        return
    table = pd.DataFrame(rows).to_markdown(index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "headline.md").write_text(table)
    print("\n### headline (also written into README.md)\n" + table)
    readme = RESULTS_DIR.parent / "README.md"  # the repo README (absent in the smoke test)
    if not readme.exists():
        return
    text = readme.read_text()
    if README_START in text and README_END in text:
        pre, rest = text.split(README_START, 1)
        _, post = rest.split(README_END, 1)
        readme.write_text(f"{pre}{README_START}\n{table}\n{README_END}{post}")


def main():
    ours = zero_shot()
    baselines(ours)
    finetune()
    headline()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nsummary.json:\n" + json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
