"""Turn results/*.csv into the report's tables and figures.

Writes markdown tables to results/tables/, figures to results/figures/ and a
machine-readable results/summary.json. Every number in Sections 3 and 5 (README
results table) should come from here, never typed by hand.

Paper targets are read from targets/*.csv (transcribed from v1's appendix; see
targets/README.md). Missing target files are skipped.

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
from configs.datasets import TIER1  # noqa: E402

TABLES, FIGS = RESULTS_DIR / "tables", RESULTS_DIR / "figures"
TARGETS = ROOT / "targets"
CROSS_CHECK_UCR128 = 0.7921  # original Mantis, RF, UCR-128 (arXiv 2602.17868, Fig. 9b / Table 1)
PAPER_FULL_MINUS_SCRATCH = {"last": 0.0508, "best": 0.0392}  # v1, Figure 7 (151 datasets)
PAPER_ECE_MANTIS = {"raw": 0.1131, "ts": 0.0673, "irm": 0.0609}  # v1, Figure 9 (131 datasets)
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
        # break the gaps down by series type (unequal-length and long series are
        # resized to 512, and v1 does not say how it handled unequal lengths)
        m["abs_delta"] = m["delta"].abs()
        m["group"] = np.where(m.unequal_length == 1, "unequal length",
                              np.where(m.max_len > 1000, "equal, longer than 1000", "equal, up to 1000"))
        grp = m.groupby("group").agg(datasets=("dataset", "count"), mean_abs_delta=("abs_delta", "mean"),
                                     over_5pp=("abs_delta", lambda x: int((x > 0.05).sum())),
                                     within_1pp=("abs_delta", lambda x: float((x <= 0.01).mean())))
        to_md(grp.reset_index(), "zs_gaps_by_series_type")
        scatter_by_type(m)
        summary["zero_shot"]["vs_paper"]["within_1pp"] = float((m.abs_delta <= 0.01).mean())
        summary["zero_shot"]["vs_paper"]["over_5pp"] = int((m.abs_delta > 0.05).sum())

        # sensitivity check: aeon's pre-equalised "_eq" files for unequal-length datasets
        eq = read(RESULTS_DIR / "zero_shot_aeon_eq.csv")
        if eq is not None:
            eq = eq[(eq.layer == -1) & (eq.token == "cls_token")].groupby("dataset").accuracy.mean()
            c = m[m.dataset.isin(eq.index)][["dataset", "paper", "ours"]].copy()
            c["ours_aeon_eq"] = c.dataset.map(eq)
            c["gap_raw"] = (c.ours - c.paper).abs()
            c["gap_eq"] = (c.ours_aeon_eq - c.paper).abs()
            to_md(c.sort_values("gap_raw", ascending=False), "zs_unequal_length_check")
            summary["zero_shot"]["aeon_eq_check"] = {
                "n": int(len(c)), "mean_abs_gap_raw": float(c.gap_raw.mean()),
                "mean_abs_gap_eq": float(c.gap_eq.mean()),
                "closer_with_eq": int((c.gap_eq < c.gap_raw).sum())}
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
    final_all = ft[ft.is_final == 1].copy()
    final_all["tier"] = np.where(final_all.dataset.isin(TIER1), 1, 2)
    # Regime comparisons, calibration and LR selection use tier 1 only: tier 2 has
    # only the full regime, one seed and a fixed LR, so pooling it would compare
    # 'full' on 113 datasets against the other regimes on 30.
    final = final_all[final_all.tier == 1]
    ft = ft[ft.dataset.isin(TIER1)]
    regimes = [r for r in ["rf", "head", "full", "scratch", "full_pkg_defaults"] if r in set(final.regime)]
    tiers(final_all)

    # accuracy per regime (last epoch), averaged over seeds then datasets
    rows = []
    for r in regimes:
        d = final[final.regime == r].groupby("dataset")[["acc_last", "acc_val_selected",
                                                         "acc_oracle_best_test"]].mean()
        rows.append({"regime": r, "datasets": len(d), **d.mean(numeric_only=True).to_dict()})
    to_md(pd.DataFrame(rows), "ft_regimes")  # tier 1 only

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
            # v1 Table 4 reports best-epoch accuracy, so compare like with like and
            # also show our last-epoch and validation-selected numbers.
            # Both tiers: tier 1 averaged over 3 seeds, tier 2 is seed 0 only.
            cols = ["acc_oracle_best_test", "acc_val_selected", "acc_last"]
            ours_ft = final_all[final_all.regime == "full"].groupby(["dataset", "tier"])[cols] \
                .mean().reset_index()
            m = ours_ft.merge(tgt[["dataset", "mantis"]], on="dataset").dropna()
            summary["finetune_vs_paper"] = {c: paired_test(m[c], m["mantis"]) for c in cols}
            rows = [{"set": lab, "datasets": len(g), "paper (best epoch)": g.mantis.mean(),
                     **{f"ours {c}": g[c].mean() for c in cols}}
                    for lab, g in [("tier 1", m[m.tier == 1]), ("tier 2", m[m.tier == 2]), ("both", m)]]
            to_md(pd.DataFrame(rows), "ft_full_vs_paper")
            to_md(m.rename(columns={"mantis": "paper"}), "ft_full_vs_paper_per_dataset")
            scatter(m["mantis"], m["acc_oracle_best_test"], "Full fine-tuning, best epoch",
                    "paper (v1 Table 4)", "ours", "ft_scatter.png")

        # regimes against v1 Tables 6 and 7 on the same tier-1 datasets
        t6, t7 = read(TARGETS / "v1_table6_regimes_last.csv"), read(TARGETS / "v1_table7_regimes_best.csv")
        if t6 is not None and t7 is not None:
            rows = []
            for r in [x for x in ["rf", "head", "scratch", "full"] if x in regimes]:
                last = wide["acc_last"][r].rename("ours_last").reset_index()
                best = wide["acc_oracle_best_test"][r].rename("ours_best").reset_index()
                g = last.merge(best, on="dataset").merge(t6[["dataset", r]].rename(columns={r: "paper_last"}),
                                                         on="dataset").merge(
                    t7[["dataset", r]].rename(columns={r: "paper_best"}), on="dataset")
                rows.append({"regime": r, "datasets": len(g), "paper_last": g.paper_last.mean(),
                             "ours_last": g.ours_last.mean(), "paper_best": g.paper_best.mean(),
                             "ours_best": g.ours_best.mean() if r != "rf" else np.nan})
            reg = pd.DataFrame(rows)
            to_md(reg, "ft_regimes_vs_paper")
            pr = reg.set_index("regime")
            if {"full", "scratch"} <= set(pr.index):
                summary["full_minus_scratch"]["paper_same_datasets_last"] = float(
                    pr.loc["full", "paper_last"] - pr.loc["scratch", "paper_last"])
                summary["full_minus_scratch"]["paper_same_datasets_best"] = float(
                    pr.loc["full", "paper_best"] - pr.loc["scratch", "paper_best"])

    # calibration
    cal = final.groupby("regime")[["ece_raw", "ece_ts", "ece_irm", "nll_raw", "nll_ts", "temperature"]].mean()
    to_md(cal.reset_index(), "calibration")
    if "full" in regimes:
        f = final[final.regime == "full"].copy()
        f["ts_gain"] = f.ece_raw - f.ece_ts
        small = f.n_val <= 10
        summary["ts_gain_by_val_size"] = {"n_val<=10": float(f[small].ts_gain.mean()),
                                          "n_val>10": float(f[~small].ts_gain.mean()),
                                          "n_datasets_small": int(f[small].dataset.nunique()),
                                          "n_datasets_large": int(f[~small].dataset.nunique())}
        # same split over both tiers (full regime, seed 0), for more datasets per bin
        fa = final_all[(final_all.regime == "full") & (final_all.seed == 0)].copy()
        fa["ts_gain"] = fa.ece_raw - fa.ece_ts
        sm = fa.n_val <= 10
        summary["ts_gain_by_val_size_both_tiers_seed0"] = {
            "n_val<=10": float(fa[sm].ts_gain.mean()), "n_val>10": float(fa[~sm].ts_gain.mean()),
            "n_datasets_small": int(sm.sum()), "n_datasets_large": int((~sm).sum())}
        reliability(final)


def tiers(final_all):
    """Full fine-tuning on tier 1 vs tier 2, and fine-tuning gain over zero-shot RF."""
    f = final_all[final_all.regime == "full"]
    if f.empty:
        return
    zs = read(RESULTS_DIR / "zero_shot.csv")
    zs_mean = None
    if zs is not None:
        zs = zs[(zs.layer == -1) & (zs.token == "cls_token")] if {"layer", "token"} <= set(zs.columns) else zs
        zs_mean = zs.groupby("dataset").accuracy.mean()
    rows, out = [], {}
    groups = [("tier 1 (3 seeds, LR selected)", f[f.tier == 1], None),
              ("tier 1, seed 0 only", f[(f.tier == 1) & (f.seed == 0)], None),
              ("tier 2 (seed 0, LR 1e-3)", f[f.tier == 2], None),
              ("tiers 1+2 (seed 0)", f[f.seed == 0], None)]
    for label, g, _ in groups:
        d = g.groupby("dataset").acc_last.mean()
        row = {"set": label, "datasets": len(d), "full_acc_last": d.mean()}
        if zs_mean is not None:
            t = paired_test(d, zs_mean.reindex(d.index))
            row.update({"zero_shot_rf": zs_mean.reindex(d.index).mean(), "gain": t["mean_diff"],
                        "wins/ties/losses": f"{t['wins']}/{t['ties']}/{t['losses']}",
                        "wilcoxon_p": t.get("wilcoxon_p", np.nan)})
            out[label] = t
        rows.append(row)
    to_md(pd.DataFrame(rows), "ft_full_by_tier")
    summary["full_vs_zero_shot_by_tier"] = out


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



def scatter_by_type(m: pd.DataFrame, n_labels: int = 5):
    """Report figure: paper vs ours zero-shot accuracy, marked by series type."""
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.plot([0, 1], [0, 1], lw=0.8, color="#55554f", zorder=1)
    for off in (-0.05, 0.05):
        ax.plot([0, 1], [off, 1 + off], lw=0.6, ls=(0, (3, 3)), color="#a3a29a", zorder=1)
    styles = [("equal, up to 1000", "#2a78d6", "o", 14), ("equal, longer than 1000", "#eb6834", "s", 18),
              ("unequal length", "#1baf7a", "^", 22)]
    for grp, col, mk, size in styles:
        g = m[m.group == grp]
        ax.scatter(g.paper, g.ours, s=size, marker=mk, color=col, edgecolor="white", linewidth=0.5,
                   label=f"{grp} ({len(g)})", zorder=3)
    # label the largest gaps in the empty corners, with thin leader lines
    slots = {True: [(0.62, 0.50), (0.62, 0.43), (0.62, 0.36)],     # below the diagonal
             False: [(0.20, 0.97), (0.20, 0.90), (0.20, 0.83)]}    # above the diagonal
    top = m.sort_values("abs_delta", ascending=False).head(n_labels)
    groups = {False: top[top.delta >= 0].sort_values("ours", ascending=False),   # above: by height
              True: top[top.delta < 0].sort_values("paper")}                      # below: left to right
    for below, rows in groups.items():
        for (_, r), xy in zip(rows.iterrows(), slots[below]):  # leader lines never cross
            ax.annotate(r.dataset, (r.paper, r.ours), xytext=xy, textcoords="data", fontsize=6,
                        color="#3d3d3a", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#8a8980", shrinkA=1, shrinkB=2))
    ax.set(xlim=(0.15, 1.02), ylim=(0.15, 1.02), xlabel="v1 Accuracy", ylabel="Reproducibility Accuracy (3 seeds)")
    ax.grid(color="#e6e5e0", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "zs_scatter_by_type.pdf")
    fig.savefig(FIGS / "zs_scatter_by_type.png", dpi=300)
    plt.close(fig)


def reliability(final, n_plots=3, n_bins=10):
    """Reliability diagrams (raw vs temperature-scaled) for the full regime, seed 0,
    on the tier-1 datasets with the largest test sets."""
    import matplotlib.pyplot as plt
    from src.calibration import fit_temperature
    f = final[(final.regime == "full") & (final.seed == final.seed.min()) & final.dataset.isin(TIER1)]
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


# ------------------------------------------------------- report numbers --

def report_numbers():
    """Every number quoted in Sections 2 and 3 that is not already in another table.
    Written to results/tables/report_numbers.md and summary.json["report_numbers"]."""
    from configs.datasets import TIER1
    zs = read(RESULTS_DIR / "zero_shot.csv")
    ft = read(RESULTS_DIR / "finetune.csv")
    mr = read(RESULTS_DIR / "minirocket.csv")
    mo = read(RESULTS_DIR / "moment.csv")
    t3, t8 = read(TARGETS / "v1_table3_zeroshot.csv"), read(TARGETS / "v1_table8_uea.csv")
    stats = read(RESULTS_DIR / "dataset_stats_all.csv")
    if zs is None or ft is None:
        return
    zs = zs[(zs.layer == -1) & (zs.token == "cls_token")]
    mantis = zs.groupby("dataset").accuracy.mean()
    arch = zs.drop_duplicates("dataset").set_index("dataset").archive
    rows = []

    def add(section, statement, value):
        rows.append({"section": section, "statement": statement, "value": value})

    final = ft[ft.is_final == 1]
    full0 = final[(final.regime == "full") & (final.seed == 0)].set_index("dataset")

    if mo is not None and t3 is not None:
        moment = mo.groupby("dataset").accuracy.mean()
        m = t3.set_index("dataset")[["moment"]].join(moment.rename("ours"), how="inner").dropna()
        t = paired_test(m.ours, m.moment)
        add("III-B", "MOMENT zero-shot, ours vs v1 Table 3",
            f"{m.ours.mean():.4f} vs {m.moment.mean():.4f} (n={len(m)}, p={t.get('wilcoxon_p', np.nan):.2f})")
        u = [d for d in moment.index if arch.get(d) == "ucr" and d in mantis.index]
        t = paired_test(mantis[u], moment[u])
        add("III-B", "Mantis minus MOMENT on UCR only",
            f"{100 * t['mean_diff']:+.2f} pp (n={t['n']}, p={t.get('wilcoxon_p', np.nan):.3f})")
        secs_mo = mo.drop_duplicates("dataset").embed_seconds.sum()
        secs_ma = zs.drop_duplicates("dataset").embed_seconds.sum()
        add("III-B", "Embedding time, Mantis vs MOMENT (all datasets)",
            f"{secs_ma / 60:.1f} min vs {secs_mo / 3600:.1f} h ({secs_mo / secs_ma:.0f} times faster)")
    if mr is not None:
        minirocket = mr.groupby("dataset").accuracy.mean()
        c = pd.concat([mantis.rename("mantis"), minirocket.rename("mr")], axis=1).dropna()
        add("III-B", "MiniRocket vs zero-shot Mantis",
            f"{c.mr.mean():.4f} vs {c.mantis.mean():.4f} (n={len(c)})")
        add("III-B", "MiniRocket total run time, all datasets and seeds", f"{mr.seconds.sum() / 60:.0f} min")
        f = pd.concat([full0.acc_last.rename("ft"), minirocket.rename("mr")], axis=1).dropna()
        t = paired_test(f.ft, f.mr)
        add("III-B", "Fine-tuned Mantis (Full, last epoch, seed 0) vs MiniRocket",
            f"{f.ft.mean():.4f} vs {f.mr.mean():.4f} (n={t['n']}; MiniRocket better on {t['losses']}, "
            f"Mantis on {t['wins']}, p={t.get('wilcoxon_p', np.nan):.4f})")
        if "EthanolLevel" in c.index:
            add("III-B", "EthanolLevel, MiniRocket vs Mantis zero-shot",
                f"{c.loc['EthanolLevel', 'mr']:.2f} vs {c.loc['EthanolLevel', 'mantis']:.2f}")
    if t8 is not None:
        u = mantis[[d for d in t8.dataset if d in mantis.index]]
        g = t8.set_index("dataset").rf.reindex(u.index)
        g3 = t3.set_index("dataset").mantis.reindex(u.index) if t3 is not None else g
        add("III-A", "UEA: mean |ours - v1 Table 3| vs mean |ours - v1 Table 8|",
            f"{100 * (u - g3).abs().mean():.1f} pp vs {100 * (u - g).abs().mean():.1f} pp "
            f"(Table 8 mean {g.mean():.4f}, Table 3 mean {g3.mean():.4f})")
    eq = read(RESULTS_DIR / "zero_shot_aeon_eq.csv")
    if eq is not None:
        e = eq[(eq.layer == -1) & (eq.token == "cls_token")].groupby("dataset").accuracy.mean()
        ch = (e - mantis.reindex(e.index)).abs()
        add("III-A", "Accuracy change from loading aeon's _eq files (15 datasets)",
            f"mean {100 * ch.mean():.1f} pp, max {100 * ch.max():.1f} pp ({ch.idxmax()})")

    t1 = final[final.dataset.isin(TIER1)]
    w = t1.groupby(["dataset", "regime"]).acc_last.mean().unstack()
    if {"full", "scratch"} <= set(w.columns) and stats is not None:
        gap = (w.full - w.scratch)
        ntr = stats.set_index("name").n_train.reindex(gap.index)
        add("III-D", "Full minus Scratch (last epoch), at most 40 training cases vs the rest",
            f"{100 * gap[ntr <= 40].mean():+.1f} pp (n={int((ntr <= 40).sum())}) vs "
            f"{100 * gap[ntr > 40].mean():+.1f} pp (n={int((ntr > 40).sum())}); "
            f"largest {100 * gap.max():+.1f} pp ({gap.idxmax()})")
    pk = final[final.regime == "full_pkg_defaults"].set_index("dataset")
    if len(pk):
        base = t1[t1.regime == "full"].groupby("dataset")[["acc_last", "fit_seconds"]].mean().reindex(pk.index)
        add("III-C", "Package defaults minus paper protocol (Full, last epoch)",
            f"{100 * (pk.acc_last - base.acc_last).mean():+.1f} pp on {len(pk)} datasets, "
            f"{(pk.fit_seconds / base.fit_seconds).mean():.1f} times the run time")
    c = full0[["ece_raw", "ece_ts", "ece_irm", "temperature"]]
    add("III-E", "ECE raw / TS / IRM, Full, seed 0, both tiers (v1 Figure 9)",
        f"{c.ece_raw.mean():.3f} / {c.ece_ts.mean():.3f} / {c.ece_irm.mean():.3f} (n={len(c)}) vs v1 "
        f"{PAPER_ECE_MANTIS['raw']:.3f} / {PAPER_ECE_MANTIS['ts']:.3f} / {PAPER_ECE_MANTIS['irm']:.3f}")
    add("III-E", "Fitted temperature above 1, and TS raising ECE",
        f"T > 1 on {int((c.temperature > 1).sum())} of {len(c)} (mean {c.temperature.mean():.2f}); "
        f"TS worse on {int((c.ece_ts > c.ece_raw).sum())}")
    add("II-F", "Fine-tuning run time (all runs)", f"{len(ft)} runs, {ft.fit_seconds.sum() / 3600:.1f} h")

    df = pd.DataFrame(rows)
    to_md(df, "report_numbers")
    summary["report_numbers"] = {f"{r.section} {r.statement}": r.value for r in df.itertuples()}


# ------------------------------------------------------- headline table --

README_START, README_END = "<!-- RESULTS-TABLE:START -->", "<!-- RESULTS-TABLE:END -->"


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "pending"
    return f"{x * 100:+.2f} pp" if pct else f"{x:.4f}"


def headline():
    """Paper-vs-ours table for the README (Section 5 requirement). Means are
    taken over the datasets present in both our results and the paper's table."""
    rows = []
    zs = read(RESULTS_DIR / "zero_shot.csv")
    t3 = read(TARGETS / "v1_table3_zeroshot.csv")
    if zs is not None:
        zs = zs[(zs.layer == -1) & (zs.token == "cls_token")]
        ours = zs.groupby(["dataset", "archive"]).accuracy.mean().reset_index()
    for arch, label in [("ucr", "Zero-shot, UCR"), ("uea", "Zero-shot, UEA"), ("other", "Zero-shot, other (3 of 4)")]:
        paper = mine = p = None
        n = 0
        if zs is not None:
            a = ours[ours.archive == arch]
            n, mine = len(a), (a.accuracy.mean() if len(a) else None)  # ours even before targets exist
        if zs is not None and t3 is not None and t3["mantis"].notna().any():
            m = ours[ours.archive == arch].merge(t3[["dataset", "mantis"]], on="dataset").dropna()
            n = len(m)
            if n:
                paper, mine = m.mantis.mean(), m.accuracy.mean()
                p = paired_test(m.accuracy, m.mantis).get("wilcoxon_p")
        rows.append({"Setting": f"{label} (RF200, last layer, CLS)", "Datasets": n, "Paper (v1)": fmt(paper),
                     "Ours": fmt(mine), "Δ": fmt(None if paper is None else mine - paper, pct=True),
                     "Wilcoxon p": pfmt(p)})
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
        full_all = final[final.regime == "full"]
        # tier 1 is averaged over its 3 seeds, tier 2 has seed 0 only (same as ft_full_vs_paper)
        for label, sub in [("Full fine-tuning, tier 1 (3 seeds)", full_all[full_all.dataset.isin(TIER1)]),
                           ("Full fine-tuning, tiers 1 and 2", full_all)]:
            for col, how in [("acc_oracle_best_test", "best epoch (ours chosen on test)"),
                             ("acc_val_selected", "best epoch (ours chosen on validation)"),
                             ("acc_last", "ours at last epoch")]:
                full = sub.groupby("dataset")[col].mean().reset_index()
                paper, mine, p = None, (full[col].mean() if len(full) else None), None
                n = len(full)
                if t4 is not None and t4["mantis"].notna().any() and len(full):
                    m = full.merge(t4[["dataset", "mantis"]], on="dataset").dropna()
                    n = len(m)
                    if n:
                        paper, mine = m.mantis.mean(), m[col].mean()
                        p = paired_test(m[col], m.mantis).get("wilcoxon_p")
                rows.append({"Setting": f"{label}, {how}", "Datasets": n, "Paper (v1)": fmt(paper),
                             "Ours": fmt(mine), "Δ": fmt(None if paper is None else mine - paper, pct=True),
                             "Wilcoxon p": pfmt(p)})
        g = summary.get("full_minus_scratch", {})
        for key, pkey, label in [("ours_last", "paper_last", "last epoch"),
                                 ("ours_best_oracle", "paper_best", "best epoch")]:
            rows.append({"Setting": f"Full minus scratch, {label}", "Datasets": g.get("test_last", {}).get("n", 0),
                         "Paper (v1)": fmt(g.get("paper_same_datasets_" + label.split()[0],
                                                 PAPER_FULL_MINUS_SCRATCH[label.split()[0]]), pct=True),
                         "Ours": fmt(g.get(key), pct=True), "Δ": "", "Wilcoxon p": ""})
        # calibration of Full fine-tuning (seed 0, both tiers) against v1 Figure 9
        f0 = full_all[full_all.seed == 0]
        for key, col, label in [("raw", "ece_raw", "no post-hoc calibration"),
                                ("ts", "ece_ts", "temperature scaling"), ("irm", "ece_irm", "IRM")]:
            if col in f0 and len(f0):
                mine = f0[col].mean()
                rows.append({"Setting": f"ECE, Full fine-tuning (seed 0), {label}", "Datasets": len(f0),
                             "Paper (v1)": f"{PAPER_ECE_MANTIS[key]:.4f}", "Ours": fmt(mine),
                             "Δ": fmt(mine - PAPER_ECE_MANTIS[key], pct=True), "Wilcoxon p": ""})
    mo_csv, t3m = read(RESULTS_DIR / "moment.csv"), read(TARGETS / "v1_table3_zeroshot.csv")
    if mo_csv is not None and t3m is not None and t3m["moment"].notna().any():
        m = mo_csv.groupby("dataset").accuracy.mean().rename("ours").reset_index().merge(
            t3m[["dataset", "moment"]], on="dataset").dropna()
        rows.append({"Setting": "Zero-shot, MOMENT-1-large (RF200)", "Datasets": len(m),
                     "Paper (v1)": fmt(m.moment.mean()), "Ours": fmt(m.ours.mean()),
                     "Δ": fmt(m.ours.mean() - m.moment.mean(), pct=True),
                     "Wilcoxon p": pfmt(paired_test(m.ours, m.moment).get("wilcoxon_p"))})
    mm = summary.get("baselines", {}).get("mantis_minus_moment")
    mo = summary.get("baselines", {}).get("MOMENT-1-large")
    if mm and mm["n"]:
        rows.append({"Setting": "Zero-shot, Mantis minus MOMENT", "Datasets": mm["n"],
                     "Paper (v1)": fmt(mm["paper"], pct=True), "Ours": fmt(mm["ours"], pct=True),
                     "Δ": "", "Wilcoxon p": f"{mo['wilcoxon_p']:.3f}" if mo and "wilcoxon_p" in mo else ""})
    elif mo:  # no paper targets yet: show ours over every dataset both models ran on
        rows.append({"Setting": "Zero-shot, Mantis minus MOMENT", "Datasets": mo["n"],
                     "Paper (v1)": fmt(None), "Ours": fmt(mo["mean_diff"], pct=True),
                     "Δ": "", "Wilcoxon p": f"{mo['wilcoxon_p']:.3f}" if "wilcoxon_p" in mo else ""})
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
    report_numbers()
    headline()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nsummary.json:\n" + json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
