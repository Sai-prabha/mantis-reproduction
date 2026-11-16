from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

PRACTICAL = 0.01


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def compare(a: pd.Series, b: pd.Series, higher_is_better: bool = True) -> dict:
    d = (a - b).dropna()
    if not higher_is_better:
        d = -d
    out = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
           "gain_>1pp": int((d > PRACTICAL).sum()), "loss_>1pp": int((d < -PRACTICAL).sum())}
    out["p"] = float(wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() >= 1 else 1.0
    return out


def comparison_table(acc: pd.DataFrame, pairs: list[tuple[str, str, str]],
                     higher_is_better: bool = True) -> pd.DataFrame:
    rows = []
    for a, b, question in pairs:
        if a in acc and b in acc:
            rows.append({"comparison": f"{a} vs {b}", "question": question,
                         **compare(acc[a], acc[b], higher_is_better)})
    if not rows:
        return pd.DataFrame()
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = pfmt(q)
        r["p"] = pfmt(r["p"])
        r["mean_diff_pp"] = 100 * r.pop("mean_diff")
    cols = ["comparison", "question", "n", "mean_diff_pp", "wins", "ties", "losses",
            "gain_>1pp", "loss_>1pp", "p", "p_holm"]
    return pd.DataFrame(rows)[cols]


def pfmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def mean_ranks(acc: pd.DataFrame) -> pd.Series:
    return acc.rank(axis=1, ascending=False, method="average").mean()


def friedman(acc: pd.DataFrame) -> dict:
    if acc.shape[1] < 3:
        return {"chi2": float("nan"), "p": float("nan"), "k": acc.shape[1], "n": acc.shape[0]}
    stat, p = friedmanchisquare(*[acc[c].values for c in acc.columns])
    return {"chi2": float(stat), "p": float(p), "k": acc.shape[1], "n": acc.shape[0]}


def to_md(df: pd.DataFrame, path: Path, floatfmt: str = ".4f", index: bool = False) -> None:
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".md")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_markdown(index=index, floatfmt=floatfmt)
    path.write_text(text, encoding="utf-8")
    print(f"\n### {path.stem}\n{text}")


def cd_diagram(acc: pd.DataFrame, path: Path) -> None:
    if acc.shape[1] < 3:
        return
    try:
        from aeon.visualisation import plot_critical_difference
        res = plot_critical_difference(acc.values, list(acc.columns), test="wilcoxon", correction="holm")
        fig = res[0] if isinstance(res, tuple) else res
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip] CD diagram {path.name}: {type(e).__name__}: {e}")
