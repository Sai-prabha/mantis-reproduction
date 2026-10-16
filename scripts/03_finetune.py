"""Fine-tuning reproduction of Mantis-8M (arXiv 2502.15637v1).

Reproduces, on a stated subset of datasets:
  * fine-tuned accuracy (v1 Figure 5, Appendix Table 4)
  * pretrained vs from-scratch training (v1 Figure 7, Tables 6/7: last and best epoch)
  * calibration of fine-tuned models (v1 Figure 9: ECE, temperature scaling, IRM)

Regimes
  rf      random forest (200 trees) on frozen embeddings     (no network training)
  head    frozen encoder, train LayerNorm+Linear head
  full    pretrained encoder + head, all weights trained      (v1 "full fine-tuning")
  scratch randomly initialised encoder + head, all trained   (v1 "scratch")

Protocol (v1 Section 3.3), implemented with the package's own FineTuningNetwork
and learning-rate scheduler, but with the paper's settings rather than the
package defaults:
  100 epochs, batch 256, AdamW (weight decay 0.05), LayerNorm+Linear head,
  learning rate chosen from {1e-4, 2e-4, 1e-3} on a 20% validation split.
Our documented choices where v1 is silent:
  * LR is selected on seed 0 only (best final-epoch validation accuracy,
    ties broken by validation NLL); seeds 1..n reuse the selected LR.
  * The model is trained on the 80% split and not retrained on 100%, because
    the 20% split is needed to fit the calibration maps.
  * The 80/20 split changes with the seed (stratified where possible).
  * "Best epoch" is reported two ways: chosen on validation accuracy, and chosen
    on test accuracy (an oracle, since v1 does not say which it used).

Usage
  python scripts/03_finetune.py --datasets tier1
  python scripts/03_finetune.py --datasets tier2 --regimes full --seeds 0 --fixed-lr 0.001
  python scripts/03_finetune.py --datasets ECG200,GunPoint --package-defaults
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (CHECKPOINT_REVISION, RESULTS_DIR, CsvLog, env_info,  # noqa: E402
                        pick_device, require_env, set_seed, sync)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from configs.datasets import resolve  # noqa: E402
from src.calibration import calibration_report, nll, softmax  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.models import cached_embeddings, load_mantis, paper_head  # noqa: E402

LR_GRID = (1e-4, 2e-4, 1e-3)
CSV_PATH = RESULTS_DIR / "finetune.csv"
HISTORY_DIR = RESULTS_DIR / "finetune_history"
KEY = ["dataset", "regime", "seed", "lr"]


# ---------------------------------------------------------------- helpers --

def split_train_val(y: np.ndarray, seed: int, frac: float = 0.2):
    idx = np.arange(len(y))
    n_val = max(int(round(frac * len(y))), 1)
    try:
        tr, va = train_test_split(idx, test_size=n_val, stratify=y, random_state=seed)
        stratified = 1
    except ValueError:  # a class with one case, or fewer validation slots than classes
        tr, va = train_test_split(idx, test_size=n_val, random_state=seed)
        stratified = 0
    return np.sort(tr), np.sort(va), stratified


@torch.no_grad()
def predict_logits(model, X: torch.Tensor, device: str, batch: int = 512) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + batch].to(device)).float().cpu() for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(1) == y).mean())


def set_train_mode(model, regime: str) -> None:
    # Mirrors MantisTrainer._set_train: frozen encoder stays in eval mode for "head".
    if regime == "head":
        model.eval()
        model.head.train()
    else:
        model.train()


# --------------------------------------------------------------- training --

def train_network(regime, Xtr, ytr, Xva, yva, Xte, yte, n_classes, n_channels,
                  lr, seed, args, device, frozen=None):
    """One training run. Returns metrics plus final validation/test logits."""
    from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
    from mantis.trainer.trainer_utils.scheduling import adjust_learning_rate

    set_seed(seed)
    if regime == "head":
        Ztr, Zva, Zte = frozen  # precomputed frozen embeddings for this split
        head = paper_head(Ztr.shape[1], n_classes).to(device)
        model = FineTuningNetwork(None, head).to(device)
        params = list(head.parameters())
        tr_x, va_x, te_x = (torch.tensor(a) for a in (Ztr, Zva, Zte))
    else:
        encoder = load_mantis(device, pretrained=(regime == "full"), seed=seed)
        head = paper_head(encoder.hidden_dim * n_channels, n_classes).to(device)
        model = FineTuningNetwork(encoder, head).to(device)
        params = list(model.parameters())
        tr_x, va_x, te_x = (torch.tensor(a) for a in (Xtr, Xva, Xte))

    optimiser = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
    dataset = TensorDataset(tr_x, torch.tensor(ytr, dtype=torch.long))
    bs = args.batch_size
    drop_last = len(dataset) > bs and len(dataset) % bs == 1  # same rule as the package
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=drop_last,
                        generator=torch.Generator().manual_seed(seed))
    criterion = nn.CrossEntropyLoss()

    history, step = [], 1
    sync(device)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, regime)
        for xb, yb in loader:
            adjust_learning_rate(args.epochs, optimiser, loader, step, lr)
            step += 1
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            lv, lt = predict_logits(model, va_x, device), predict_logits(model, te_x, device)
            history.append({"epoch": epoch, "val_acc": accuracy(lv, yva),
                            "val_nll": nll(softmax(lv), yva), "test_acc": accuracy(lt, yte)})
    sync(device)
    seconds = time.perf_counter() - t0

    # lv, lt hold the final-epoch logits (the last epoch is always evaluated)
    h = pd.DataFrame(history)
    by_val = h.loc[h["val_acc"] == h["val_acc"].max()].iloc[-1]   # latest epoch among ties
    by_test = h.loc[h["test_acc"] == h["test_acc"].max()].iloc[0]
    return {
        "acc_last": accuracy(lt, yte),
        "val_acc_last": accuracy(lv, yva),
        "val_nll_last": nll(softmax(lv), yva),
        "acc_val_selected": float(by_val["test_acc"]),
        "best_val_epoch": int(by_val["epoch"]),
        "acc_oracle_best_test": float(by_test["test_acc"]),
        "best_test_epoch": int(by_test["epoch"]),
        "fit_seconds": seconds,
        "_logits_val": lv, "_logits_test": lt, "_history": history,
    }


def run_rf(Ztr, ytr, Zva, Zte, seed):
    """RF on frozen embeddings. RF probabilities can be exactly 0, so clipped
    log-probabilities stand in for logits. Columns follow rf.classes_."""
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
    rf.fit(Ztr, ytr)
    lv = np.log(np.clip(rf.predict_proba(Zva), 1e-12, 1))
    lt = np.log(np.clip(rf.predict_proba(Zte), 1e-12, 1))
    return lv, lt, rf.classes_, time.perf_counter() - t0


def expand_proba_columns(logp: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((logp.shape[0], n_classes), np.log(1e-12))
    out[:, classes] = logp
    return out


def run_package_defaults(Xtr, ytr, Xva, yva, Xte, yte, seed, device):
    """mantis-tsfm 1.1.0 MantisTrainer.fit with its defaults (500 epochs,
    BatchNorm+Linear head, lr 2e-4), for comparison with the paper's settings."""
    from mantis.trainer import MantisTrainer
    set_seed(seed)
    trainer = MantisTrainer(device=device, network=load_mantis(device))
    sync(device)
    t0 = time.perf_counter()
    trainer.fit(Xtr, ytr)
    sync(device)
    seconds = time.perf_counter() - t0
    lv = np.log(np.clip(trainer.predict_proba(Xva), 1e-12, 1))
    lt = np.log(np.clip(trainer.predict_proba(Xte), 1e-12, 1))
    return lv, lt, seconds


# ------------------------------------------------------------------- main --

def main():
    require_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="tier1")
    ap.add_argument("--regimes", default="rf,head,full,scratch")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="evaluate val/test every N epochs (best-epoch granularity)")
    ap.add_argument("--fixed-lr", type=float, default=None,
                    help="skip LR selection and use this LR (tier 2)")
    ap.add_argument("--package-defaults", action="store_true",
                    help="extra regime 'full_pkg_defaults' using MantisTrainer.fit defaults")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--experiment", default="finetune_v1")
    args = ap.parse_args()

    device = pick_device(args.device)
    info = env_info(device)
    print(json.dumps(info, indent=2))
    log = CsvLog(CSV_PATH, KEY)
    regimes = [r.strip() for r in args.regimes.split(",")]
    if args.package_defaults:
        regimes = ["full_pkg_defaults"]
    seeds = sorted(args.seeds)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    for name in resolve(args.datasets):
        try:
            data = load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: could not load ({e})")
            continue
        meta = data["meta"]
        n_classes, n_channels = meta["n_classes"], meta["n_channels"]
        Z_all = None  # frozen embeddings, shared with 02_zero_shot.py through the cache

        for regime in regimes:
            lr_for_seed = {}
            for seed in seeds:
                tr, va, strat = split_train_val(data["y_train"], seed)
                Xtr, ytr = data["X_train"][tr], data["y_train"][tr]
                Xva, yva = data["X_train"][va], data["y_train"][va]
                Xte, yte = data["X_test"], data["y_test"]

                if regime in ("rf", "full_pkg_defaults"):
                    candidate_lrs = [0.0]
                elif args.fixed_lr is not None:
                    candidate_lrs = [args.fixed_lr]
                elif seed == seeds[0]:
                    candidate_lrs = list(LR_GRID)  # selection runs
                elif seeds[0] in lr_for_seed:
                    candidate_lrs = [lr_for_seed[seeds[0]]]
                else:
                    print(f"[skip] {name} {regime} seed {seed}: no LR was selected on seed {seeds[0]}")
                    continue

                frozen = None
                if regime in ("rf", "head"):
                    if Z_all is None:
                        Z_all = cached_embeddings(name, data, device)[:2]
                    frozen = (Z_all[0][tr], Z_all[0][va], Z_all[1])

                seed_rows = []
                for lr in candidate_lrs:
                    if log.done(dataset=name, regime=regime, seed=seed, lr=lr):
                        prev = pd.read_csv(CSV_PATH)
                        prev = prev[(prev.dataset == name) & (prev.regime == regime)
                                    & (prev.seed == seed) & np.isclose(prev.lr, lr)]
                        seed_rows.append(prev.iloc[0].to_dict())
                        continue
                    print(f"[run] {name} | {regime} | seed {seed} | lr {lr}")
                    try:
                        if regime == "rf":
                            lv, lt, classes, secs = run_rf(frozen[0], ytr, frozen[1], frozen[2], seed)
                            lv = expand_proba_columns(lv, classes, n_classes)
                            lt = expand_proba_columns(lt, classes, n_classes)
                            res = {"acc_last": accuracy(lt, yte), "val_acc_last": accuracy(lv, yva),
                                   "val_nll_last": nll(softmax(lv), yva), "fit_seconds": secs,
                                   "_logits_val": lv, "_logits_test": lt, "_history": []}
                        elif regime == "full_pkg_defaults":
                            lv, lt, secs = run_package_defaults(Xtr, ytr, Xva, yva, Xte, yte, seed, device)
                            lv = expand_proba_columns(lv, np.arange(lv.shape[1]), n_classes)
                            lt = expand_proba_columns(lt, np.arange(lt.shape[1]), n_classes)
                            res = {"acc_last": accuracy(lt, yte), "val_acc_last": accuracy(lv, yva),
                                   "val_nll_last": nll(softmax(lv), yva), "fit_seconds": secs,
                                   "_logits_val": lv, "_logits_test": lt, "_history": []}
                        else:
                            res = train_network(regime, Xtr, ytr, Xva, yva, Xte, yte, n_classes,
                                                n_channels, lr, seed, args, device, frozen)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        print(f"[fail] {name} {regime} seed {seed} lr {lr}: {e}")
                        continue

                    finally:
                        if device == "mps":
                            torch.mps.empty_cache()  # long runs: release cached GPU memory

                    cal = calibration_report(res["_logits_val"], yva, res["_logits_test"], yte)
                    stem = f"{name}__{regime}__s{seed}__lr{lr:g}"
                    (HISTORY_DIR / f"{stem}.json").write_text(json.dumps(res["_history"]))
                    # final-epoch logits, for reliability diagrams in 06_summarise.py
                    np.savez_compressed(HISTORY_DIR / f"{stem}.npz", logits_val=res["_logits_val"],
                                        y_val=yva, logits_test=res["_logits_test"], y_test=yte)
                    row = {
                        "experiment": args.experiment, "dataset": name, "regime": regime,
                        "seed": seed, "lr": lr,
                        "lr_selection_run": int(len(candidate_lrs) > 1),
                        "is_final": 0,  # set below once the LR is chosen
                        "epochs": args.epochs if regime in ("head", "full", "scratch") else (
                            500 if regime == "full_pkg_defaults" else 0),
                        "eval_every": args.eval_every,
                        "head": {"head": "LayerNorm+Linear", "full": "LayerNorm+Linear",
                                 "scratch": "LayerNorm+Linear", "rf": "RandomForest200",
                                 "full_pkg_defaults": "BatchNorm1d+Linear"}[regime],
                        "n_train_fit": len(tr), "n_val": len(va), "n_test": len(yte),
                        "stratified_val": strat, "n_classes": n_classes, "n_channels": n_channels,
                        "series_len_min": meta["min_len"], "series_len_max": meta["max_len"],
                        "acc_last": res["acc_last"], "val_acc_last": res["val_acc_last"],
                        "val_nll_last": res["val_nll_last"],
                        "acc_val_selected": res.get("acc_val_selected", ""),
                        "best_val_epoch": res.get("best_val_epoch", ""),
                        "acc_oracle_best_test": res.get("acc_oracle_best_test", ""),
                        "best_test_epoch": res.get("best_test_epoch", ""),
                        **cal,
                        "fit_seconds": round(res["fit_seconds"], 2),
                        "device": device, "torch": info["torch"], "mantis": info["mantis"],
                        "hardware": info["hardware"], "checkpoint_revision": CHECKPOINT_REVISION or "latest",
                    }
                    log.write(row)
                    seed_rows.append(row)

                if not seed_rows:
                    continue
                # choose the LR on the first seed: best val accuracy, then lowest val NLL
                if seed == seeds[0]:
                    best = sorted(seed_rows, key=lambda r: (-float(r["val_acc_last"]),
                                                            float(r["val_nll_last"])))[0]
                    lr_for_seed[seed] = float(best["lr"])
                else:
                    lr_for_seed[seed] = float(seed_rows[0]["lr"])
            mark_final(name, regime, lr_for_seed)


def mark_final(name, regime, lr_for_seed):
    """Flag the rows that count as results (one per seed, at the chosen LR)."""
    if not CSV_PATH.exists() or not lr_for_seed:
        return
    df = pd.read_csv(CSV_PATH)
    for seed, lr in lr_for_seed.items():
        m = (df.dataset == name) & (df.regime == regime) & (df.seed == seed)
        df.loc[m, "is_final"] = np.isclose(df.loc[m, "lr"], lr).astype(int)
    df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    main()
