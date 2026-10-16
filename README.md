# Reproducing Mantis-8M (COMP41850 AI4TS)

Reproduction of Feofanov et al., "Mantis: Lightweight Calibrated Foundation
Model for User-Friendly Time Series Classification", arXiv
[2502.15637v1](https://arxiv.org/pdf/2502.15637v1) (Feb 2025), using the
authors' code ([vfeofanov/mantis](https://github.com/vfeofanov/mantis),
`mantis-tsfm==1.1.0`) and checkpoint
([paris-noah/Mantis-8M](https://huggingface.co/paris-noah/Mantis-8M), 8.11M parameters).

> Cite v1 explicitly. The un-versioned arXiv link opens a later version, which
> describes a different, 4.2M-parameter model.

Group members: start with [`docs/STATUS.md`](docs/STATUS.md) (who is doing what,
what's done and what's next) and [`docs/SECTION4_GUIDE.md`](docs/SECTION4_GUIDE.md)
(setting up and running the Section 4 experiments).

## What is reproduced

| v1 claim | Where in v1 | Script |
|---|---|---|
| Zero-shot accuracy, RF on frozen features | Fig. 3, App. Table 3 | `02_zero_shot.py` |
| Zero-shot comparison with MOMENT | Fig. 3 | `05_moment.py` |
| Fine-tuned accuracy | Fig. 5, App. Table 4 | `03_finetune.py` |
| Full fine-tuning vs training from scratch | Fig. 7, Tables 6 and 7 | `03_finetune.py` |
| Calibration of fine-tuned models (ECE, TS, IRM) | Fig. 9 | `03_finetune.py` |
| Non-foundation-model baseline (not in v1) | not in v1 | `04_minirocket.py` |

Not reproduced: anything requiring pretraining (e.g. the differential-block
ablation, v1 Table 5).

## 1. Environment

Tested on: MacBook Pro, Apple M5 Pro, 48 GB, Python 3.11.17, torch 2.14.1 (MPS backend),
mantis-tsfm 1.1.0, aeon 1.6.0. Exact versions: `requirements-lock.txt`.

Use an isolated Python 3.11 environment (never the base/system Python).
With conda (Anaconda/Miniconda):

```bash
conda create -n mantis python=3.11 -y
conda activate mantis
pip install -r requirements.txt
```

Exact versions used for the report are in `requirements-lock.txt`
(`pip install -r requirements-lock.txt` reproduces them). Without conda,
`python3.11 -m venv` works the same way.

The MOMENT baseline needs its own environment, because `momentfm` pins old
versions of numpy and transformers:

```bash
conda create -n mantis-moment python=3.11 -y
conda run -n mantis-moment pip install -r requirements-moment.txt
```

Keep large files outside the repo. The scripts read these two variables
(add them to `~/.zshrc` or `~/.bashrc`):

```bash
export MANTIS_DATA_DIR=~/mantis-work/data
export MANTIS_CACHE_DIR=~/mantis-work/cache
```

`MANTIS_DATA_DIR` holds the downloaded archives; `MANTIS_CACHE_DIR` holds
preprocessed data and embeddings.

The checkpoint is pinned in `src/common.py` to Hugging Face commit
`bc7d5ab40c02133386a28e2c127f35c17c86901d` of `paris-noah/Mantis-8M`, so everyone
gets identical weights.

> zsh users: paste commands without trailing `# comments`. Interactive zsh
> treats `#` as an argument unless you run `setopt interactivecomments`.

## 2. Data

Datasets download automatically on first use through aeon (from the archive's
Zenodo records). Zenodo occasionally returns HTTP 503; rerun the same command
later and already-downloaded datasets are skipped.

| Command | What it does |
|---|---|
| `python scripts/00_dataset_stats.py --datasets all` | Download and describe all 159 zero-shot datasets |
| `python scripts/00_dataset_stats.py --datasets tier1` | Download and describe the 30 tier-1 fine-tuning datasets |

Preprocessing (identical for every method): raw archive files
(`load_equal_length=False, load_no_missing=False` set explicitly, because
aeon's defaults for these changed between versions), linear interpolation of
missing values, linear resizing of every series to length 512.

## 3. Commands for each result

Run from the repo root with the `mantis` environment active.

| Command | Result |
|---|---|
| `python scripts/00_dataset_stats.py --datasets all` | Download and describe all datasets (also converts caches for the MOMENT env) |
| `python scripts/01_check_and_benchmark.py` | Environment checks, MPS vs CPU, timing |
| `python scripts/02_zero_shot.py --datasets ucr` | Zero-shot, UCR-128 |
| `python scripts/02_zero_shot.py --datasets uea` | Zero-shot, UEA-27 |
| `python scripts/02_zero_shot.py --datasets others` | Zero-shot, 4 extra datasets (EMOPain is no longer downloadable) |
| `python scripts/02_zero_shot.py --datasets unequal --aeon-equal-length` | Sensitivity check: unequal-length datasets loaded from aeon's pre-equalised `_eq` files |
| `python scripts/04_minirocket.py --datasets all` | MiniRocket baseline |
| `conda run --no-capture-output -n mantis-moment python scripts/05_moment.py --datasets all` | MOMENT-1-large baseline (about 6.6 h on MPS) |
| `python scripts/03_finetune.py --datasets tier1` | Fine-tuning: 30 datasets, 4 regimes, calibration |
| `python scripts/03_finetune.py --datasets ECG200,GunPoint,Wafer,Adiac,FiftyWords --package-defaults --seeds 0` | Package defaults vs paper protocol |
| `python scripts/03_finetune.py --datasets tier2 --regimes full --seeds 0 --fixed-lr 0.001` | Fine-tuning breadth: the other 83 UCR datasets of v1's 131 |
| `python scripts/06_summarise.py` | All tables, figures and the results table below |

Every script appends to a CSV in `results/` and skips work already recorded,
so an interrupted run resumes where it stopped.

To check the code runs without downloading anything, `python tests/smoke_test.py`
runs every script on synthetic data with a randomly initialised model.

### Where each result in the report comes from

`scripts/06_summarise.py` writes every table and figure below from the CSVs in
`results/`. Numbers quoted in the report text that are not in another table are
collected in `results/tables/report_numbers.md`.

| Report item | Commands | Output |
|---|---|---|
| Section II-A, parameter and buffer counts | `01_check_and_benchmark.py` | `results/check_and_benchmark.json` |
| Table II, zero-shot rows | `02_zero_shot.py` (ucr, uea, others), then `06_summarise.py` | `results/tables/headline.md`, `zs_paper_vs_ours_by_archive.md` |
| Table II, MOMENT rows | `05_moment.py --datasets all`, then `06_summarise.py` | `results/tables/headline.md` |
| Table II, fine-tuning rows | `03_finetune.py` (tier1 and tier2), then `06_summarise.py` | `results/tables/ft_full_vs_paper.md` |
| Table III and Full minus Scratch | `03_finetune.py --datasets tier1`, then `06_summarise.py` | `results/tables/ft_regimes_vs_paper.md`, `ft_regimes.md` |
| Fig. 1 | `02_zero_shot.py`, then `06_summarise.py` | `results/figures/zs_scatter_by_type.pdf` |
| Section III-A, gaps by series type | `06_summarise.py` | `results/tables/zs_gaps_by_series_type.md`, `zs_largest_gaps.md` |
| Section III-A, unequal-length check | `02_zero_shot.py --datasets unequal --aeon-equal-length`, then `06_summarise.py` | `results/tables/zs_unequal_length_check.md` |
| Section III-B, MiniRocket | `04_minirocket.py --datasets all`, then `06_summarise.py` | `results/tables/baselines_archive_means.md`, `report_numbers.md` |
| Section III-C, package defaults | `03_finetune.py ... --package-defaults`, then `06_summarise.py` | `results/tables/report_numbers.md` |
| Section III-E, calibration | `03_finetune.py` (tier1 and tier2), then `06_summarise.py` | `results/tables/calibration.md`, `report_numbers.md`, `results/summary.json` |
| Section III-F, MPS against CPU | `01_check_and_benchmark.py` | `results/check_and_benchmark.json` |

## 4. Results: ours vs the paper

Generated by `scripts/06_summarise.py`, which rewrites the table between the
markers below, so don't edit it by hand. Paper numbers come from v1's appendix
tables (`targets/`) and, for calibration, from v1's Figure 9. Means are over the
datasets present in both. In the fine-tuning rows, tier 1 is averaged over its
3 seeds and tier 2 has one seed. Per-dataset tables and figures are in
`results/tables/` and `results/figures/`.

<!-- RESULTS-TABLE:START -->
| Setting                                                                    |   Datasets | Paper (v1)   | Ours     | Δ        | Wilcoxon p   |
|:---------------------------------------------------------------------------|-----------:|:-------------|:---------|:---------|:-------------|
| Zero-shot, UCR (RF200, last layer, CLS)                                    |        128 | 0.7922       | 0.7925   | +0.03 pp | 0.720        |
| Zero-shot, UEA (RF200, last layer, CLS)                                    |         27 | 0.7170       | 0.7203   | +0.34 pp | 0.499        |
| Zero-shot, other (3 of 4) (RF200, last layer, CLS)                         |          3 | 0.8935       | 0.8938   | +0.03 pp |              |
| Zero-shot, UCR-128 cross-check (arXiv 2602.17868)                          |        128 | 0.7921       | 0.7925   | +0.04 pp |              |
| Full fine-tuning, tier 1 (3 seeds), best epoch (ours chosen on test)       |         30 | 0.8413       | 0.8549   | +1.36 pp | 0.101        |
| Full fine-tuning, tier 1 (3 seeds), best epoch (ours chosen on validation) |         30 | 0.8413       | 0.8421   | +0.07 pp | 0.952        |
| Full fine-tuning, tier 1 (3 seeds), ours at last epoch                     |         30 | 0.8413       | 0.8445   | +0.32 pp | 0.984        |
| Full fine-tuning, tiers 1 and 2, best epoch (ours chosen on test)          |        113 | 0.8428       | 0.8551   | +1.23 pp | <0.001       |
| Full fine-tuning, tiers 1 and 2, best epoch (ours chosen on validation)    |        113 | 0.8428       | 0.8382   | -0.46 pp | 0.139        |
| Full fine-tuning, tiers 1 and 2, ours at last epoch                        |        113 | 0.8428       | 0.8395   | -0.33 pp | 0.282        |
| Full minus scratch, last epoch                                             |         30 | +5.14 pp     | +6.35 pp |          |              |
| Full minus scratch, best epoch                                             |         30 | +3.83 pp     | +5.55 pp |          |              |
| ECE, Full fine-tuning (seed 0), no post-hoc calibration                    |        113 | 0.1131       | 0.1200   | +0.69 pp |              |
| ECE, Full fine-tuning (seed 0), temperature scaling                        |        113 | 0.0673       | 0.0737   | +0.64 pp |              |
| ECE, Full fine-tuning (seed 0), IRM                                        |        113 | 0.0609       | 0.0706   | +0.97 pp |              |
| Zero-shot, MOMENT-1-large (RF200)                                          |        158 | 0.7658       | 0.7643   | -0.15 pp | 0.587        |
| Zero-shot, Mantis minus MOMENT                                             |        158 | +1.54 pp     | +1.78 pp |          | 0.018        |
<!-- RESULTS-TABLE:END -->

## 5. Deviations from the paper

| Paper (v1) | Ours | Reason |
|---|---|---|
| 1× V100 32 GB | Apple M5 Pro, 48 GB (MPS) | available hardware |
| 159 zero-shot datasets | 158 (EMOPain no longer in aeon's download index) | data availability |
| Fine-tuning on 131 datasets | the 113 UCR ones: tier 1 (30, all regimes, 3 seeds) + tier 2 (83, Full, 1 seed) | compute; scope agreed with demonstrator |
| LR chosen on validation (details unstated) | tier 1: chosen on seed 0, reused for seeds 1 and 2; tier 2: fixed 1e-3 | compute |
| Regime study (Tables 6 and 7): fixed LR 2e-4 | LR selected as above | one protocol for all fine-tuning |
| Model after LR selection: unstated | trained on 80%, not refit on 100% | 20% split needed for calibration |
| "Best epoch": selection unstated | evaluated every 5 epochs; reported as validation-selected and test-selected (oracle) | make the ambiguity measurable |
| Missing values: unstated | linear interpolation | assumption |
| Unequal lengths: "resized to 512" | raw series (`load_equal_length=False`), each resized; `_eq` files tested separately | assumption, checked |
| InsectWingbeat subset: sizes only | stratified 1000 train / 1000 test, seed 0 | assumption |
| Seeds: 3, values unstated | 0, 1, 2 | |
| NuTime, GPT4TS, UniTS comparators | not run; MiniRocket added instead | time |

## Repository layout

```
configs/datasets.py     dataset lists (UCR-128, UEA-27, others, tier 1, tier 2, unequal)
src/                    data loading, model helpers, calibration metrics
scripts/00-06           one script per step above
tests/smoke_test.py     offline run of every script on synthetic data
targets/                paper numbers from v1's appendix Tables 3, 4, 6, 7 and 8 (see targets/README.md)
results/                CSV logs, tables, figures (committed)
```
