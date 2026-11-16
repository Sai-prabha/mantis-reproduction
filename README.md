# Reproducing and improving Mantis-8M (COMP41850 AI4TS)

This repository reproduces Feofanov et al., "Mantis: Lightweight Calibrated
Foundation Model for User-Friendly Time Series Classification", arXiv
[2502.15637v1](https://arxiv.org/pdf/2502.15637v1) (Feb 2025), using the
authors' code ([vfeofanov/mantis](https://github.com/vfeofanov/mantis),
`mantis-tsfm==1.1.0`) and checkpoint
([paris-noah/Mantis-8M](https://huggingface.co/paris-noah/Mantis-8M), 8.11M parameters).
It then tests improvements on top of the frozen model, using standard
scikit-learn and aeon components only.

> Cite v1 explicitly. The un-versioned arXiv link opens a later version, which
> describes a different, 4.2M-parameter model.

## Key results

| | Result |
|---|---|
| Zero-shot accuracy | Matches the paper: UCR 0.7925 vs 0.7922 (128 datasets), UEA 0.7203 vs 0.7170 (27 datasets) |
| Fine-tuning | Matches only if "best epoch" is not picked on the test set: validation-selected 0.8382 vs the paper's 0.8428 (113 datasets), test-selected 0.8551 |
| Pretraining benefit | Full fine-tuning beats training from scratch by +6.35 pp (paper +5.14 pp) |
| Against MOMENT | Mantis is +1.78 pp better zero-shot (p = 0.018) and embeds about 42 times faster |
| Against a non-foundation baseline | MiniRocket beats zero-shot Mantis, 0.8378 vs 0.7821 on 158 datasets, in about 11 minutes of CPU time |
| Improvement | Embedding raw windows of long series next to the published embedding gains +1.6 to +2.6 pp on 44 long datasets (Holm p below 0.01). It does not close the gap to MiniRocket |

Details, tests and caveats are in sections 4 and 6.

## What is reproduced

| v1 claim | Where in v1 | Script |
|---|---|---|
| Zero-shot accuracy, RF on frozen features | Fig. 3, App. Table 3 | `02_zero_shot.py` |
| Zero-shot comparison with MOMENT | Fig. 3 | `05_moment.py` |
| Fine-tuned accuracy | Fig. 5, App. Table 4 | `03_finetune.py` |
| Full fine-tuning vs training from scratch | Fig. 7, Tables 6 and 7 | `03_finetune.py` |
| Calibration of fine-tuned models (ECE, TS, IRM) | Fig. 9 | `03_finetune.py` |
| Non-foundation-model baseline (not in v1) | not in v1 | `04_minirocket.py` |

Not reproduced: anything requiring pretraining (for example the
differential-block ablation, v1 Table 5).

## 1. Environment

The reproduction (sections 3 to 5) ran on a MacBook Pro, Apple M5 Pro, 48 GB,
Python 3.11.17, torch 2.14.1 (MPS backend), mantis-tsfm 1.1.0, aeon 1.6.0.
Exact versions are in `requirements-lock.txt`.

The improvement experiments (section 6) ran on Windows 10 with an AMD64 CPU,
Python 3.11.18, torch 2.15.0+cpu, scikit-learn 1.9.1, aeon 1.6.0. Each result
row records its hardware in the `hardware` column of its CSV.

Use an isolated Python 3.11 environment, never the base or system Python.
With conda:

```bash
conda create -n mantis python=3.11 -y
conda activate mantis
pip install -r requirements.txt
```

`pip install -r requirements-lock.txt` reproduces the exact versions of the
reproduction. Without conda, `python3.11 -m venv` works the same way.

The MOMENT baseline needs its own environment, because `momentfm` pins old
versions of numpy and transformers:

```bash
conda create -n mantis-moment python=3.11 -y
conda run -n mantis-moment pip install -r requirements-moment.txt
```

Keep large files outside the repository. The scripts read two variables.

macOS or Linux (add to `~/.zshrc` or `~/.bashrc`):

```bash
export MANTIS_DATA_DIR=~/mantis-work/data
export MANTIS_CACHE_DIR=~/mantis-work/cache
```

Windows PowerShell:

```powershell
$env:MANTIS_DATA_DIR = "$env:USERPROFILE\mantis-work\data"
$env:MANTIS_CACHE_DIR = "$env:USERPROFILE\mantis-work\cache"
```

`MANTIS_DATA_DIR` holds the downloaded archives. `MANTIS_CACHE_DIR` holds
preprocessed data and embeddings, including the window embeddings of section
6.2 under `emb/windows/`. Run every script from the same shell, or it will not
find the cache.

The checkpoint is pinned in `src/common.py` to Hugging Face commit
`bc7d5ab40c02133386a28e2c127f35c17c86901d` of `paris-noah/Mantis-8M`, so every
run uses identical weights.

> zsh users: paste commands without trailing `# comments`. Interactive zsh
> treats `#` as an argument unless you run `setopt interactivecomments`.

## 2. Data

Datasets download automatically on first use through aeon, from the archive's
Zenodo records. Zenodo occasionally returns HTTP 503. Rerun the same command
later and already-downloaded datasets are skipped.

| Command | What it does |
|---|---|
| `python scripts/00_dataset_stats.py --datasets all` | Download and describe all 159 zero-shot datasets |
| `python scripts/00_dataset_stats.py --datasets tier1` | Download and describe the 30 tier-1 fine-tuning datasets |

Preprocessing, identical for every method: raw archive files
(`load_equal_length=False, load_no_missing=False` set explicitly, because
aeon's defaults changed between versions), linear interpolation of missing
values, and linear resizing of every series to length 512.

## 3. Commands for each result

Run from the repository root with the `mantis` environment active.

| Command | Result |
|---|---|
| `python scripts/00_dataset_stats.py --datasets all` | Download and describe all datasets (also converts caches for the MOMENT environment) |
| `python scripts/01_check_and_benchmark.py` | Environment checks, MPS against CPU, timing |
| `python scripts/02_zero_shot.py --datasets ucr` | Zero-shot, UCR-128 |
| `python scripts/02_zero_shot.py --datasets uea` | Zero-shot, UEA-27 |
| `python scripts/02_zero_shot.py --datasets others` | Zero-shot, 4 extra datasets (EMOPain is no longer downloadable) |
| `python scripts/02_zero_shot.py --datasets unequal --aeon-equal-length` | Sensitivity check: unequal-length datasets loaded from aeon's pre-equalised `_eq` files |
| `python scripts/04_minirocket.py --datasets all` | MiniRocket baseline |
| `conda run --no-capture-output -n mantis-moment python scripts/05_moment.py --datasets all` | MOMENT-1-large baseline (about 6.6 h on MPS) |
| `python scripts/03_finetune.py --datasets tier1` | Fine-tuning: 30 datasets, 4 regimes, calibration |
| `python scripts/03_finetune.py --datasets ECG200,GunPoint,Wafer,Adiac,FiftyWords --package-defaults --seeds 0` | Package defaults against the paper protocol |
| `python scripts/03_finetune.py --datasets tier2 --regimes full --seeds 0 --fixed-lr 0.001` | Fine-tuning breadth: the other 83 UCR datasets of v1's 131 |
| `python scripts/06_summarise.py` | All tables, figures and the results table in section 4 |

Every script appends to a CSV in `results/` and skips work already recorded,
so an interrupted run resumes where it stopped.

### Where each result in the report comes from

`scripts/06_summarise.py` writes every table and figure below from the CSVs in
`results/`. Numbers quoted in the report text that are not in another table
are collected in `results/tables/report_numbers.md`.

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
| Improvements (section 6) | see the table in section 6.4 | `results/tables/fusion_*.md`, `v3_*.md` |

## 4. Results: ours vs the paper

Generated by `scripts/06_summarise.py`, which rewrites the table between the
markers below, so do not edit it by hand. Paper numbers come from v1's
appendix tables (`targets/`) and, for calibration, from v1's Figure 9. Means
are over the datasets present in both. In the fine-tuning rows, tier 1 is
averaged over its 3 seeds and tier 2 has one seed. Per-dataset tables and
figures are in `results/tables/` and `results/figures/`.

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
| 1× V100 32 GB | Apple M5 Pro, 48 GB (MPS) for the reproduction; a Windows CPU for section 6 | available hardware |
| 159 zero-shot datasets | 158 (EMOPain is no longer in aeon's download index) | data availability |
| Fine-tuning on 131 datasets | the 113 UCR ones: tier 1 (30, all regimes, 3 seeds) + tier 2 (83, Full, 1 seed) | compute; scope agreed with the demonstrator |
| LR chosen on validation (details unstated) | tier 1: chosen on seed 0, reused for seeds 1 and 2; tier 2: fixed 1e-3 | compute |
| Regime study (Tables 6 and 7): fixed LR 2e-4 | LR selected as above | one protocol for all fine-tuning |
| Model after LR selection: unstated | trained on 80%, not refit on 100% | the 20% split is needed for calibration |
| "Best epoch": selection unstated | evaluated every 5 epochs; reported as validation-selected and test-selected (oracle) | make the ambiguity measurable |
| Missing values: unstated | linear interpolation | assumption |
| Unequal lengths: "resized to 512" | raw series (`load_equal_length=False`), each resized; `_eq` files tested separately | assumption, checked |
| InsectWingbeat subset: sizes only | stratified 1000 train / 1000 test, seed 0 | assumption |
| Seeds: 3, values unstated | 0, 1, 2 | |
| NuTime, GPT4TS, UniTS comparators | not run; MiniRocket added instead | time |

## 6. Improvements

All improvements keep the Mantis encoder frozen and use the official
train/test splits. Every comparison is a two-sided Wilcoxon signed-rank test
over datasets, with Holm correction inside one family of comparisons. The
planned comparisons were written down before the results were looked at.
Comparisons added afterwards are labelled post hoc and corrected separately.
Ranks, not mean accuracy, are the headline, as in the course evaluation
checklist. A "gain" or "loss" in the tables is a change of more than one
accuracy point on a dataset.

### 6.1 Fusion with MiniRocket and statistics (scripts 07 to 10)

Question: does the Mantis embedding add information that a strong random-kernel
method does not already have, and does adding raw-series statistics (level,
spread, slope, first-difference statistics) repair Mantis's loss of scale
information? Feature blocks are Mantis (M), nine statistics per channel (S) and
MiniRocket (R). Blocks are concatenated, equally weighted, and fed to one
ridge classifier. 158 datasets, 3 seeds.

| Method | Mean rank | Mean accuracy | Total minutes |
|---|---:|---:|---:|
| MiniRocket + Mantis + Stats | 2.93 | 0.8396 | 416 |
| MiniRocket + Mantis | 3.00 | 0.8412 | 413 |
| MiniRocket | 3.65 | 0.8379 | 17 |
| MiniRocket + Stats | 3.74 | 0.8330 | 21 |
| Mantis + Stats (ridge) | 5.24 | 0.8023 | 401 |
| Mantis (ridge) | 5.58 | 0.8004 | 397 |
| Mantis + Stats (RF) | 6.00 | 0.7929 | 400 |
| Mantis (RF, as in the paper) | 6.75 | 0.7816 | 397 |
| Stats (ridge) | 8.26 | 0.6326 | 3 |
| Majority class | 9.85 | 0.3097 | 0 |

| Planned comparison | Mean diff | W/T/L | Holm p |
|---|---:|---|---:|
| Mantis + Stats vs Mantis (RF) | +1.13 pp | 106/15/37 | below 0.001 |
| MiniRocket + Mantis vs MiniRocket | +0.33 pp | 82/18/58 | 0.259 |
| MiniRocket + Mantis + Stats vs MiniRocket + Mantis | -0.16 pp | 66/29/63 | 0.775 |
| MiniRocket + Mantis + Stats vs MiniRocket | +0.17 pp | 77/18/63 | 0.284 |
| Mantis (ridge) vs Mantis (RF), control | +1.88 pp | 103/10/45 | below 0.001 |

Reading: statistics help the paper's forest head. Mantis adds nothing
significant on top of MiniRocket and costs about 24 times the run time.
Script 09 shows which statistics matter (exploratory), and script 10 adds
1NN-DTW as the course baseline. DTW skips the 26 datasets listed in
`results/dtw_skipped.csv` because of its run time.

### 6.2 Multi-scale window embeddings (script 11)

Question: the published protocol resizes every series to 512 points, so a
2,000-point series loses three quarters of its samples. Does embedding k raw
windows of the series as well recover the lost detail? Each window is cut from
the raw series, resized to 512, and embedded with the same frozen encoder. The
window embeddings are averaged (`mean`) or placed side by side (`concat`) next
to the published whole-series embedding. Only the 44 datasets whose raw series
are longer than 512 are used, a group fixed from the metadata. Window
embeddings are cached per (k, overlap).

| Method (44 datasets) | Mean accuracy | Mean rank |
|---|---:|---:|
| MiniRocket (ridge) | 0.7891 | 4.74 |
| Mantis + 4 windows, mean (ridge) | 0.7752 | 4.72 |
| Mantis + 2 windows, mean (ridge) | 0.7710 | 4.65 |
| Mantis (ridge) | 0.7447 | 7.15 |
| Mantis + 2 windows, mean (RF) | 0.7443 | 6.78 |
| Mantis (RF, as in the paper) | 0.7279 | 8.13 |

Friedman test over all 11 methods: chi-squared 75.06, p below 0.001.

| Planned comparison | Mean diff | W/T/L | Holm p |
|---|---:|---|---:|
| 2 windows vs plain Mantis, paper's RF head | +1.63 pp | 32/2/10 | 0.004 |
| 2 windows vs plain Mantis, ridge head | +2.63 pp | 34/3/7 | below 0.001 |
| Concatenate vs average, 2 windows, RF | +0.12 pp | 23/2/19 | 1.000 |
| 4 windows vs 2 windows, RF | +0.39 pp | 23/3/18 | 1.000 |
| 2 windows (ridge) vs MiniRocket | -1.81 pp | 18/2/24 | 1.000 |

Reading: windows help significantly with either head. The pooling choice and
the number of windows do not matter. Windowed Mantis does not close the gap to
MiniRocket. Two windows roughly double the encoder time and four windows
roughly triple it. The ridge head and the encoder are deterministic, so the
three seeds give identical ridge results and these numbers rest on the single
official split. Script 13 below addresses that.

### 6.3 Further experiments and evaluation upgrades (scripts 12 to 14)

| Script | What it tests |
|---|---|
| `12_channel_reduce.py` | Channel reduction before the encoder for datasets with more than 8 channels: PCA fitted on the training series, a random channel subset, or summary channels (magnitude, mean, standard deviation). Accuracy against encoder time. |
| `13_resample.py` | Stratified train/test resamples (resample 0 is the official split). Reports the spread over resamples for Mantis, MiniRocket, their fusion and windowed Mantis. Use `--min-len 512` for the windowed group. |
| `14_proba_metrics.py` | AUROC, log loss, macro F1 and ECE for heads that output probabilities (the paper's RF and a logistic head) on Mantis, MiniRocket and both. |

Results for these three are written by script 15 to `results/tables/v3_*.md`.
Do not quote them from a partial run: each script logs the datasets it has
finished, and the report tables include only datasets where every compared
method has finished.

### 6.4 Commands for the improvements

Run from the repository root with the `mantis` environment active. Scripts 07
to 14 need the zero-shot embeddings from `02_zero_shot.py` (cached) and the
MiniRocket rows of `07_fusion.py` for the report.

| Command | Result |
|---|---|
| `python scripts/07_fusion.py --datasets all` | Mantis, statistics and MiniRocket fusion (10 methods, 3 seeds) |
| `python scripts/09_stats_ablation.py --datasets all` | Which statistics help Mantis (level, slope, diff, all; RF and ridge) |
| `python scripts/10_dtw_baseline.py --datasets all` | 1NN-DTW baseline with a 10% window, untuned |
| `python scripts/08_fusion_report.py` | Fusion tables, tests and figures, including the ablation and DTW tables |
| `python scripts/11_multiscale.py --datasets all` | Window embeddings for series longer than 512 (`--windows 2 4`, `--overlap`, `--pools mean concat`) |
| `python scripts/12_channel_reduce.py --datasets uea,others` | Channel reduction (`--k 8`, `--min-channels 8`) |
| `python scripts/13_resample.py --datasets all --resamples 10 --min-len 512 --methods M:RF200,M:ridge,R:ridge,M+W2mean:RF200,M+W2mean:ridge` | Resampled evaluation of windowed Mantis |
| `python scripts/13_resample.py --datasets ucr --resamples 10` | Resampled Mantis, MiniRocket and fusion on all UCR datasets |
| `python scripts/14_proba_metrics.py --datasets all` | Probability metrics |
| `python scripts/15_improve_report.py` | Ranks, Friedman tests, planned comparisons with Holm correction, critical difference diagrams, cost tables |

| Report item | Output |
|---|---|
| Fusion overview, planned and post hoc tests, MCM row, groups | `results/tables/fusion_*.md`, `results/figures/fusion_cd.pdf`, `fusion_rank_vs_runtime.pdf`, `results/fusion_summary.json` |
| Multi-scale tables and critical difference diagram | `results/tables/v3_multiscale_*.md`, `results/figures/v3_multiscale_cd.pdf` |
| Channel reduction | `results/tables/v3_channels_*.md`, `results/figures/v3_channels_cd.pdf` |
| Resampling | `results/tables/v3_resample_*.md`, `results/figures/v3_resample_cd.pdf` |
| Probability metrics | `results/tables/v3_proba_*.md` |
| All improvement numbers | `results/v3_summary.json` |

The planned comparisons for each experiment are listed at the top of
`scripts/15_improve_report.py` and `scripts/08_fusion_report.py`. Edit them
before running a new experiment, not after seeing its results.

## 7. Tests

Neither test needs a download or a trained checkpoint.

| Command | What it checks |
|---|---|
| `python tests/test_v3_units.py` | Window, channel-reduction and statistics helpers on small arrays |
| `python tests/smoke_test.py` | Runs scripts 00 to 10 end to end on synthetic data with a randomly initialised model |
| `python tests/smoke_test_v3.py` | Runs scripts 11 to 15 end to end on synthetic data |

The accuracies the smoke tests print are meaningless. They only show that the
scripts run.

## Repository layout

```
configs/datasets.py     dataset lists (UCR-128, UEA-27, others, tier 1, tier 2, unequal)
src/                    data loading, model helpers, calibration metrics, feature blocks,
                        windows.py, channels.py, stats_utils.py
scripts/00-15           one script per step above
tests/                  unit test and offline smoke tests
targets/                paper numbers from v1's appendix Tables 3, 4, 6, 7 and 8 (see targets/README.md)
results/                CSV logs, tables, figures (committed)
```
