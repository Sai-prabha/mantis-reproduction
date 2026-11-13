# Inference readings of frozen Mantis-8M

Same checkpoint and the same 200-tree forest as the reproduction. The encoder
weights are not updated. What changes is the array the forest is trained on.

The two limits under test are the ones the published protocol builds in.
Each channel is embedded on its own, so one forward pass does not mix
channels, and every series is resized to 512 before the encoder sees it.

Uses the reproduction environment in `requirements.txt`. No extra packages.

## What is measured

| Reading | What the forest reads | Where it is applied |
|---|---|---|
| Frozen encoder | Last-layer CLS token, one 256-d block per channel | Every dataset in the run |
| Channel pooling | Mean of those blocks | Every dataset. With one channel this is the published embedding |
| Cross-channel views | Published embedding, plus the mean across channels, the standard deviation across channels, and the leading principal component | Multivariate only. The three series are computed inside that example |
| Temporal halves | Published embedding, plus the two halves of the raw series, each resized to 512 | Only when every raw series is at least 1024 steps. Otherwise the published embedding is scored again |
| Channel shuffle | The frozen forest, scored after a fixed permutation of the test-channel blocks | Multivariate only. A check that the channel index carries the class |
| Calibration | Temperature scaling and isotonic regression on the frozen forest | Forest on 80% of the training split, maps fitted on the other 20%, ECE on the official test split |

## 1. Environment

Tested on: Windows 10, Intel CPU, Python 3.11.18, torch 2.15.0+cpu,
mantis-tsfm 1.1.0, aeon 1.6.0, scikit-learn 1.9.1. The forest is fitted with
one thread (`n_jobs=1`).

Use the `mantis` environment from the repository README. The same two
variables point at the data and the embedding cache:

```bash
export MANTIS_DATA_DIR=~/mantis-work/data
export MANTIS_CACHE_DIR=~/mantis-work/cache
```

On PowerShell:

```powershell
$env:MANTIS_DATA_DIR = "$env:USERPROFILE\mantis-work\data"
$env:MANTIS_CACHE_DIR = "$env:USERPROFILE\mantis-work\cache"
```

Run `python scripts/00_dataset_stats.py --datasets all` and
`python scripts/02_zero_shot.py` from the repository root first, so the
archives and the published CLS embeddings are already cached. These scripts
read that cache.

## 2. Data

`study` is the slice on which at least one of the two limits applies:

* every multivariate dataset that loaded (29)
* every series whose raw length is at least 1024
* twelve short univariate controls: Chinatown, ItalyPowerDemand, ECG200, CBF, TwoLeadECG, GunPoint, SwedishLeaf, Wafer, Adiac, Coffee, FiftyWords, Beef

The union is 62 datasets. EMOPain is left out because it no longer loads.
Pooling is an identity on one channel, and the halves are not added below
length 1024, so the other short univariate series in the 158 would tie the
frozen encoder on both readings.

Preprocessing is the reproduction's: raw archive files, linear interpolation
of missing values, linear resize to 512.

## 3. Commands for each result

Run from this directory, with the `mantis` environment active. The scripts
import the reproduction's `src` and `configs` and write to `results/` here.

| Command | Result |
|---|---|
| `python tests/test_views.py` | Checks that do not need the checkpoint or a download |
| `python scripts/run_views.py --datasets study` | Frozen encoder, pooling, cross-channel views, temporal halves, channel shuffle |
| `python scripts/run_calibration.py --datasets study` | ECE of the frozen forest, before and after temperature scaling and isotonic regression |
| `python scripts/run_bottleneck.py` | Parameter counts, resize time, forward time against the channel count |
| `python scripts/evaluate.py` | Tables and figures below |

Every run appends to a CSV under `results/` and skips a finished
`(dataset, method, seed)` row, so an interrupted run resumes.

Smaller slices, before the full study:

```bash
python scripts/run_views.py --datasets ECG200 --methods frozen_rf,channel_pool,segment
python scripts/run_views.py --datasets Epilepsy,BasicMotions --methods frozen_rf,channel_pool,cross_channel,channel_shuffle
python scripts/run_views.py --datasets long --methods segment
```

`channel_pool` and `channel_shuffle` only refit the forest.
`cross_channel` embeds three extra series on every multivariate dataset.
`segment` embeds two extra copies, and only on long series.
`run_bottleneck.py` uses synthetic batches and does not download data.

### Where each result comes from

`python scripts/evaluate.py` writes the tables and figures from the CSVs.

| Report item | Command | Output |
|---|---|---|
| Paired accuracy | `scripts/run_views.py --datasets study`, then `scripts/evaluate.py` | `results/tables/headline.md` |
| Per dataset | same | `results/tables/by_dataset.md` |
| Critical difference, multivariate | same | `results/tables/cd_multivariate.md`, `results/figures/cd_multivariate.pdf` |
| Channel pooling | same | `results/figures/scatter_channel_pool.pdf` |
| Cross-channel views | same | `results/figures/scatter_cross_channel.pdf`, `delta_vs_channels.pdf`, `delta_vs_train_size.pdf` |
| Temporal halves | same | `results/figures/scatter_segment.pdf`, `delta_vs_length.pdf` |
| Extra embedding time | same | `results/figures/tradeoff_accuracy_vs_time.pdf` |
| Calibration | `scripts/run_calibration.py --datasets study`, then `scripts/evaluate.py` | `results/calibration.csv`, `results/figures/calibration_ece.pdf` |
| Forward-pass cost | `scripts/run_bottleneck.py`, then `scripts/evaluate.py` | `results/bottleneck.json`, `results/figures/bottleneck_channels.pdf` |
| Refit against the reproduction | `scripts/evaluate.py` | row in `results/tables/headline.md`, from the reproduction `results/zero_shot.csv` |

## 4. Results

Means are over the seeds recorded for that dataset. A mean accuracy is for
orientation. The comparison is the median gap, wins/ties/losses, and a
two-sided Wilcoxon signed-rank test across datasets, with zero gaps dropped.
α = 0.05.

| Reading | Datasets | Frozen | This reading | Median Δ | W/T/L | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|
| Channel pool, univariate | 33 | 0.7710 | 0.7710 | 0.00 pp | 0/33/0 | ties |
| Channel pool, multivariate | 29 | 0.7385 | 0.6827 | -3.51 pp | 6/3/20 | <0.001 |
| Cross-channel views | 29 | 0.7385 | 0.7445 | +0.11 pp | 15/3/11 | 0.237 |
| Cross-channel views minus pooling | 29 |  |  | +4.30 pp | 21/2/6 | <0.001 |
| Temporal halves, length at least 1024 | 26 | 0.6902 | 0.7028 | +1.93 pp | 18/3/5 | 0.038 |
| Temporal halves not added | 36 | 0.8031 | 0.8031 | 0.00 pp | 0/36/0 | ties |
| Channel shuffle | 29 |  |  | -35 pp | 3/0/26 | <0.001 |
| Frozen forest refit against `results/zero_shot.csv` | 62 | 0.7564 | 0.7558 | 0.00 pp | 15/28/19 | 0.673 |

The refit mean is -0.07 pp. The largest single-dataset gap is 2.67 pp, on
DuckDuckGeese (50 test series). The reproduction forest used `n_jobs=-1`.

On the multivariate critical-difference diagram, cross-channel views take the
best average rank and stay grouped with the frozen encoder. Pooling is
separated from both and ranks worse.

Largest gaps outside the 5 pp band:

| Dataset | Reading | Frozen | This reading | Δ |
|---|---|---:|---:|---:|
| InsectWingbeat | Channel pool | 0.607 | 0.233 | -37.4 pp |
| Libras | Channel pool | 0.898 | 0.698 | -20.0 pp |
| UWaveGestureLibrary | Channel pool | 0.816 | 0.631 | -18.4 pp |
| JapaneseVowels | Channel pool | 0.962 | 0.787 | -17.5 pp |
| PigAirwayPressure | Temporal halves | 0.494 | 0.341 | -15.2 pp |
| PigCVP | Temporal halves | 0.788 | 0.659 | -13.0 pp |
| SemgHandSubjectCh2 | Temporal halves | 0.801 | 0.887 | +8.6 pp |
| HandMovementDirection | Cross-channel views | 0.203 | 0.284 | +8.1 pp |
| EthanolConcentration | Temporal halves | 0.275 | 0.356 | +8.1 pp |
| EthanolLevel | Temporal halves | 0.295 | 0.375 | +7.9 pp |

| Calibration | Datasets | ECE raw | ECE, temperature scaling | ECE, isotonic regression |
|---|---:|---:|---:|---:|
| All 62 | 62 | 0.212 | 0.081 | 0.074 |
| Univariate | 33 | 0.179 | 0.112 | 0.087 |
| Multivariate | 29 | 0.248 | 0.044 | 0.060 |

Both drops in ECE are significant (p < 0.001). Temperature scaling does not
change the predicted class. Isotonic regression moves accuracy from 0.741 to
0.734. Those accuracies sit below the table above because this forest sees
80% of the training split.

Resizing 128 series onto length 512 takes about 0.02 s from raw length 64
through 16384. A forward pass of the same 128 series takes 0.7 s at one
channel and 27 s at 32 channels. The six transformer blocks have 1,312,768
parameters each. The first block takes 0.39 of a full forward.

## 5. Deviations from the reproduction

| Reproduction | These runs | Reason |
|---|---|---|
| 1× V100 32 GB | GeForce RTX 2080 SUPER  | available hardware |
| 158 datasets | 62 | the two limits, plus 12 short controls |
| Calibration after fine-tuning | Calibration of the frozen forest | the encoder weights stay fixed |
| One embedding per channel, resized to 512 | Same resize. Extra series are still length 512 | the released tokeniser is built for 512 |

## Repository layout

```
scripts/     run_views.py, run_calibration.py, run_bottleneck.py, evaluate.py
config/      dataset list for the 62-dataset study
src/         pooling, cross-channel views, temporal halves, embeddings
tests/       test_views.py
results/     CSV logs, tables, figures
```
