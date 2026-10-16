# Guide for Abel and Andrew: Section 4 and the remaining report work

Last updated: 16 October 2026, by Sai.

Sections 2 and 3 (reproduction) are drafted and every experiment behind them is
finished and in this repo. This guide covers what is left:

- **Section 4, Improvements (30 marks).** Abel leads, with Andrew. Most of this guide.
- **Section 1, Motivation (10 marks)** and the **work-plan page**. Owners to agree in the group chat (Section 7 below).
- Shared jobs before submission (Section 8 below).

Read `docs/STATUS.md` for decisions already locked, and skim the Section 2–3
draft so Section 4 builds on what we found rather than repeating it.

---

## 1. What the spec asks for in Section 4

> "Proposed improvements and evaluation of improvements (e.g., show experiments
> to test how effective are the proposed improvements and discuss)."
> "At least one improvement should involve a non-trivial experimental
> contribution beyond simply changing a hyperparameter."

What this means for us:

- **At least one improvement must be more than a setting change.** Trying another
  layer or another token is a hyperparameter; fine as a side result, not as the headline.
- **Every improvement needs an evaluation**: the same datasets, the same seeds,
  a baseline from our reproduction, a paired test across datasets, and an honest
  discussion of where it helps and where it does not.
- **No pre-training.** The demonstrator ruled it out. Everything below uses the
  released Mantis-8M weights.
- **About 2.25 pages.** Sections 2 + 3 take about 2.75 of the 6 content pages,
  Section 1 about 0.75, leaving about 2.25 for Section 4 plus a short conclusion.
  References do not count.

---

## 2. What Section 3 found (build Section 4 on these)

Each improvement below answers one of these findings. Saying so in the report
("Section III-B showed X, so we try Y") is what makes Section 4 read as a
continuation instead of a list of tricks.

| # | Finding (numbers from `results/summary.json`) | Opens the door to |
|---|---|---|
| F1 | Zero-shot Mantis reproduces the paper (0.7821 vs 0.7812 on 158 datasets). | A trustworthy baseline to improve on. |
| F2 | **MiniRocket beats zero-shot Mantis by 5.6 pp** (0.8378 vs 0.7821, better on 126/158) and even beats fine-tuned Mantis on the 113 fine-tuning datasets (0.8565 vs 0.8395). | Combining the two (Improvement A). |
| F3 | The largest zero-shot gaps are on **long series** (over 1,000 steps, 26 datasets) and **unequal-length** series (16 datasets). Resizing everything to 512 throws information away; MiniRocket does far better on the same long series. | Better handling of long series (Improvement C). |
| F4 | **Temperature scaling makes calibration worse when the validation split is tiny**: ECE goes up by 2.2 pp on the 32 datasets with at most 10 validation cases, but down by 7.3 pp on the other 81. | Calibration that works with little data (Improvement B). |
| F5 | Fine-tuning adds 3.9 pp over zero-shot but needs a GPU; zero-shot embedding of all 158 datasets takes 9 minutes. | Keep improvements on the cheap zero-shot path where possible. |

---

## 3. Getting set up (about 1 hour, mostly downloads)

### 3.1 Clone and install

```bash
git clone https://github.com/Sai-prabha/mantis-reproduction
cd mantis-reproduction
conda create -n mantis python=3.11 -y
conda activate mantis
pip install -r requirements.txt
```

If you cloned before 16 October, Sai rewrote the history, so reset once
(this discards local changes, so commit or copy anything you need first):

```bash
git fetch
git reset --hard origin/main
```

### 3.2 Keep data and caches outside the repo

Add these to `~/.zshrc` (or `~/.bashrc`), then open a new terminal:

```bash
export MANTIS_DATA_DIR=~/mantis-work/data
export MANTIS_CACHE_DIR=~/mantis-work/cache
```

### 3.3 Download the data and check your setup

```bash
python scripts/00_dataset_stats.py --datasets all
python scripts/01_check_and_benchmark.py
```

The first downloads about 4.3 GB from Zenodo (an occasional HTTP 503 is a
Zenodo hiccup: rerun and it resumes). EMOPain always fails; that is expected.
The second prints your device (`mps`, `cuda` or `cpu`) and timings.

### 3.4 Rebuild the zero-shot embeddings once

Every zero-shot improvement reuses the cached Mantis embeddings. Write to a
separate CSV so you never touch the committed `results/zero_shot.csv`:

```bash
python scripts/02_zero_shot.py --datasets all --out results/zs_check_$(whoami).csv
```

Time: about 10 minutes on an Apple M-series GPU, roughly an hour on CPU.
Your accuracies should match `results/zero_shot.csv` to within about 0.5 pp
per dataset; if they don't, stop and tell Sai before building on them.

### 3.5 Code you can reuse

| Need | Use |
|---|---|
| Train/test data, resized to 512, labels encoded | `src.data.load_dataset(name)` returns `X_train (n, channels, 512)`, `y_train`, `X_test`, `y_test`, `meta` |
| Mantis embeddings (cached to disk) | `src.models.cached_embeddings(name, data, device)` returns `Z_train, Z_test, seconds, from_cache` |
| Mantis at another layer or token | `src.models.load_mantis(device, layer=..., token=...)` |
| Dataset lists | `configs.datasets.resolve("all" / "ucr" / "uea" / "tier1" / "unequal" / "A,B,C")` |
| Resumable CSV logging | `src.common.CsvLog(path, key_fields)` with `.done(**key)` and `.write(row)` |
| ECE, NLL, temperature scaling, IRM | `src.calibration.ece`, `nll`, `fit_temperature`, `IRM` |
| Saved fine-tuning outputs (no GPU needed) | `results/finetune_history/{dataset}__{regime}__s{seed}__lr{lr}.npz` holds `logits_val`, `y_val`, `logits_test`, `y_test` for every run |

Look at `scripts/02_zero_shot.py` as the template for a new script: it shows
argument parsing, the dataset loop, caching, failure logging and the CSV row format.

---

## 4. Recommended improvements

Pick **A and B** as the core (one accuracy, one calibration, each tied to a
Section 3 finding). Add **C** if time allows. A suggested split: Abel takes A
(and C), Andrew takes B. Agree it in the group chat.

### Improvement A: fuse Mantis and MiniRocket features (accuracy, non-trivial)

**Why.** F2: MiniRocket beats Mantis by 5.6 pp, yet Mantis still wins on 27
datasets. If the two make different mistakes, combining their features should
beat both. This tests whether the foundation model adds anything on top of a
strong classical method, which is a question v1 never asks.

**Method.**
1. For each dataset and seed, compute MiniRocket features on the same resized input
   (`from aeon.transformations.collection.convolution_based import MiniRocket`;
   check the import path in aeon 1.6, and fit the transform on the training set only).
2. Take the cached Mantis embeddings for the same dataset.
3. Standardise each block separately (fit the scaler on train only), concatenate, and
   train `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`, which is MiniRocket's own
   classifier. Also try the same classifier on each block alone, so the comparison
   isolates the effect of fusion rather than of the classifier.
4. MiniRocket gives about 10,000 features against 256 per channel for Mantis, so the
   larger block can swamp the smaller one. Try one simple re-weighting (for example,
   scale each block so it has equal total variance) and report both.

**Script.** `scripts/07_fusion.py`, writing `results/fusion.csv` with one row per
(dataset, seed, variant): `mantis_ridge`, `minirocket_ridge`, `fusion`, `fusion_weighted`.

**Evaluation.** All 158 datasets, seeds 0–2. Report means; wins/ties/losses and a
paired Wilcoxon test against (a) zero-shot Mantis RF, our reproduction baseline, and
(b) MiniRocket alone. Break results down by the series-type groups from Section 3
(long, unequal length, the rest) using `results/tables/zs_gaps_by_series_type.md`.

**Cost.** Minutes per seed on a laptop; no GPU needed once embeddings are cached.

**Pitfall.** Any scaler or transform fitted on train + test is leakage. Fit on train only.

### Improvement B: calibration that works with tiny validation sets (calibration, non-trivial)

**Why.** F4: temperature scaling, which v1 recommends, makes things worse on 32 of
our 113 fine-tuned datasets, the ones with at most 10 validation cases. A
practitioner with a small dataset is exactly who would use a foundation model.

**Method.** All the inputs already exist: every fine-tuning run saved its final
validation and test logits (`results/finetune_history/*.npz`), so this needs **no GPU
and no retraining**. Compare, on the 113 Full-regime runs (seed 0) and on the
30 tier-1 datasets (3 seeds):

1. No calibration (baseline).
2. Temperature scaling on the validation split (v1's method, our reproduction).
3. **Shrunk temperature**: `T = w * T_val + (1 - w) * T_prior`, where `T_prior` is the
   median temperature fitted on the *other* datasets (leave-one-dataset-out, so the
   dataset being scored never informs its own prior) and `w = n_val / (n_val + k)`.
   Choose `k` by leave-one-dataset-out as well, never on test.
4. A simple rule: calibrate only when `n_val` exceeds a threshold, chosen the same way.
5. IRM, for completeness.

**Script.** `scripts/08_calibration_small_val.py`, writing `results/calibration_small_val.csv`
with ECE and NLL per dataset, seed and method.

**Evaluation.** Mean ECE and NLL per method, overall and split at `n_val <= 10`;
wins/ties/losses and Wilcoxon against "no calibration" and against plain temperature
scaling. A reliability diagram for one small and one large dataset makes a good figure.

**Cost.** Seconds. This is the safest improvement to finish.

**Pitfall.** The test logits are for evaluation only. Every choice (`k`, threshold,
prior) must come from validation data or from other datasets.

### Improvement C (stretch): stop throwing away long series (representation)

**Why.** F3: long series lose most of their detail when squeezed into 512 steps,
and that is where Mantis loses most ground to MiniRocket.

**Method options (pick one).**
- **Windowing.** Cut each series longer than 512 into overlapping 512-step windows
  (for example stride 256), embed every window with Mantis, and average (or average and
  max-pool) the window embeddings. Short series are unchanged.
- **Multi-resolution.** Embed the series at lengths 256, 512 and 1024 and concatenate.
  `MantisV1`'s convolution kernel is sized for 512, so check that other lengths run
  before relying on them.

**Evaluation.** The 26 long datasets are the target; also run all 158 to show nothing
gets worse elsewhere. Compare with zero-shot Mantis and with MiniRocket.

**Cost.** Embedding several windows per series multiplies embedding time on long
datasets; still under an hour on a GPU.

### Cheap side results (not the headline)

`02_zero_shot.py` already supports these, so they cost one command each. Report them
in a sentence or a small table, labelled as hyperparameter checks:

```bash
python scripts/02_zero_shot.py --datasets all --layer 2 --out results/zs_layer2.csv
python scripts/02_zero_shot.py --datasets all --token mean_token --out results/zs_mean_token.csv
```

The MantisV2 paper (arXiv 2602.17868) reports that an intermediate layer improves
zero-shot accuracy on UCR, so `--layer 2` is a direct check of that claim on Mantis-8M.

---

## 5. Rules for every Section 4 experiment

These keep Section 4 as rigorous as Section 3 and protect the Section 5 marks.

1. **Same protocol as the reproduction**: datasets from `configs/datasets.py`, seeds 0, 1, 2,
   pre-processing from `src/data.py`. Change one thing at a time.
2. **Never select anything on the test set.** Hyperparameters come from training data,
   a validation split or other datasets. Section 3 shows how much test-set selection
   inflates numbers.
3. **Log every run** through `CsvLog` to a new CSV in `results/`, with the script name,
   settings, seed, accuracy (or ECE), run time and `env_info(device)`.
4. **One summary script** (`scripts/09_summarise_s4.py`) turns those CSVs into the
   Section 4 tables and figures. No number in the report should be typed by hand.
5. **README**: add a "Section 4: improvements" block to the commands table with the
   exact command for each result. The spec marks this.
6. **Report the cost** (time, memory) next to the accuracy. "Better by 1 pp at 10 times
   the cost" is a different conclusion from "better by 1 pp for free".
7. **Report what didn't work** as clearly as what did. A well-tested negative result
   still earns marks.

## 6. Working in the repo together

- Work on a branch: `git checkout -b section4-abel` (or `section4-andrew`), push it,
  and open a pull request into `main` when a result is finished.
- Don't edit files someone else's script writes, and don't run the same script into the
  same CSV on two machines (merge conflicts in CSVs are painful). Use your own output file.
- `data/` and `cache/` stay out of git; each machine builds its own.
- Commit messages: say what changed and why, in one line.

---

## 7. Section 1 and the work-plan page

### Section 1: Motivation and impact (10 marks, about 0.75 page with the abstract and introduction)

The spec asks for the motivation for choosing the paper and its impact on the
AI/ML community. Points to cover:

- Foundation models for time series *classification* are rare compared with
  forecasting; Mantis is one of the first open, lightweight ones (8M parameters, trained
  with contrastive learning, open code and weights).
- Its practical claims (runs on modest hardware, works without fine-tuning, well
  calibrated) matter to practitioners with small labelled datasets, which is where
  foundation models are supposed to help most.
- Why reproduce it: reproducibility of ML results is an active concern (the spec's
  reading list), and foundation-model papers often compare only against other foundation
  models (the spec's "proper baselining" reading). Section 3 confirms this matters:
  MiniRocket beats Mantis.
- One paragraph previewing what we did and found (two or three numbers from Section 3),
  and the repo link.

Cite v1 as arXiv 2502.15637**v1**: later versions describe a different model.

### Work-plan page (1 extra page, required)

The spec asks for the project work plan, how the work was divided, and each
member's final contribution. A table works well: task, owner, dates, status.
Keep a short log as you go (commits and the group chat give the dates).
Remember the generative AI declaration the module requires.

## 8. Shared checklist before submission

- [ ] Section 4 results merged into `main`; README commands table updated.
- [ ] References from all sections merged into one `.bib` (Sai's are in the Section 2–3 bundle).
- [ ] Whole report compiled in the official IEEE Overleaf template: 6 content pages,
      1 work-plan page, references on top.
- [ ] Repo made **public** (markers cannot see a private repo) and the link in the report works.
- [ ] Generative AI use declared as the module requires.
- [ ] One person (the group contact) submits the PDF on Brightspace.
