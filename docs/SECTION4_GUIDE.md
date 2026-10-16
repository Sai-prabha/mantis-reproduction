# Guide for Andrew and Abel: Section 4 and the rest of the report

Last updated: 16 October 2026, by Sai.

Sections 2 and 3 (the reproduction) are drafted, every experiment behind them is
finished and committed, and the repo (Section 5) is set up. This guide covers what
is left so that every part of the report has someone working on it.

## 0. Every part of the report

The spec asks for 6 pages of content in the IEEE double-column template, plus 1 page
for the work plan and 1 to 2 pages of references, with the repo link in the report.

| Part | Marks | Who | Status | Where to look in this guide |
|---|---|---|---|---|
| Title, abstract, introduction | (part of 1) | to agree | not started | Section 7 |
| 1. Motivation and impact | 10 | to agree | not started | Section 7 |
| 2. Approach to reproduction | 10 | Sai | draft done | read it before starting Section 4 |
| 3. Lessons learned | 30 | Sai | draft done | summarised in Section 2 below |
| 4. Improvements | 30 | Andrew and Abel, split however you like | not started | Sections 1 to 6 |
| 5. GitHub repo and README | 20 | Sai, with everyone adding their own commands | reproduction part done | Section 5, rules 4 and 5 |
| Conclusion (short) | (part of 4) | to agree | not started | Section 7 |
| Work-plan page | required | to agree, everyone contributes | not started | Section 7 |
| References | 1 to 2 pages, not counted | everyone adds theirs | Sai's list ready | Section 8 |

`docs/STATUS.md` lists the decisions already made. Skim the Sections 2 and 3 draft
before you start so Section 4 builds on those results instead of repeating them.

## 1. What the spec asks for in Section 4

> "Proposed improvements and evaluation of improvements (e.g., show experiments
> to test how effective are the proposed improvements and discuss)."
> "At least one improvement should involve a non-trivial experimental
> contribution beyond simply changing a hyperparameter."

For us that means:

- At least one improvement has to be more than a settings change. Trying another
  layer or token counts as a hyperparameter, so it can be a side result but not the
  main one.
- Every improvement needs an evaluation on the same datasets and seeds as the
  reproduction, compared against our reproduced baseline with a paired test across
  datasets, plus a discussion of where it helps and where it doesn't.
- No pretraining. The demonstrator ruled it out, so everything uses the released
  Mantis-8M weights.
- About 2.25 pages. Sections 2 and 3 take about 2.75 of the 6 content pages and
  Section 1 about 0.75, which leaves roughly 2.25 for Section 4 and a short
  conclusion. References don't count towards the 6.

## 2. What Section 3 found

Each improvement in Section 4 below follows from one of these findings. Say so in
the report (for example, "Section III-B showed X, so we test Y"), so the reader
can see why each improvement was chosen.

| # | Finding (numbers from `results/summary.json`) | Leads to |
|---|---|---|
| F1 | Zero-shot Mantis matches the paper: 0.7821 against 0.7812 over 158 datasets. | A reliable baseline to improve on. |
| F2 | MiniRocket beats zero-shot Mantis by 5.6 pp (0.8378 against 0.7821, better on 126 of 158 datasets), and also beats fine-tuned Mantis on the 113 fine-tuning datasets (0.8565 against 0.8395). | Improvement A |
| F3 | The largest zero-shot gaps are on long series (over 1,000 steps, 26 datasets) and unequal-length series (16 datasets). Resizing everything to 512 loses information, and MiniRocket does much better on the same long series. | Improvement C |
| F4 | Temperature scaling makes calibration worse when the validation split is small: ECE goes up by 2.2 pp on the 32 datasets with at most 10 validation cases and down by 7.3 pp on the other 81. | Improvement B |
| F5 | Fine-tuning adds 3.9 pp over zero-shot but needs a GPU, while zero-shot embedding of all 158 datasets takes 9 minutes. | Keep improvements on the zero-shot side where possible. |

## 3. Getting set up (about an hour, mostly downloads)

### 3.1 Clone and install

```bash
git clone https://github.com/Sai-prabha/mantis-reproduction
cd mantis-reproduction
conda create -n mantis python=3.11 -y
conda activate mantis
pip install -r requirements.txt
```

### 3.2 Keep data and caches outside the repo

Add these lines to `~/.zshrc` (or `~/.bashrc`) and open a new terminal:

```bash
export MANTIS_DATA_DIR=~/mantis-work/data
export MANTIS_CACHE_DIR=~/mantis-work/cache
```

### 3.3 Download the data and check your setup

```bash
python scripts/00_dataset_stats.py --datasets all
python scripts/01_check_and_benchmark.py
```

The first command downloads about 4.3 GB from Zenodo. Zenodo sometimes returns
HTTP 503; rerun the command and it carries on from where it stopped. EMOPain always
fails because it is no longer available. The second command prints your device
(`mps`, `cuda` or `cpu`) and some timings.

### 3.4 Build the zero-shot embeddings once

Every zero-shot improvement reuses the cached Mantis embeddings. Write the results
to your own CSV so the committed `results/zero_shot.csv` stays untouched:

```bash
python scripts/02_zero_shot.py --datasets all --out results/zs_check_$(whoami).csv
```

This takes about 10 minutes on an Apple M-series GPU and roughly an hour on a CPU.
Your accuracies should be within about 0.5 pp of `results/zero_shot.csv` for each
dataset. If they aren't, tell Sai before building anything on them.

### 3.5 Code you can reuse

| Need | Use |
|---|---|
| Train and test data, resized to 512, labels encoded | `src.data.load_dataset(name)` returns `X_train (n, channels, 512)`, `y_train`, `X_test`, `y_test`, `meta` |
| Mantis embeddings, cached to disk | `src.models.cached_embeddings(name, data, device)` returns `Z_train, Z_test, seconds, from_cache` |
| Mantis at another layer or token | `src.models.load_mantis(device, layer=..., token=...)` |
| Dataset lists | `configs.datasets.resolve(...)` with `"all"`, `"ucr"`, `"uea"`, `"tier1"`, `"unequal"` or `"A,B,C"` |
| Resumable CSV logging | `src.common.CsvLog(path, key_fields)` with `.done(**key)` and `.write(row)` |
| ECE, NLL, temperature scaling, IRM | `src.calibration.ece`, `nll`, `fit_temperature`, `IRM` |
| Saved fine-tuning outputs (no GPU needed) | `results/finetune_history/{dataset}__{regime}__s{seed}__lr{lr}.npz` holds `logits_val`, `y_val`, `logits_test` and `y_test` for every run |

`scripts/02_zero_shot.py` is a good template for a new script: it has the argument
parsing, the dataset loop, caching, failure logging and the CSV row format.

## 4. Recommended improvements

A and B are the core: one targets accuracy and one targets calibration, and each
follows from a Section 3 finding. Add C if there is time. A and B share no code,
so they can run in parallel, and how you split them is up to you.

### Improvement A: combine Mantis and MiniRocket features (accuracy)

Motivation: F2. MiniRocket beats Mantis by 5.6 pp, but Mantis still wins on 27
datasets. If the two make different mistakes, combining their features should beat
both. It also answers a question v1 never asks: does the foundation model add
anything on top of a strong classical method?

Method:

1. For each dataset and seed, compute MiniRocket features on the same resized input
   (`from aeon.transformations.collection.convolution_based import MiniRocket`; check
   the import path in aeon 1.6). Fit the transform on the training set only.
2. Load the cached Mantis embeddings for the same dataset.
3. Standardise each block separately (fit the scaler on the training set only),
   concatenate them, and train `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`,
   which is the classifier MiniRocket itself uses. Train the same classifier on each
   block on its own as well, so any gain can be put down to the fusion and not to the
   classifier.
4. MiniRocket produces about 10,000 features against 256 per channel for Mantis, so
   the larger block can drown out the smaller one. Try one simple re-weighting, such
   as scaling each block to the same total variance, and report both versions.

Script: `scripts/07_fusion.py`, writing `results/fusion.csv` with one row per dataset,
seed and variant (`mantis_ridge`, `minirocket_ridge`, `fusion`, `fusion_weighted`).

Evaluation: all 158 datasets with seeds 0, 1 and 2. Report the means, wins, ties and
losses, and a paired Wilcoxon test against zero-shot Mantis with the random forest
(our reproduced baseline) and against MiniRocket on its own. Break the results down
by the series-type groups from Section 3 (long, unequal length and the rest; see
`results/tables/zs_gaps_by_series_type.md`).

Cost: a few minutes per seed on a laptop, with no GPU needed once the embeddings
are cached.

Watch out for leakage: a scaler or transform fitted on training and test data
together gives over-optimistic results.

### Improvement B: calibration for small validation sets (calibration)

Motivation: F4. Temperature scaling, which v1 recommends, makes calibration worse on
32 of our 113 fine-tuned datasets, the ones with at most 10 validation cases. Small
datasets are exactly where people would want to use a foundation model.

Method: every fine-tuning run already saved its final validation and test logits in
`results/finetune_history/*.npz`, so this needs no GPU and no retraining. Compare
these methods on the 113 Full-regime runs (seed 0) and on the 30 tier-1 datasets
(3 seeds):

1. No calibration, as the baseline.
2. Temperature scaling on the validation split (v1's method, as in our reproduction).
3. A shrunk temperature, `T = w * T_val + (1 - w) * T_prior`, where `T_prior` is the
   median temperature fitted on the other datasets (leave one dataset out, so a dataset
   never informs its own prior) and `w = n_val / (n_val + k)`. Choose `k` by leaving
   one dataset out as well, never on the test set.
4. A threshold rule: only calibrate when `n_val` is above a threshold, chosen the
   same way.
5. IRM, for comparison.

Script: `scripts/08_calibration_small_val.py`, writing `results/calibration_small_val.csv`
with ECE and NLL per dataset, seed and method.

Evaluation: mean ECE and NLL for each method, overall and split at `n_val <= 10`,
with wins, ties and losses and a Wilcoxon test against no calibration and against
plain temperature scaling. A reliability diagram for one small and one large dataset
would make a good figure.

Cost: seconds. This is the safest of the three to finish.

Watch out: the test logits are for evaluation only. The value of `k`, the threshold
and the prior must all come from validation data or from other datasets.

### Improvement C (if there is time): long series (representation)

Motivation: F3. Long series lose most of their detail when squeezed into 512 steps,
and that is where Mantis falls furthest behind MiniRocket.

Pick one of these:

- Windowing: cut each series longer than 512 into overlapping 512-step windows (for
  example with a stride of 256), embed every window with Mantis, and average the
  window embeddings (or concatenate their mean and max). Shorter series are left as
  they are.
- Multiple resolutions: embed each series at lengths 256, 512 and 1024 and
  concatenate the embeddings. `MantisV1`'s convolution kernel is sized for 512, so
  check that the other lengths run before relying on them.

Evaluation: the 26 long datasets are the target, but run all 158 to show nothing
gets worse elsewhere. Compare with zero-shot Mantis and with MiniRocket.

Cost: embedding several windows per series multiplies the time on long datasets,
though it should still be under an hour on a GPU.

### Quick side results

`02_zero_shot.py` already supports these, so each is one command. Report them in a
sentence or a small table and describe them as hyperparameter checks:

```bash
python scripts/02_zero_shot.py --datasets all --layer 2 --out results/zs_layer2.csv
python scripts/02_zero_shot.py --datasets all --token mean_token --out results/zs_mean_token.csv
```

The MantisV2 paper (arXiv 2602.17868) reports that an intermediate layer improves
zero-shot accuracy on UCR, so `--layer 2` tests that claim on Mantis-8M directly.

`docs/STATUS.md` lists a few more options (logistic regression, layer pruning,
zero-shot calibration, catch22 fusion).

## 5. Rules for every Section 4 experiment

These keep Section 4 to the same standard as Section 3 and count towards the
Section 5 marks.

1. Use the same protocol as the reproduction: datasets from `configs/datasets.py`,
   seeds 0, 1 and 2, and the pre-processing in `src/data.py`. Change one thing at a time.
2. Never select anything on the test set. Hyperparameters come from training data, a
   validation split or other datasets. Section 3 shows how much test-set selection
   inflates results.
3. Log every run through `CsvLog` to a new CSV in `results/`, with the script name,
   settings, seed, accuracy (or ECE), run time and `env_info(device)`.
4. Write one summary script (`scripts/09_summarise_s4.py`) that turns those CSVs into
   the Section 4 tables and figures, so no number in the report is typed by hand.
5. Add a "Section 4: improvements" block to the README commands table, with the exact
   command for each result. The spec marks this.
6. Report the cost (time and memory) next to the accuracy. A 1 pp gain at ten times
   the cost is a different result from a 1 pp gain for free.
7. Report what didn't work as clearly as what did. A properly tested negative result
   still counts.

## 6. Working in the repo together

- Work on your own branch (`git checkout -b section4-<your-name>`), push it, and open a
  pull request into `main` when a result is finished.
- Don't edit files that someone else's script writes, and don't run the same script
  into the same CSV on two machines, because merge conflicts in CSVs are painful. Use
  your own output file.
- `data/` and `cache/` stay out of git; each machine builds its own.
- Keep commit messages to one line saying what changed and why.

## 7. Abstract, Section 1, conclusion and the work-plan page

### Title, abstract and introduction

A title, an abstract of 150 to 200 words (what we reproduced, the headline numbers
and what we improved) and a short introduction leading into Section 1. Write the
abstract last, once Section 4 has numbers.

### Section 1: Motivation and impact (10 marks, about 0.75 page including the abstract and introduction)

The spec asks why we chose the paper and what impact it has on the AI/ML community.
Points to cover:

- Foundation models for time series classification are rare compared with
  forecasting. Mantis is one of the first open, lightweight ones: 8M parameters,
  pretrained with contrastive learning, with open code and weights.
- Its practical claims (runs on modest hardware, works without fine-tuning, well
  calibrated) matter most to people with small labelled datasets, which is where
  foundation models are supposed to help.
- Why reproduce it: reproducibility is a known problem in ML (see the spec's reading
  list), and foundation-model papers often compare only against other foundation
  models (the spec's reading on baselines). Section 3 backs this up, since MiniRocket
  beats Mantis.
- A short paragraph on what we did and found, with two or three numbers from
  Section 3, and the repo link.

Cite the paper as arXiv 2502.15637v1, with the "v1", because later versions describe
a different model.

### Conclusion (a few sentences at the end of Section 4)

What reproduced, what didn't, what the improvements added, and one line on future work.

### Work-plan page (1 extra page, required)

The spec asks for the project work plan, how the work was divided, and what each
member contributed in the end. A table with task, owner, dates and status works well.
Keep short notes as you go; the commit history and the group chat give the dates.

## 8. Checklist before submission

- [ ] Every part in the table in Section 0 has an owner agreed in the group chat.
- [ ] Section 4 results merged into `main`, with the README commands table updated.
- [ ] Everyone's references merged into one `.bib` file (Sai's list is ready).
- [ ] The whole report compiled in the official IEEE Overleaf template: 6 content
      pages, 1 work-plan page, and references on top.
- [ ] Repo made public before submission (markers can't see a private repo), and the
      link in the report works.
- [ ] One person, the group contact, submits the PDF on Brightspace.
