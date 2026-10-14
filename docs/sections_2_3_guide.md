# Guide: Sections 2 and 3 of the Mantis-8M Reproduction Report (COMP41850 AI4TS)

**Scope of this guide:** how to plan, run and write up

- **Section 2 [10/100]:** approach taken to reproduce the results
- **Section 3 [30/100]:** lessons learned (what works as published, what doesn't, other findings)

The companion code is this repository (scripts `00` to `06`). Each experiment below names the script that produces it.

**Reference paper:** Feofanov et al., *Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification*, arXiv 2502.15637**v1** (Feb 2025): https://arxiv.org/pdf/2502.15637v1. Always cite the explicit `v1` link; the un-versioned link opens v2, which describes a different 4.2M model.

**Model:** Mantis-8M, class `MantisV1`, checkpoint `paris-noah/Mantis-8M` (8.11M parameters).

**Confidence markers used below:**

| Marker | Meaning |
|---|---|
| [Certain] | Checked against the package source (`mantis-tsfm` 1.1.0), Hugging Face or the paper text |
| [Likely] | Strong inference |
| [Verify] | Check it yourself before relying on it |

---

## 0. Locked decisions

| Item | Decision |
|---|---|
| Paper | v1, **approved by the lecturer** |
| Pretraining | Out of scope (demonstrator) |
| Zero-shot | All 159 datasets: UCR-128, UEA-27, and the 4 others (Blink, EMOPain, MotionSenseHAR, SharePriceIncrease) if they load |
| Fine-tuning, tier 1 | 30 UCR datasets (list in `configs/datasets.py`), full protocol: 4 regimes × 3 seeds, LR grid on seed 0 |
| Fine-tuning, tier 2 | Optional: remaining 83 UCR datasets of the paper's fine-tuning set, full fine-tuning only, 1 seed, fixed LR |
| Baselines | MiniRocket (non-foundation model), MOMENT-1-large (zero-shot claim), NuTime (calibration comparator, only if its code runs; 2-hour cap) |
| Section 4 | Owned by teammate (later-paper improvements). Section 3 does not use them. |
| Compute | Your M5 Pro, 48 GB, MPS backend |

**Why tier 1 has 30 datasets, not "as many as possible":** each tier-1 dataset costs 18 runs (RF 3, head 5, full 5, scratch 5), and the 10 full and scratch runs are 100 epochs of training through the whole encoder. [Likely] 30 paired datasets give a Wilcoxon test enough power to detect a gap of a few points, and 3 seeds per dataset let you answer the spec's question "within expected variance, or systematic?". Tier 2 adds breadth after that, without variance estimates.

**How the 30 were chosen** (say this in Section 2): spread across training-set size (16 to 3,601 cases), series length (24 to 2,844), number of classes (2 to 50), and data quirks (2 with missing values, 2 with unequal lengths, 5 long series downsampled to 512). All 30 are inside the paper's 131-dataset fine-tuning set. `scripts/00_dataset_stats.py` prints the real characteristics; use that output, not the comments in the config.

---

## 1. Traps found in the code and data (each is a Section 3 finding)

These were found by reading `mantis-tsfm` 1.1.0 and aeon before running anything. The repository already handles them; the report should say what each one is and what it would have done to the numbers.

### 1.1 `fine_tuning_type="scratch"` does not train from scratch

[Certain] In `MantisTrainer`, "scratch" selects the same parameters as "full" (`self.fine_tuned_model.parameters()`) and never re-initialises them. Called on a pretrained network, "scratch" is just full fine-tuning under another name. Anyone reproducing v1's "pretrained beats scratch by 5.08%" this way would measure a gap of roughly zero and conclude pretraining doesn't help.

The repository builds a randomly initialised `MantisV1()` (no `from_pretrained`) for the scratch regime.

### 1.2 Package fine-tuning defaults differ from the paper

[Certain]

| Setting | Paper (v1, Section 3.3) | `mantis-tsfm` 1.1.0 default |
|---|---|---|
| Epochs | 100 | 500 |
| Head | LayerNorm + Linear | BatchNorm1d + Linear |
| Learning rate | Chosen from {1e-4, 2e-4, 1e-3} on a 20% validation split | Fixed 2e-4 |
| Schedule | Not stated | 10-epoch warm-up, then cosine decay to 0 |
| Optimiser | AdamW, weight decay 0.05 | Same |
| Batch size | 256 | Same |

`03_finetune.py --package-defaults` runs the defaults on 3 to 5 datasets so you can report the difference.

### 1.3 The same aeon call returns different data depending on the version

[Certain] In aeon 0.11 to 1.2, `load_classification` defaults to `load_equal_length=True, load_no_missing=True`: for datasets with unequal lengths or missing values, it silently returns the archive's pre-padded or pre-imputed copy. In aeon 1.6 both defaults are `False`, so the same one-line call returns the raw data. v1 (Feb 2025) doesn't say how it loaded data, so its unequal-length and missing-value datasets may have been preprocessed differently from yours, depending on the tool and version used.

The repository sets both flags explicitly to `False`, then applies its own documented steps: linear interpolation of missing values, then linear resizing to 512. Check the unequal-length and missing-value datasets when you look at the largest gaps in Z1.

### 1.4 Two classes load the same checkpoint

[Certain] The package ships `MantisV1` and a legacy `Mantis8M` class described as "functionally identical". `01_check_and_benchmark.py` compares their embeddings.

### 1.5 Apple Silicon was never tested by the authors

[Certain] The package documents only `cpu` and `cuda`. The `MantisV1` path has no hard-coded CUDA calls and no float64, so MPS should work. `01_check_and_benchmark.py` measures:

- MPS vs CPU embedding differences;
- whether two runs with the same seed give identical results (MPS is not fully deterministic);
- throughput and fine-tuning cost, plus a tier-1 time estimate.

---

## 2. What you're comparing against

[Certain] v1's main-text figures (3, 5, 7, 9) print no numbers, only relative statements. Targets come from:

| Target | Source | File in repo |
|---|---|---|
| Per-dataset zero-shot accuracy, 159 datasets | v1 Appendix Table 3 | `targets/v1_table3_zeroshot.csv` |
| Per-dataset fine-tuned accuracy, 131 datasets | v1 Appendix Table 4 | `targets/v1_table4_finetune.csv` |
| Full beats scratch by 5.08% (last epoch), 3.92% (best epoch) | v1 Fig. 7 text | hard-coded in `06_summarise.py` |
| Mantis beats NuTime by 1.77%, MOMENT by 1.53%; wins 74/53/38 | v1 Fig. 3 text | comparison in `06_summarise.py` |
| Original Mantis on UCR-128 (RF): 0.7921 | arXiv 2602.17868, Fig. 9b / Table 1 | cross-check in `06_summarise.py` |

You transcribe Tables 3 and 4 (instructions in `targets/README.md`). [Verify] the column set in the PDF first.

---

## 3. Order of work

| Step | Command | What it gives you | Time |
|---|---|---|---|
| 1 | `00_dataset_stats.py --datasets tier1` | Real tier-1 stats, data cached | minutes |
| 2 | `01_check_and_benchmark.py` | Checks 1.4, 1.5; tier-1 time estimate; checkpoint hash to pin | minutes |
| 3 | `02_zero_shot.py --datasets ECG200` | Sanity check against its Table 3 row | seconds |
| 4 | `00_dataset_stats.py --datasets all` | All 159 cached; which "others" fail | [Guessing] under an hour |
| 5 | `02_zero_shot.py` on ucr, uea, others | Z1 | hours |
| 6 | `04_minirocket.py --datasets all` | B1 | hours on CPU |
| 7 | `03_finetune.py --datasets tier1` | F1, F2, F3 | overnight (see step 2's estimate) |
| 8 | `03_finetune.py --package-defaults` on 3 to 5 datasets | Trap 1.2 | about an hour |
| 9 | `05_moment.py` (separate venv) | B2 | hours |
| 10 | NuTime attempt, 2-hour cap | B3 or "untested" | ≤ 2 h |
| 11 | Tier 2, if time | breadth | overnight |
| 12 | `06_summarise.py` | every table and figure | seconds |

Steps 5, 6 and 7 can run in parallel: zero-shot and fine-tuning use the GPU, MiniRocket uses CPU cores. [Likely] Running two GPU jobs at once on one Mac just slows both, so keep it to one GPU job at a time.

---

## 4. Section 2: Approach [10/100]

### What the marker wants

"Document all your reproduction steps and make it easy for others to verify your results." About 0.75 of a page.

### Structure

**2.1 Scope.**

- We reproduce v1's zero-shot and fine-tuning evaluations of Mantis-8M, using the authors' code and checkpoint.
- Pretraining is out of scope (4 × V100 GPUs, 100 epochs on ~7M series).
- Fine-tuning runs on a stated subset (agreed with the demonstrator).

**2.2 Model.** `MantisV1`, `paris-noah/Mantis-8M` at the pinned commit, parameter count as measured, `mantis-tsfm` 1.1.0.

**2.3 Data.**

- 159 datasets for zero-shot, official splits; say which of the 4 others loaded.
- Tier 1 and how it was chosen (Section 0).
- Preprocessing: raw files, missing values interpolated (our assumption; v1 is silent), every series resized to 512 (as v1 states), no extra normalisation.
- InsectWingbeat subset: stratified 1,000 / 1,000, seed 0 (our assumption; v1 is silent).

**2.4 Zero-shot protocol.** Final layer, CLS token, channels embedded independently and concatenated, RF with 200 trees and unlimited depth, seeds 0, 1, 2.

**2.5 Fine-tuning protocol.**

- Four regimes: RF on frozen features, head only, full fine-tuning, from scratch (randomly initialised encoder).
- Paper settings from Section 1.2, implemented with the package's own `FineTuningNetwork` and scheduler.
- LR chosen on seed 0 by final-epoch validation accuracy, reused for seeds 1 and 2.
- Models trained on the 80% split and not refit, because the 20% split is needed for calibration.
- Validation and test evaluated every 5 epochs, so "best epoch" is reported two ways: chosen on validation, and chosen on test (oracle).

**2.6 Calibration protocol.** ECE with 10 equal-width bins; temperature scaling and multi-class isotonic regression (IRM) fitted on the 20% split.

**2.7 Baselines.** MiniRocket and MOMENT-1-large on the identical preprocessed input; MOMENT embeds channels one at a time, like Mantis.

**2.8 Evaluation.** Per-dataset accuracy against the appendix tables, mean ± std over seeds, Wilcoxon signed-rank tests.

**2.9 Deviations table.** Copy it from the README (Section 5 there), so the two never disagree.

---

## 5. Section 3: Lessons learned [30/100]

### What the marker wants (from the spec)

1. Compare your numbers with the paper's.
2. Are discrepancies within variance, or systematic?
3. Which unreported details or preprocessing steps matter?
4. What works as published, what doesn't, and other findings.

Organise by **claim**, not by experiment.

### Z1: Zero-shot accuracy (core)

| | |
|---|---|
| **Claim** | Per-dataset accuracy in Table 3. |
| **Script** | `02_zero_shot.py`, `06_summarise.py` → `zs_paper_vs_ours_by_archive`, `zs_largest_gaps`, `zs_scatter.png` |
| **Interpret** | Share of datasets within 2 seed-std of the paper; Wilcoxon p. Where are the largest gaps, and do they share a property (short series stretched to 512, tiny training sets, unequal length, missing values)? Does UCR-128 land near the 0.7921 cross-check? |

### Z2: Code and environment checks (cheap, concrete)

From `01_check_and_benchmark.py`: parameter count vs 8.11M; legacy-class agreement; MPS vs CPU drift; same-seed determinism on MPS. One short paragraph.

### B1: Is a foundation model needed? (MiniRocket)

| | |
|---|---|
| **Claim** | Implicit: Mantis is a strong time series classifier. v1 compares only against other foundation models. |
| **Script** | `04_minirocket.py` → `baselines_archive_means` |
| **Interpret** | [Likely] MiniRocket beats Mantis zero-shot on UCR. If so, say it plainly and connect it to the spec's recommended reading on baselining. Note where Mantis wins (by archive, by dataset type). |

### B2: Mantis vs MOMENT (claim C2)

| | |
|---|---|
| **Claim** | Mantis beats MOMENT by 1.53% zero-shot; 74 vs 53 wins. |
| **Script** | `05_moment.py` → `summary.json: baselines` |
| **Interpret** | Does the gap's sign and size hold? Our MOMENT embeds channels separately; the paper doesn't say how it did. |

### F1: Fine-tuned accuracy (core)

Full regime on tier 1 against Table 4: Δ, Wilcoxon, seed spread. Plus the package-defaults comparison (trap 1.2).

### F2: Value of pretraining (core)

| | |
|---|---|
| **Claim** | Full fine-tuning beats scratch by 5.08% (last epoch) and 3.92% (best epoch). |
| **Script** | `03_finetune.py` → `ft_regimes`, `summary.json: full_minus_scratch` |
| **Interpret** | Does the gap hold on tier 1? Is it larger on the very small datasets? Report the scratch trap (1.1) here: the published code's "scratch" option would have hidden the effect. |
| **Best epoch** | Compare last epoch, validation-selected and test-selected accuracy. [Likely] If test-selected is clearly higher, "best epoch" numbers that peek at the test set overstate performance; v1 doesn't say how it chose the best epoch. |

### F3: Calibration (core for Mantis-8M)

| | |
|---|---|
| **Claim** | Mantis is the most calibrated foundation model, before and after post-hoc calibration. |
| **Script** | `03_finetune.py` → `calibration`, `reliability_full.png`, `summary.json: ts_gain_by_val_size` |
| **Interpret** | Raw ECE of fine-tuned Mantis; how much temperature scaling and IRM reduce it. On tiny datasets the 20% split is 3 or 4 cases, so the calibration maps are fitted on almost nothing; check whether the gain disappears there. "Most calibrated" needs a comparator: NuTime if it ran, otherwise state it as untested. v1 prints no ECE values, so you can't compare numbers directly; say so. |

### Not reproducible (state explicitly)

- Any claim needing pretraining (differential-block ablation, Table 5).
- Exact comparator numbers in Figures 3, 5 and 9 (no values printed).
- Adapters on UEA (Table 2): optional and heavy; drop unless time allows.

### "Missing details" checklist

- [ ] "Scratch" option doesn't re-initialise (1.1).
- [ ] Package defaults differ from the paper (1.2).
- [ ] aeon's version-dependent loading defaults (1.3).
- [ ] What happens after LR selection; how "best epoch" is chosen.
- [ ] Missing-value handling; InsectWingbeat subset; seed values.
- [ ] Main-text figures print no numbers.
- [ ] Pretraining corpus size (7M in v1 vs 1.89M in 2602.17868) and UCR overlap ("not part of pretraining" in v1 vs "UCR included" in 2602.17868).
- [ ] Un-versioned arXiv link shows a different model.
- [ ] Every surprise you hit while running.

### Writing template per finding

> **Claim:** [statement + number + figure/table].
> **Result:** [ours ± std, setting, hardware].
> **Verdict:** reproduced / partially / not reproduced / untestable.
> **Explanation:** within variance or systematic, and the likely cause.
> **Implication:** what a practitioner should take from it.

### Figures and tables (pick 4; space is tight)

1. Table: headline reproduction (zero-shot and fine-tuned means, paper vs ours, Δ, verdict). Also goes in the README.
2. Figure: zero-shot scatter, paper vs ours.
3. Table: four regimes plus full minus scratch, with last / validation-selected / oracle columns.
4. Table: ECE raw, after TS, after IRM; plus the reliability diagrams.

---

## Sources

- Mantis v1 (reference paper): https://arxiv.org/pdf/2502.15637v1
- MantisV2 and Mantis+ report (cross-check 0.7921, leakage table): https://arxiv.org/pdf/2602.17868
- Code: https://github.com/vfeofanov/mantis (`mantis-tsfm` 1.1.0: `architecture/version1.py`, `trainer/trainer.py`)
- Checkpoint: https://huggingface.co/paris-noah/Mantis-8M
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment (`momentfm` 0.1.4)
- NuTime: https://github.com/chenguolin/NuTime
- COMP41850 AI4TS project specification (criteria 2, 3 and 5)
