# Project status (for the group)

Last updated: 16 October 2026, by Sai.

## Where things stand

| Report section | Marks | Owner | Status |
|---|---|---|---|
| 1. Motivation and impact | 10 | *TBC* | Not started |
| 2. Approach to reproduction | 10 | Sai | Draft done (about 0.75 page) |
| 3. Lessons learned | 30 | Sai | Draft done (about 2 pages); all experiments finished |
| 4. Improvements | 30 | *Section 4 owner* | Candidate list below |
| 5. GitHub repo + README | 20 | Shared | Commands, results table vs paper and deviations up to date |
| Work plan page | – | *TBC* | Not started |

Page budget agreed: Sections 2 + 3 take about 2.75 of the 6 content pages; references are separate.

## Agreed decisions

- **Paper:** Mantis v1, arXiv [2502.15637v1](https://arxiv.org/pdf/2502.15637v1). Approved by the lecturer. Always cite the `v1` link: the un-versioned link opens a later paper about a different, 4.2M-parameter model.
- **Model:** Mantis-8M (`paris-noah/Mantis-8M`, 8.11M parameters), code class `MantisV1`.
- **Pretraining:** out of scope (demonstrator). **Fine-tuning:** allowed on a subset (demonstrator).
- **Zero-shot:** all 159 datasets of v1 (UCR-128, UEA-27, 4 others if they load).
- **Fine-tuning:** tier 1 = 30 UCR datasets with the full protocol; tier 2 (83 more) optional.
- **Baselines:** MiniRocket, MOMENT-1-large; NuTime only if its code runs within 2 hours.
- **Compute:** Sai's M5 Pro (MPS).

## What's done

- [x] Experiment code: scripts `00` to `06` (see README). Every run logs to `results/*.csv` and resumes after interruption.
- [x] Zero-shot on 158 datasets (3 seeds); MiniRocket and MOMENT-1-large baselines on the same 158.
- [x] Fine-tuning: tier 1 (30 datasets, RF/Head/Full/Scratch, 3 seeds, calibration), tier 2 (83 datasets, Full, 1 seed), package defaults on 5 datasets.
- [x] v1 appendix Tables 3, 4, 6 and 7 transcribed into `targets/`, checked against every average printed in the paper's figures.
- [x] Sensitivity check on unequal-length datasets (aeon `_eq` files): ruled out as the cause of the gaps.
- [x] README results table compares every headline number with the paper.
- [x] Sections 2 and 3 drafted.

## What's next

1. Section 1 and the work-plan page (owners TBC).
2. Section 4 (see candidates below). Reuse the cached embeddings; don't re-extract.
3. Merge references from all sections; make the repo public before submission.

## Section 4 candidates (owner: Section 4)

All work on Mantis-8M with no pretraining. `02_zero_shot.py --layer/--token` already supports items 1 and 2, and `ece_rf` in `results/zero_shot.csv` is logged for item 6.

1. Intermediate-layer features (`--layer 2`). arXiv 2602.17868 Fig. 9b: 0.7921 → 0.7994 on UCR-128.
2. Combined token (`--token combined`): CLS concatenated with the mean of the other tokens. Not tested by the authors on Mantis-8M.
3. StandardScaler + logistic regression instead of the random forest.
4. Self-ensembling: embeddings at lengths 128, 256, 512, 1024 plus the first-difference signal, concatenated. Note: `MantisV1`'s convolution kernel (17) is sized for length 512.
5. Layer pruning: 8.1M → 2.9M parameters. `MantisV1` has no `remove_transf_layers()` (only `MantisV2` does), so it needs a small code change.
6. Calibration of zero-shot predictions (temperature scaling or isotonic regression on RF / logistic-regression probabilities). v1 only tested fine-tuned models.
7. Feature fusion with MiniRocket or catch22.

The spec requires at least one non-trivial improvement beyond a hyperparameter change: items 1 and 3 alone are arguably hyperparameters; 4 to 7 are safer as the headline contribution. Reuse the cached embeddings in `cache/emb/` rather than re-extracting.

## Working together in this repo

- Branch per person (`git checkout -b section4-abel`), pull request into `main`.
- `results/*.csv` is append-only and written by scripts: don't edit by hand; don't run the same script on two machines into the same CSV (merge conflicts).
- `data/` and `cache/` are git-ignored; each machine downloads its own.
- Declare generative AI use in the report as the module requires.
