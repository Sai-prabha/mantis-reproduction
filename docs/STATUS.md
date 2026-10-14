# Project status (for the group)

Last updated: 14 October 2026, by Sai.

## Where things stand

| Report section | Marks | Owner | Status |
|---|---|---|---|
| 1. Motivation and impact | 10 | *TBC* | Not started |
| 2. Approach to reproduction | 10 | Sai | Plan done ([guide](sections_2_3_guide.md)); write-up waits for results |
| 3. Lessons learned | 30 | Sai | Plan done; experiments ready to run |
| 4. Improvements | 30 | *Section 4 owner* | Candidate list below; not started |
| 5. GitHub repo + README | 20 | Shared | Repo, scripts, README skeleton done; results table fills itself |
| Work plan page | – | *TBC* | Not started |

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
- [x] Offline smoke test passes (`python tests/smoke_test.py`, random weights + synthetic data).
- [x] Target templates for the paper's appendix tables (`targets/`), with all 159 / 131 dataset names.
- [x] README with environment, data, commands and an auto-generated paper-vs-ours table.
- [x] Code-level findings for Section 3 (details in the guide, Section 1):
  - the package's `fine_tuning_type="scratch"` does not re-initialise weights;
  - the package's fine-tuning defaults differ from the paper's settings;
  - aeon's loading defaults changed between versions (raw vs pre-processed data);
  - removing v1's stated exclusions gives exactly its 131 fine-tuning datasets.

## What's next

1. Install and run `00_dataset_stats.py`, `01_check_and_benchmark.py` on the M5 Pro; pin the checkpoint hash.
2. Transcribe v1 Appendix Tables 3 and 4 into `targets/` (spot-check 10 rows).
3. Zero-shot on all 159; MiniRocket in parallel.
4. Tier-1 fine-tuning overnight; package-defaults comparison.
5. MOMENT (separate venv); NuTime attempt.
6. `06_summarise.py`, then write Sections 2 and 3.

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
