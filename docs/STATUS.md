# Project status

Last updated: 16 October 2026, by Sai.

## Who is doing what

| Part of the report | Marks | Who | Status |
|---|---|---|---|
| Title, abstract, introduction | (in 1) | to agree | Not started |
| 1. Motivation and impact | 10 | to agree | Not started |
| 2. Approach to reproduction | 10 | Sai | Draft done (about 0.75 page) |
| 3. Lessons learned | 30 | Sai | Draft done (about 2 pages); all experiments finished |
| 4. Improvements, plus a short conclusion | 30 | Andrew and Abel | Not started; plan in [SECTION4_GUIDE.md](SECTION4_GUIDE.md) |
| 5. GitHub repo and README | 20 | Sai; each member adds their own commands | Reproduction commands, results table and deviations done |
| Work-plan page | required | to agree; everyone contributes | Not started |
| References | not counted | everyone | Sai's references ready |

Sections 2 and 3 take about 2.75 of the 6 content pages. References are on top of that.

## Decisions already made

- Paper: Mantis v1, arXiv [2502.15637v1](https://arxiv.org/pdf/2502.15637v1), approved by the lecturer. Always cite the v1 link, because the un-versioned link opens a later paper about a different, 4.2M-parameter model.
- Model: Mantis-8M (`paris-noah/Mantis-8M`, 8.11M parameters), code class `MantisV1`.
- Pretraining is out of scope and fine-tuning on a subset of datasets is allowed (both agreed with the demonstrator).
- Zero-shot runs on all of v1's datasets that still load: 158 of 159 (EMOPain is no longer available).
- Fine-tuning: tier 1 is 30 UCR datasets with every regime and 3 seeds; tier 2 is the other 83 UCR datasets of v1's fine-tuning set, full fine-tuning only, 1 seed.
- Baselines: MiniRocket and MOMENT-1-large. NuTime was dropped for time.
- Compute: Sai's MacBook Pro (Apple M5 Pro, MPS).

## Done

- [x] Experiment code: scripts `00` to `06` (see the README). Every run logs to `results/*.csv` and resumes after an interruption.
- [x] Zero-shot on 158 datasets with 3 seeds, plus MiniRocket and MOMENT-1-large on the same 158.
- [x] Fine-tuning: tier 1 (30 datasets, RF/Head/Full/Scratch, 3 seeds, calibration), tier 2 (83 datasets, Full, 1 seed), and the package defaults on 5 datasets.
- [x] v1 appendix Tables 3, 4, 6 and 7 in `targets/`, checked against every average the paper prints in its figures.
- [x] Unequal-length check with aeon's `_eq` files: it does not explain the gaps on those datasets.
- [x] README results table comparing every headline number with the paper.
- [x] Sections 2 and 3 drafted.

## Next

1. Agree who writes the title, abstract and introduction, Section 1 and the work-plan page.
2. Section 4: follow [SECTION4_GUIDE.md](SECTION4_GUIDE.md). Build the embedding cache once (step 3.4 of the guide), then reuse it.
3. Merge everyone's references and make the repo public before submission.

## Other Section 4 ideas

The guide recommends three improvements. These are further options, all on Mantis-8M without pretraining:

1. Intermediate-layer features (`--layer 2`). The MantisV2 paper (arXiv 2602.17868, Fig. 9b) reports 0.7921 rising to 0.7994 on UCR-128 for the original Mantis.
2. A combined token (`--token combined`): the CLS token concatenated with the mean of the other tokens. The authors did not test this on Mantis-8M.
3. Logistic regression on standardised embeddings instead of the random forest.
4. Layer pruning, from 8.1M down to about 2.9M parameters. `MantisV1` has no `remove_transf_layers()` (only `MantisV2` does), so this needs a small code change.
5. Calibrating zero-shot predictions (temperature scaling or isotonic regression on the classifier's probabilities). v1 only tested fine-tuned models; `ece_rf` in `results/zero_shot.csv` gives the starting point.
6. Fusion with catch22 features instead of, or as well as, MiniRocket.

Items 1 to 3 are close to hyperparameter changes, so on their own they would not meet the spec's requirement for a non-trivial improvement.

## Working together in this repo

- One branch per person (`git checkout -b section4-<your-name>`), merged into `main` through a pull request.
- Scripts append to the CSVs in `results/`. Don't edit them by hand, and don't run the same script into the same CSV on two machines.
- `data/` and `cache/` are not in git; each machine downloads and builds its own.
