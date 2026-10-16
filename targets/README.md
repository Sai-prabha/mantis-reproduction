# Paper target numbers (Mantis v1, arXiv 2502.15637v1)

`scripts/06_summarise.py` compares our results with these files. They are
transcribed from the **appendix** of v1, because the main-text figures print
no numbers. Never type a number here that is not in the PDF.

| File | Source in v1 | Columns |
|---|---|---|
| `v1_table3_zeroshot.csv` | Appendix Table 3 (zero-shot, RF, 159 datasets) | `dataset, mantis, nutime, moment` |
| `v1_table4_finetune.csv` | Appendix Table 4 (full fine-tuning, 131 datasets, **best-epoch** accuracy) | `dataset, mantis, nutime, moment, gpt4ts, units` |
| `v1_table6_regimes_last.csv` | Appendix Table 6 (RF/Head/Scratch/Full, 151 datasets, last epoch, fixed LR 2e-4) | `dataset, rf, head, scratch, full` |
| `v1_table7_regimes_best.csv` | Appendix Table 7 (same, best epoch) | `dataset, rf, head, scratch, full` |

Every model column has a matching `_std` column (the ± value in the PDF).

**Status: filled (16 Oct 2026).** Extracted with `pdftotext -layout` and a regex
parser, then checked against every average the paper prints in its main-text
figures. All match to 4 decimal places: Table 3 Mantis 0.7816 and MOMENT 0.7663
(Fig. 3/6), Table 4 Mantis 0.8408 (Fig. 5), Table 6 0.7883/0.7745/0.7928/0.8436
and Table 7 0.7883/0.7798/0.8233/0.8625 (Fig. 7). Row counts: 159, 131, 151, 151.
`InsectWingbeatSubset` in the PDF is stored as `InsectWingbeat`.

Note for the report: v1 Table 8 (RF, UEA-27, mean 0.7048) disagrees with
Table 3 on all 27 UEA datasets (Table 3 UEA mean 0.7170). We compare with
Table 3, which our numbers match more closely.

Rows are pre-filled with dataset names; fill the empty cells.

**Before filling:**
- [Verify] the column set of each table in the PDF. If a table has different
  models or extra columns (for example ± std), add columns rather than dropping
  information. Keep `dataset` and `mantis` names exactly as they are.
- Use accuracy as a fraction (0.8123), not a percentage.
- Dataset names must match the aeon names used in `configs/datasets.py`
  (for example `InsectWingbeat`, not `InsectWingbeatSubset`).

**Faster than typing:** `pdftotext -layout 2502.15637v1.pdf v1.txt`
(poppler, `brew install poppler`) usually gives clean table rows that a short
script can parse. Spot-check at least 10 rows against the PDF afterwards and
say in the report that you did.
