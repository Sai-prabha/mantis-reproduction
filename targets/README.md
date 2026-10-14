# Paper target numbers (Mantis v1, arXiv 2502.15637v1)

`scripts/06_summarise.py` compares our results with these files. They are
transcribed from the **appendix** of v1, because the main-text figures print
no numbers. Never type a number here that is not in the PDF.

| File | Source in v1 | Columns |
|---|---|---|
| `v1_table3_zeroshot.csv` | Appendix Table 3 (zero-shot, RF, 159 datasets) | `dataset, mantis, nutime, moment` |
| `v1_table4_finetune.csv` | Appendix Table 4 (full fine-tuning, 131 datasets) | `dataset, mantis, nutime, moment, gpt4ts, units` |

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
