# Paper numbers (Mantis v1, arXiv 2502.15637v1)

`scripts/06_summarise.py` compares our results with the files in this folder.
The numbers come from the appendix of v1, because the figures in the main text
only print averages.

| File | Source in v1 | Columns |
|---|---|---|
| `v1_table3_zeroshot.csv` | Appendix Table 3 (zero-shot, RF, 159 datasets) | `dataset, mantis, nutime, moment` |
| `v1_table4_finetune.csv` | Appendix Table 4 (full fine-tuning, 131 datasets, best-epoch accuracy) | `dataset, mantis, nutime, moment, gpt4ts, units` |
| `v1_table6_regimes_last.csv` | Appendix Table 6 (RF, Head, Scratch, Full; 151 datasets; last epoch; fixed LR 2e-4) | `dataset, rf, head, scratch, full` |
| `v1_table7_regimes_best.csv` | Appendix Table 7 (as Table 6, best epoch) | `dataset, rf, head, scratch, full` |
| `v1_table8_uea.csv` | Appendix Table 8 (UEA-27: RF, and full fine-tuning with the best adapter) | `dataset, rf, full_best_adapter` |

Each model column has a matching `_std` column holding the ± value from the PDF.
Accuracies are fractions (0.8123), not percentages.

## How the files were made

The tables were extracted with `pdftotext -layout` and a short parser, then
checked against every average the paper prints in its figures. All of them match
to four decimal places:

- Table 3: Mantis 0.7816 and MOMENT 0.7663 (Figs. 3 and 6)
- Table 4: Mantis 0.8408 (Fig. 5)
- Table 6: 0.7883, 0.7745, 0.7928 and 0.8436 (Fig. 7)
- Table 7: 0.7883, 0.7798, 0.8233 and 0.8625 (Fig. 7)
- Table 8: 0.7048 and 0.7941 (Fig. 8)

The row counts are 159, 131, 151, 151 and 27. Dataset names follow aeon, so the
PDF's `InsectWingbeatSubset` is stored as `InsectWingbeat`.

v1 Table 8 reports random-forest accuracies on UEA-27 that differ from Table 3
on all 27 datasets (mean 0.7048 against 0.7170). We compare with Table 3, which
our results match more closely.
