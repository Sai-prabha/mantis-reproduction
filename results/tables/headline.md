| Setting                                                                    |   Datasets | Paper (v1)   | Ours     | Δ        | Wilcoxon p   |
|:---------------------------------------------------------------------------|-----------:|:-------------|:---------|:---------|:-------------|
| Zero-shot, UCR (RF200, last layer, CLS)                                    |        128 | 0.7922       | 0.7925   | +0.03 pp | 0.720        |
| Zero-shot, UEA (RF200, last layer, CLS)                                    |         27 | 0.7170       | 0.7203   | +0.34 pp | 0.499        |
| Zero-shot, other (3 of 4) (RF200, last layer, CLS)                         |          3 | 0.8935       | 0.8938   | +0.03 pp |              |
| Zero-shot, UCR-128 cross-check (arXiv 2602.17868)                          |        128 | 0.7921       | 0.7925   | +0.04 pp |              |
| Full fine-tuning, tier 1 (3 seeds), best epoch (ours chosen on test)       |         30 | 0.8413       | 0.8549   | +1.36 pp | 0.101        |
| Full fine-tuning, tier 1 (3 seeds), best epoch (ours chosen on validation) |         30 | 0.8413       | 0.8421   | +0.07 pp | 0.952        |
| Full fine-tuning, tier 1 (3 seeds), ours at last epoch                     |         30 | 0.8413       | 0.8445   | +0.32 pp | 0.984        |
| Full fine-tuning, tiers 1 and 2, best epoch (ours chosen on test)          |        113 | 0.8428       | 0.8551   | +1.23 pp | <0.001       |
| Full fine-tuning, tiers 1 and 2, best epoch (ours chosen on validation)    |        113 | 0.8428       | 0.8382   | -0.46 pp | 0.139        |
| Full fine-tuning, tiers 1 and 2, ours at last epoch                        |        113 | 0.8428       | 0.8395   | -0.33 pp | 0.282        |
| Full minus scratch, last epoch                                             |         30 | +5.14 pp     | +6.35 pp |          |              |
| Full minus scratch, best epoch                                             |         30 | +3.83 pp     | +5.55 pp |          |              |
| ECE, Full fine-tuning (seed 0), no post-hoc calibration                    |        113 | 0.1131       | 0.1200   | +0.69 pp |              |
| ECE, Full fine-tuning (seed 0), temperature scaling                        |        113 | 0.0673       | 0.0737   | +0.64 pp |              |
| ECE, Full fine-tuning (seed 0), IRM                                        |        113 | 0.0609       | 0.0706   | +0.97 pp |              |
| Zero-shot, MOMENT-1-large (RF200)                                          |        158 | 0.7658       | 0.7643   | -0.15 pp | 0.587        |
| Zero-shot, Mantis minus MOMENT                                             |        158 | +1.54 pp     | +1.78 pp |          | 0.018        |