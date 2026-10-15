| Setting                                           |   Datasets | Paper (v1)   | Ours     | Δ        | Wilcoxon p   |
|:--------------------------------------------------|-----------:|:-------------|:---------|:---------|:-------------|
| Zero-shot, UCR (RF200, last layer, CLS)           |          0 | pending      | pending  | pending  |              |
| Zero-shot, UEA (RF200, last layer, CLS)           |          0 | pending      | pending  | pending  |              |
| Zero-shot, other 4 (RF200, last layer, CLS)       |          0 | pending      | pending  | pending  |              |
| Zero-shot, UCR-128 cross-check (arXiv 2602.17868) |        128 | 0.7921       | 0.7925   | +0.04 pp |              |
| Full fine-tuning, tier 1 (3 seeds)                |         30 | pending      | 0.8445   | pending  |              |
| Full fine-tuning, tiers 1+2 (seed 0)              |        113 | pending      | 0.8395   | pending  |              |
| Full minus scratch, last epoch                    |         30 | +5.08 pp     | +6.35 pp |          |              |
| Full minus scratch, best epoch                    |         30 | +3.92 pp     | +5.55 pp |          |              |
| Zero-shot, Mantis minus MOMENT                    |          0 | pending      | pending  |          |              |