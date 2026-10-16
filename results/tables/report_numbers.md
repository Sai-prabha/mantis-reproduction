| section   | statement                                                              | value                                                                     |
|:----------|:-----------------------------------------------------------------------|:--------------------------------------------------------------------------|
| III-B     | MOMENT zero-shot, ours vs v1 Table 3                                   | 0.7643 vs 0.7658 (n=158, p=0.59)                                          |
| III-B     | Mantis minus MOMENT on UCR only                                        | +1.57 pp (n=128, p=0.058)                                                 |
| III-B     | Embedding time, Mantis vs MOMENT (all datasets)                        | 9.3 min vs 6.6 h (42 times faster)                                        |
| III-B     | MiniRocket vs zero-shot Mantis                                         | 0.8378 vs 0.7821 (n=158)                                                  |
| III-B     | MiniRocket total run time, all datasets and seeds                      | 11 min                                                                    |
| III-B     | Fine-tuned Mantis (Full, last epoch, seed 0) vs MiniRocket             | 0.8395 vs 0.8565 (n=113; MiniRocket better on 72, Mantis on 34, p=0.0002) |
| III-B     | EthanolLevel, MiniRocket vs Mantis zero-shot                           | 0.61 vs 0.29                                                              |
| III-A     | UEA: mean |ours - v1 Table 3| vs mean |ours - v1 Table 8|              | 1.1 pp vs 3.1 pp (Table 8 mean 0.7048, Table 3 mean 0.7170)               |
| III-A     | Accuracy change from loading aeon's _eq files (15 datasets)            | mean 6.1 pp, max 18.7 pp (PickupGestureWiimoteZ)                          |
| III-D     | Full minus Scratch (last epoch), at most 40 training cases vs the rest | +9.4 pp (n=10) vs +4.8 pp (n=20); largest +36.7 pp (BeetleFly)            |
| III-C     | Package defaults minus paper protocol (Full, last epoch)               | +0.6 pp on 5 datasets, 4.5 times the run time                             |
| III-E     | ECE raw / TS / IRM, Full, seed 0, both tiers (v1 Figure 9)             | 0.120 / 0.074 / 0.071 (n=113) vs v1 0.113 / 0.067 / 0.061                 |
| III-E     | Fitted temperature above 1, and TS raising ECE                         | T > 1 on 75 of 113 (mean 2.36); TS worse on 33                            |
| II-F      | Fine-tuning run time (all runs)                                        | 628 runs, 3.3 h                                                           |