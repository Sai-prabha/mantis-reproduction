# Inference experiments on frozen Mantis-8M

Each number is the mean over the seeds recorded for that dataset. Mean accuracy is for orientation: accuracies on different datasets are not on a common scale (Demšar, 2006). Comparisons use wins/ties/losses, the median gap, and a two-sided Wilcoxon signed-rank test on datasets. Zero gaps are dropped before ranking. α = 0.05.

## Against the frozen encoder

| Comparison                                    | Datasets | Frozen | This method |   Mean Δ | Median Δ | W/T/L   | Wilcoxon p |
| --------------------------------------------- | -------: | -----: | ----------: | -------: | -------: | ------- | ---------- |
| Channel pool, univariate (reproduction check) |       33 | 0.7710 |      0.7710 | +0.00 pp | +0.00 pp | 0/33/0  | all ties   |
| Channel pool, multivariate                    |       29 | 0.7385 |      0.6827 | -5.58 pp | -3.51 pp | 6/3/20  | <0.001     |
| Cross-channel views, multivariate             |       29 | 0.7385 |      0.7445 | +0.60 pp | +0.11 pp | 15/3/11 | 0.237      |
| Temporal halves, raw length at least 1024     |       26 | 0.6902 |      0.7028 | +1.26 pp | +1.93 pp | 18/3/5  | 0.038      |
| Temporal halves not added (short series)      |       36 | 0.8031 |      0.8031 | +0.00 pp | +0.00 pp | 0/36/0  | all ties   |

| Check                                         | Datasets | Repo zero-shot | Refit here |   Mean Δ | Median Δ | W/T/L    | Wilcoxon p |
| --------------------------------------------- | -------: | -------------: | ---------: | -------: | -------: | -------- | ---------: |
| Frozen RF refit against results/zero_shot.csv |       62 |         0.7564 |     0.7558 | -0.07 pp | +0.00 pp | 15/28/19 |      0.673 |

Largest absolute gap on a dataset: 0.0267. The repository run used `n_jobs=-1`. This folder refits the same forest with one thread, so a gap of a few tenths of a percentage point is thread noise in the forest, not a new encoder.

## Per dataset

Accuracies for every dataset are in `tables/by_dataset.md` (62 rows).

## Largest gaps

### Cross-channel views

| Dataset               | Channels | Min length | Train | Frozen | This method |        Δ |
| --------------------- | -------: | ---------: | ----: | -----: | ----------: | -------: |
| HandMovementDirection |       10 |        400 |   160 | 0.2027 |      0.2838 | +8.11 pp |
| MotorImagery          |       64 |       3000 |   278 | 0.4767 |      0.5167 | +4.00 pp |
| Libras                |        2 |         45 |   180 | 0.8981 |      0.9296 | +3.15 pp |
| EigenWorms            |        6 |      17984 |   128 | 0.7405 |      0.7125 | -2.80 pp |
| DuckDuckGeese         |     1345 |        270 |    50 | 0.4067 |      0.4333 | +2.67 pp |
| SelfRegulationSCP2    |        7 |       1152 |   200 | 0.4907 |      0.5167 | +2.59 pp |
| ERing                 |        4 |         65 |    30 | 0.9420 |      0.9667 | +2.47 pp |
| LSST                  |        6 |         36 |  2459 | 0.6049 |      0.6288 | +2.39 pp |

### Temporal halves, where they were added

| Dataset              | Channels | Min length | Train | Frozen | This method |         Δ |
| -------------------- | -------: | ---------: | ----: | -----: | ----------: | --------: |
| PigAirwayPressure    |        1 |       2000 |   104 | 0.4936 |      0.3413 | -15.22 pp |
| PigCVP               |        1 |       2000 |   104 | 0.7885 |      0.6587 | -12.98 pp |
| SemgHandSubjectCh2   |        1 |       1500 |   450 | 0.8015 |      0.8874 |  +8.59 pp |
| EthanolConcentration |        3 |       1751 |   261 | 0.2750 |      0.3561 |  +8.11 pp |
| EthanolLevel         |        1 |       1751 |   504 | 0.2953 |      0.3747 |  +7.93 pp |
| CinCECGTorso         |        1 |       1639 |    40 | 0.6739 |      0.7273 |  +5.34 pp |
| MotorImagery         |       64 |       3000 |   278 | 0.4767 |      0.5300 |  +5.33 pp |
| Rock                 |        1 |       2844 |    20 | 0.7733 |      0.8267 |  +5.33 pp |

### Channel pool, multivariate

| Dataset                   | Channels | Min length | Train | Frozen | This method |         Δ |
| ------------------------- | -------: | ---------: | ----: | -----: | ----------: | --------: |
| InsectWingbeat            |      200 |          3 |  1000 | 0.6073 |      0.2333 | -37.40 pp |
| Libras                    |        2 |         45 |   180 | 0.8981 |      0.6981 | -20.00 pp |
| UWaveGestureLibrary       |        3 |        315 |   120 | 0.8156 |      0.6312 | -18.44 pp |
| JapaneseVowels            |       12 |          7 |   270 | 0.9622 |      0.7874 | -17.48 pp |
| RacketSports              |        6 |         30 |   151 | 0.9254 |      0.7807 | -14.47 pp |
| ArticularyWordRecognition |        9 |        144 |   275 | 0.9944 |      0.8889 | -10.56 pp |
| MotorImagery              |       64 |       3000 |   278 | 0.4767 |      0.5700 |  +9.33 pp |
| SpokenArabicDigits        |       13 |          4 |  6599 | 0.9579 |      0.8784 |  -7.94 pp |


## Multivariate comparison

Critical difference diagram on the multivariate datasets that have all three methods. α = 0.05, Wilcoxon with Holm correction when the installed aeon accepts that argument. Saved to `cd_multivariate.pdf`.

| Comparison on multivariate datasets                   | Datasets |   Mean Δ | Median Δ | W/T/L   | Wilcoxon p |
| ----------------------------------------------------- | -------: | -------: | -------: | ------- | ---------: |
| Channel pool + RF200 vs Frozen encoder + RF200        |       29 | -5.58 pp | -3.51 pp | 6/3/20  |     <0.001 |
| Cross-channel views + RF200 vs Frozen encoder + RF200 |       29 | +0.60 pp | +0.11 pp | 15/3/11 |      0.237 |
| Cross-channel views + RF200 vs Channel pool + RF200   |       29 | +6.18 pp | +4.30 pp | 21/2/6  |     <0.001 |

## Calibration of the frozen forest

| Group        | Datasets | ECE raw | ECE TS |      TS Δ |   TS p | ECE IRM |     IRM Δ |  IRM p | Acc raw | Acc IRM |
| ------------ | -------: | ------: | -----: | --------: | -----: | ------: | --------: | -----: | ------: | ------: |
| all          |       62 |  0.2115 | 0.0806 | -13.09 pp | <0.001 |  0.0744 | -13.71 pp | <0.001 |  0.7410 |  0.7336 |
| univariate   |       33 |  0.1794 | 0.1124 |  -6.70 pp |  0.002 |  0.0869 |  -9.25 pp | <0.001 |  0.7476 |  0.7404 |
| multivariate |       29 |  0.2479 | 0.0444 | -20.35 pp | <0.001 |  0.0601 | -18.79 pp | <0.001 |  0.7336 |  0.7259 |

Δ is calibrated ECE minus raw ECE, so a negative value is a lower error. Temperature scaling changed argmax accuracy by at most 0.0000. The forest was fit on 80% of the training split, so these accuracies are not the full-training numbers in the view table.

## Forward-pass bottlenecks

Total parameters: 8,103,936.

| Transformer block | Parameters |
| ----------------: | ---------- |
|                 0 | 1,312,768  |
|                 1 | 1,312,768  |
|                 2 | 1,312,768  |
|                 3 | 1,312,768  |
|                 4 | 1,312,768  |
|                 5 | 1,312,768  |

Blocks after the third hold 3,938,304 parameters (48.6% of the network).

| Forward pass                                 | Seconds |
| -------------------------------------------- | ------: |
| layer -1, model built that way               |   1.430 |
| layer 0, attribute set on the full model     |   0.558 |
| layer 0, model constructed with that request |   0.518 |

First-layer time divided by full forward time: 0.39.

| Channels | Seconds for 128 series | Seconds per channel |
| -------: | ---------------------: | ------------------: |
|        1 |                  0.712 |               0.712 |
|        4 |                  2.843 |               0.711 |
|       16 |                 11.555 |               0.722 |
|       32 |                 27.282 |               0.853 |

| Raw length | Seconds to resize 128 series onto 512 |
| ---------: | ------------------------------------: |
|         64 |                                 0.019 |
|        512 |                                 0.017 |
|       4096 |                                 0.020 |
|      16384 |                                 0.034 |

## Figures

Captions below are for the report. The figures themselves carry the axis labels only.

- `figures/scatter_channel_pool.pdf`: Channel pooling against the frozen encoder. Each point is one dataset, averaged over seeds. The solid line is equal accuracy and the dotted lines are ±5 percentage points. Univariate points lie on the line: with one channel the average leaves the embedding unchanged. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points.
- `figures/scatter_cross_channel.pdf`: Cross-channel views against the frozen encoder, multivariate datasets only. Each point is one dataset, averaged over seeds. The solid line is equal accuracy and the dotted lines are ±5 percentage points. Points above the line are gains. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points.
- `figures/scatter_segment.pdf`: Temporal halves against the frozen encoder, restricted to series of raw length at least 1024, where the halves are actually embedded. Each point is one dataset, averaged over seeds. The solid line is equal accuracy and the dotted lines are ±5 percentage points. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points.
- `figures/delta_vs_channels.pdf`: Accuracy change from cross-channel views against the number of channels. The change is this method minus the frozen encoder, in percentage points, averaged over seeds. The horizontal line is no change. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points.
- `figures/delta_vs_train_size.pdf`: Accuracy change from cross-channel views against the number of training series. The change is this method minus the frozen encoder, in percentage points, averaged over seeds. The horizontal line is no change. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points.
- `figures/delta_vs_length.pdf`: Accuracy change from temporal halves against raw length, on series of length at least 1024. The change is this method minus the frozen encoder, in percentage points, averaged over seeds. The horizontal line is no change. Names mark gaps of at least 5 percentage points, stacked in the same vertical order as the points.
- `figures/tradeoff_accuracy_vs_time.pdf`: Accuracy change against the extra time spent embedding the added series. The change is the method minus the frozen encoder, in percentage points, averaged over seeds. The horizontal line is no change. Cross-channel views add three series on every multivariate dataset. Temporal halves add two embeddings, and only on long series.
- `figures/cd_multivariate.pdf`: Average ranks of the frozen encoder, channel pooling, and cross-channel views on the multivariate datasets. Rank 1 is best. A bar joins methods that the Wilcoxon test with Holm correction does not separate at α = 0.05. The classifier is RF200 in every case.
- `figures/calibration_ece.pdf`: Mean expected calibration error of the frozen forest, with 10 equal-width bins, before and after post-hoc calibration. The forest is fit on 80% of the official training split. Temperature scaling and isotonic regression are fit on the remaining 20% and scored on the official test split. A lower bar is a better-calibrated probability.
- `figures/bottleneck_channels.pdf`: Forward time of frozen Mantis-8M against the number of channels. Each point is one synthetic batch of 128 series of length 512, timed on CPU. Time grows with the channel count because each channel is embedded on its own.

## Reading for the report

The encoder weights are the released Mantis-8M checkpoint. The classifier is the same 200-tree forest used in the reproduction. The experiments change the array that forest reads. On 33 univariate datasets, averaging the single channel block leaves the embedding unchanged. The largest absolute accuracy gap against the frozen forest is 0.0000 (mean +0.00 pp, 0/33/0 wins/ties/losses). On 29 multivariate datasets, that same average changes accuracy by -5.58 pp on average (median -3.51 pp, 6/3/20, p <0.001). A loss here means the forest was using channel identity, which the average throws away. Adding the per-series mean, standard deviation, and leading component on those multivariate datasets changes accuracy by +0.60 pp on average (median +0.11 pp, 15/3/11, p 0.237). The extra forward pass is three series, whatever the channel count. On 26 datasets whose raw series are at least 1024 steps, embedding each temporal half changes accuracy by +1.26 pp on average (median +1.93 pp, 18/3/5, p 0.038). On 36 shorter datasets the halves are not added. The largest absolute gap against the frozen forest is 0.0000, which is the check that this path reproduces the published inference. Permuting test channels, with the forest trained on the true order, changes accuracy by -34.45 pp on 29 multivariate datasets (median -35.00 pp, 3/0/26, p <0.001). A negative gap means the channel index carries class information the average would delete. Use the scatter plots for the paired view, the delta plots for where a gain concentrates, and the time plot before describing a method as cheap. A non-significant p-value with a consistent median gap is a small sample, not evidence that the two methods are equal.
