"""Feature blocks and classifiers for the fusion experiment (scripts/07_fusion.py).

Three feature blocks, each computed per series:
  M  Mantis-8M frozen embedding (last layer, CLS token, 256 per channel), cached
  S  summary statistics of the series and of its first difference, per channel
  R  MiniRocket features (about 10,000), fitted on the training set only

Why S: Mantis instance-normalises every series inside its forward pass, so the
absolute level and spread of a series are only seen through its patch
statistics. S hands them to the classifier directly. The arrays from
src.data.load_dataset are resized to 512 but NOT normalised, so these
statistics keep the original level. Statistics of the difference are computed
on the resized series, so they are in "per resized step" units; the same
transformation is applied to train and test.

Leakage (lecture weeks 7 and 9): nothing here looks at the test set. S has no
fitted parameters. MiniRocket's biases are fitted on X_train. The scaler and the
ridge penalty are fitted inside one sklearn Pipeline on the training data, and
the ridge penalty is chosen by leave-one-out CV on the training data
(RidgeClassifierCV), as in aeon's MiniRocketClassifier.
"""
import inspect

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

STAT_NAMES = ["mean", "std", "min", "max", "slope",
              "diff_mean", "diff_std", "diff_min", "diff_max"]

# Same grid as aeon's MiniRocketClassifier, so "R + ridge" reproduces 04_minirocket.py.
RIDGE_ALPHAS = np.logspace(-3, 3, 10)


def series_stats(X: np.ndarray) -> np.ndarray:
    """X: (n, channels, length), not normalised. Returns (n, 9 * channels)."""
    X = np.asarray(X, dtype=np.float64)
    d = np.diff(X, axis=-1)
    t = np.arange(X.shape[-1], dtype=np.float64)
    t = (t - t.mean()) / t.std()
    slope = (X * t).mean(-1)  # least-squares slope against standardised time
    feats = [X.mean(-1), X.std(-1), X.min(-1), X.max(-1), slope,
             d.mean(-1), d.std(-1), d.min(-1), d.max(-1)]
    return np.concatenate(feats, axis=1).astype(np.float32)


def make_minirocket_transform(seed: int, n_jobs: int = -1):
    """aeon's MiniRocket transform (same defaults as MiniRocketClassifier)."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    params = inspect.signature(MiniRocket).parameters
    kwargs = {"random_state": seed}
    if "n_jobs" in params:
        kwargs["n_jobs"] = n_jobs
    return MiniRocket(**kwargs)


def block_weights(sizes: list[int]) -> np.ndarray:
    """Column weights that give every block the same total variance after
    standardisation, while keeping the total variance equal to the number of
    features. Without this, MiniRocket's ~10,000 columns would swamp Mantis's
    256 and the 9 statistics in the ridge penalty.

        w_b = sqrt( D / (B * d_b) )   D = total columns, B = blocks, d_b = block size

    With one block, w = 1, so single-block runs are exactly the standard
    StandardScaler + RidgeClassifierCV pipeline and the alpha grid keeps its meaning.
    Fixed before seeing any results (no tuning on test data)."""
    D, B = sum(sizes), len(sizes)
    return np.concatenate([np.full(d, np.sqrt(D / (B * d))) for d in sizes]).astype(np.float64)


def make_ridge(sizes: list[int], weighting: str = "equal"):
    """StandardScaler -> block weights -> RidgeClassifierCV, all fitted on train only."""
    steps = [StandardScaler()]
    if weighting == "equal" and len(sizes) > 1:
        w = block_weights(sizes)
        steps.append(FunctionTransformer(lambda Z, w=w: Z * w))
    steps.append(RidgeClassifierCV(alphas=RIDGE_ALPHAS))
    return make_pipeline(*steps)
