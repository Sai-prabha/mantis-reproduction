"""Calibration metrics and post-hoc methods, following v1 Section on calibration:
ECE with 10 equal-width bins; temperature scaling and multi-class isotonic
regression (IRM, Zhang et al. 2020), both fitted on the 20% validation split."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.sum() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total / len(y))


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Single temperature minimising validation NLL."""
    def loss(t):
        return nll(softmax(logits_val / t), y_val)
    return float(minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded").x)


class IRM:
    """Multi-class isotonic regression: one isotonic map shared by all classes,
    fitted on flattened (probability, one-hot label) pairs, then rows renormalised."""

    def fit(self, proba_val: np.ndarray, y_val: np.ndarray):
        onehot = np.eye(proba_val.shape[1])[y_val]
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(proba_val.ravel(), onehot.ravel())
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = self.iso.predict(proba.ravel()).reshape(proba.shape) + 1e-12
        return p / p.sum(axis=1, keepdims=True)


def calibration_report(logits_val, y_val, logits_test, y_test) -> dict:
    p_val, p_test = softmax(logits_val), softmax(logits_test)
    t = fit_temperature(logits_val, y_val)
    p_ts = softmax(logits_test / t)
    p_irm = IRM().fit(p_val, y_val).transform(p_test)
    return {
        "ece_raw": ece(p_test, y_test), "nll_raw": nll(p_test, y_test),
        "temperature": t,
        "ece_ts": ece(p_ts, y_test), "nll_ts": nll(p_ts, y_test),
        "ece_irm": ece(p_irm, y_test), "nll_irm": nll(p_irm, y_test),
    }
