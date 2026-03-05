

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """
    Post-hoc isotonic regression calibrator for a pre-fitted classifier.

    Wraps any model that exposes ``predict_proba`` and remaps its raw
    probability outputs to better-calibrated posteriors using isotonic
    regression — a non-parametric monotone mapping that outperforms
    Platt (sigmoid) scaling on large, imbalanced datasets.

    Works with all scikit-learn versions (avoids the deprecated
    ``cv="prefit"`` string in ``CalibratedClassifierCV``).

    The ``estimator`` property exposes the underlying model so that
    SHAP's TreeExplainer can still access the native booster.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self._ir = IsotonicRegression(out_of_bounds="clip")

    @property
    def estimator(self):
        """Return the underlying (uncalibrated) model, e.g. for SHAP."""
        return self.base_model

    def fit(self, X, y) -> "IsotonicCalibrator":
        """Fit isotonic regression on the validation set predictions."""
        raw = self.base_model.predict_proba(X)[:, 1]
        self._ir.fit(raw, np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated probabilities as a (n, 2) array."""
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self._ir.predict(raw)
        return np.column_stack([1.0 - cal, cal])
