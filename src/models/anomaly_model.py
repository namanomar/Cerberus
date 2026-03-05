

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class AnomalyDetector:
    """
    Wraps Isolation Forest with:
      - RobustScaler preprocessing (resistant to outliers)
      - Score normalisation to [0, 1]
      - Serialisation / loading

    Usage
    -----
    >>> detector = AnomalyDetector("config/config.yaml")
    >>> detector.fit(X_train)
    >>> scores = detector.score(X_new)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        anomaly_cfg = cfg["anomaly"]
        self.contamination = anomaly_cfg["contamination"]
        self.n_estimators = anomaly_cfg["n_estimators"]
        self.max_samples = anomaly_cfg["max_samples"]
        self.random_state = anomaly_cfg["random_state"]

        self.model_dir = Path(cfg["data"].get("models", "data/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: IsolationForest | None = None
        self.scaler = RobustScaler()
        self._score_min: float = -1.0
        self._score_max: float = 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame | np.ndarray) -> "AnomalyDetector":
        """Fit the Isolation Forest on training data (exclude the target column)."""
        X_scaled = self.scaler.fit_transform(X)

        logger.info(
            f"Fitting Isolation Forest: "
            f"n_estimators={self.n_estimators}, "
            f"contamination={self.contamination}, "
            f"n_samples={len(X_scaled):,}"
        )

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        # Calibrate score range on training data
        raw_scores = self.model.decision_function(X_scaled)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())

        logger.success("Isolation Forest fitted.")
        return self

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in [0, 1].
        Higher = more anomalous / fraudulent.
        """
        if self.model is None:
            raise RuntimeError("Call fit() first or load a saved model.")

        X_scaled = self.scaler.transform(X)
        raw = self.model.decision_function(X_scaled)

        # Invert: decision_function returns negative for outliers
        # Normalise to [0, 1]: 0 = normal, 1 = most anomalous
        scores = 1 - self._minmax_scale(raw)
        return scores

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return -1 (anomaly) or 1 (normal) — raw Isolation Forest output."""
        if self.model is None:
            raise RuntimeError("Call fit() first or load a saved model.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self) -> None:
        model_path = self.model_dir / "anomaly_model.pkl"
        scaler_path = self.model_dir / "anomaly_scaler.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self, f)

        logger.success(f"Saved anomaly model → {model_path}")

    @classmethod
    def load(cls, model_dir: str | Path) -> "AnomalyDetector":
        model_path = Path(model_dir) / "anomaly_model.pkl"
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _minmax_scale(self, arr: np.ndarray) -> np.ndarray:
        """Normalise array to [0, 1] using training set min/max."""
        denom = self._score_max - self._score_min
        if denom == 0:
            return np.zeros_like(arr)
        return (arr - self._score_min) / denom

    # ──────────────────────────────────────────────────────────────────────────
    # Feature selection for anomaly detection
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def select_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Select a compact, high-signal feature subset for anomaly detection.
        Isolation Forest works best with fewer, meaningful features.
        """
        priority_features = [
            "TransactionAmt", "tx_amount_log",
            "amount_zscore", "card_amount_ratio",
            "time_since_last_tx",
            "tx_hour", "tx_is_night", "tx_is_weekend",
            "device_user_count", "device_risk_score", "device_fraud_rate",
            "card_tx_count", "card_tx_mean_amt", "card_tx_std_amt",
            "card_device_count", "card_email_count",
            "identity_mismatch_score",
            "email_domain_fraud_rate", "addr_fraud_rate",
            "card_tx_count_24h", "card_tx_count_1h",
        ]

        available = [c for c in priority_features if c in df.columns]
        return df[available].fillna(0)
