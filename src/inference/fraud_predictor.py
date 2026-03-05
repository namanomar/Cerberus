

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
import yaml
from loguru import logger

from src.preprocessing.clean_data import DataPreprocessor
from src.features.behavioral_features import BehavioralFeatureEngineer, add_cross_entity_features
from src.features.device_features import DeviceFeatureEngineer
from src.models.anomaly_model import AnomalyDetector
from src.graph.build_graph import FraudGraphBuilder
from src.graph.graph_embeddings import GraphEmbedder


@dataclass
class FraudPrediction:
    transaction_id: Any
    fraud_score: float
    risk_level: str
    lgbm_score: float
    graph_score: float
    anomaly_score: float
    reasons: list[str]
    top_shap_features: list[dict]
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "transaction_id": str(self.transaction_id),
            "fraud_score": round(self.fraud_score, 4),
            "risk_level": self.risk_level,
            "component_scores": {
                "lgbm": round(self.lgbm_score, 4),
                "graph": round(self.graph_score, 4),
                "anomaly": round(self.anomaly_score, 4),
            },
            "reasons": self.reasons,
            "top_shap_features": self.top_shap_features,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
        }


class FraudPredictor:
    """
    Real-time fraud scoring engine.

    Usage
    -----
    >>> predictor = FraudPredictor("config/config.yaml")
    >>> predictor.load_models()
    >>> result = predictor.predict(transaction_dict)
    >>> print(result.risk_level, result.fraud_score)
    """

    # Risk thresholds for the calibrated-probability ensemble.
    # Isotonic calibration pushes fraud scores into a proper 0–1 posterior range,
    # so the ensemble score is meaningful again (similar to the original design).
    RISK_LEVELS = [
        (0.65, "CRITICAL"),
        (0.40, "HIGH"),
        (0.20, "MEDIUM"),
        (0.00, "LOW"),
    ]

    NON_FEATURE_COLS = {
        "isFraud", "TransactionID", "TransactionDT",
        "tx_hour", "tx_day_of_week", "tx_day", "tx_week",
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.model_dir = Path(cfg["data"].get("models", "data/models"))
        self.processed_dir = Path(cfg["data"]["processed_dir"])

        ens = cfg["ensemble"]
        self.lgbm_weight = ens["lgbm_weight"]
        self.graph_weight = ens["graph_weight"]
        self.anomaly_weight = ens["anomaly_weight"]

        # Sub-components (loaded lazily)
        self.preprocessor: Optional[DataPreprocessor] = None
        self.behavioral_eng: Optional[BehavioralFeatureEngineer] = None
        self.device_eng: Optional[DeviceFeatureEngineer] = None
        self.lgbm_model = None          # scoring model (calibrated if available)
        self.lgbm_raw_model = None      # raw LightGBM — used for SHAP only
        self.feature_names: list[str] = []
        self.anomaly_model: Optional[AnomalyDetector] = None
        self.graph_builder: Optional[FraudGraphBuilder] = None
        self.graph_embedder: Optional[GraphEmbedder] = None
        self.graph: Any = None
        self.shap_explainer = None

        # Training data cache (for behavioural feature context)
        self._train_df: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load all serialised model artifacts from model_dir."""
        logger.info("Loading fraud detection models …")

        # Raw LightGBM (always loaded — required for SHAP)
        lgbm_path = self.model_dir / "lgbm_fraud_model.pkl"
        with open(lgbm_path, "rb") as f:
            self.lgbm_raw_model = pickle.load(f)

        # Prefer calibrated model for scoring (better probability outputs)
        cal_path = self.model_dir / "lgbm_calibrated_model.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                self.lgbm_model = pickle.load(f)
            logger.info("Loaded calibrated LightGBM model for scoring.")
        else:
            self.lgbm_model = self.lgbm_raw_model
            logger.info("No calibrated model found — using raw LightGBM.")

        feature_path = self.model_dir / "feature_names.pkl"
        with open(feature_path, "rb") as f:
            self.feature_names = pickle.load(f)

        # SHAP explainer must use the raw LightGBM (TreeExplainer needs the booster)
        self.shap_explainer = shap.TreeExplainer(self.lgbm_raw_model)

        # Anomaly model
        self.anomaly_model = AnomalyDetector.load(self.model_dir)

        # Graph
        try:
            self.graph = FraudGraphBuilder.load_graph(self.model_dir)
            self.graph_builder = FraudGraphBuilder(
                config_path="config/config.yaml"
            )
        except FileNotFoundError:
            logger.warning("No saved graph found. Graph score will be 0.")

        # Graph embeddings
        self.graph_embedder = GraphEmbedder(config_path="config/config.yaml")
        try:
            self.graph_embedder.load()
        except FileNotFoundError:
            logger.warning("No graph embeddings found.")

        # Preprocessor with saved encoders/medians
        self.preprocessor = DataPreprocessor(config_path="config/config.yaml")
        self.preprocessor.load_encoders()

        # Load cached training context for behavioral features
        train_path = self.processed_dir / "processed_train.parquet"
        if train_path.exists():
            self._train_df = pd.read_parquet(train_path)
            self.behavioral_eng = BehavioralFeatureEngineer()
            self.behavioral_eng.fit_transform(self._train_df)
            self.device_eng = DeviceFeatureEngineer()
            self.device_eng.fit_transform(self._train_df)

        logger.success("All models loaded.")

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, transaction: dict) -> FraudPrediction:
        """
        Score a single transaction.

        Parameters
        ----------
        transaction : dict  (raw transaction fields as received from API)

        Returns
        -------
        FraudPrediction
        """
        t0 = time.perf_counter()

        # ── 1. Preprocess ─────────────────────────────────────────────────────
        df = self.preprocessor.transform_single(transaction)

        # ── 2. Behavioral features ────────────────────────────────────────────
        if self.behavioral_eng is not None:
            df = self.behavioral_eng.transform(df)

        # ── 3. Device features ────────────────────────────────────────────────
        if self.device_eng is not None:
            df = self.device_eng.transform(df)

        # ── 3b. Cross-entity interaction features ─────────────────────────────
        # Must mirror the same call in LightGBMTrainer._load_data() so that
        # inference features match what the model was trained on.
        df = add_cross_entity_features(df)

        # ── 4. Graph features ─────────────────────────────────────────────────
        if self.graph is not None and self.graph_builder is not None:
            graph_feats = self.graph_builder.extract_node_features(df, self.graph)
            df = pd.concat([df.reset_index(drop=True),
                            graph_feats.reset_index(drop=True)], axis=1)

        # ── 5. Graph embeddings ───────────────────────────────────────────────
        if self.graph_embedder is not None and "card1" in df.columns:
            emb = self.graph_embedder.get_card_embeddings(df)
            df = pd.concat([df.reset_index(drop=True),
                            emb.reset_index(drop=True)], axis=1)

        # ── 6. LightGBM score ─────────────────────────────────────────────────
        X = self._align_features(df)
        lgbm_score = float(self.lgbm_model.predict_proba(X)[:, 1][0])

        # ── 7. Anomaly score ──────────────────────────────────────────────────
        anomaly_score = self._get_anomaly_score(df)

        # ── 8. Graph score (PageRank-based) ───────────────────────────────────
        graph_score = self._get_graph_score(df, transaction)

        # ── 9. Ensemble ───────────────────────────────────────────────────────
        final_score = (
            self.lgbm_weight * lgbm_score
            + self.graph_weight * graph_score
            + self.anomaly_weight * anomaly_score
        )
        final_score = float(np.clip(final_score, 0, 1))

        # ── 10. Explanations ──────────────────────────────────────────────────
        shap_features = self._explain(X)
        reasons = self._generate_reasons(
            transaction, lgbm_score, graph_score, anomaly_score, shap_features
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return FraudPrediction(
            transaction_id=transaction.get("TransactionID", "UNKNOWN"),
            fraud_score=final_score,
            risk_level=self._classify_risk(final_score),
            lgbm_score=lgbm_score,
            graph_score=graph_score,
            anomaly_score=anomaly_score,
            reasons=reasons,
            top_shap_features=shap_features,
            latency_ms=latency_ms,
        )

    def predict_batch(self, transactions: list[dict]) -> list[FraudPrediction]:
        """Score a batch of transactions (for Kafka consumer / batch inference)."""
        return [self.predict(tx) for tx in transactions]

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align df columns to the model's expected feature set."""
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names].fillna(0)

    def _get_anomaly_score(self, df: pd.DataFrame) -> float:
        if self.anomaly_model is None:
            return 0.0
        X_anom = AnomalyDetector.select_anomaly_features(df)
        try:
            return float(self.anomaly_model.score(X_anom)[0])
        except Exception:
            return 0.0

    def _get_graph_score(self, df: pd.DataFrame, transaction: dict) -> float:
        """
        Use graph PageRank of the card node as graph fraud signal.
        High PageRank on a fraud-dense graph → higher risk.
        """
        if self.graph is None:
            return 0.0

        card_val = str(transaction.get("card1", ""))
        node_id = f"card:{card_val}"

        if node_id not in self.graph:
            return 0.0

        # Normalise PageRank (values are tiny; rescale to [0,1])
        pr_values = list(dict(self.graph.degree()).values())
        node_degree = self.graph.degree(node_id)

        if not pr_values or max(pr_values) == 0:
            return 0.0

        # Use normalised degree as a simple proxy for PageRank
        return min(float(node_degree) / max(pr_values), 1.0)

    def _explain(self, X: pd.DataFrame) -> list[dict]:
        """Return top 5 SHAP features for this transaction."""
        if self.shap_explainer is None:
            return []
        try:
            shap_vals = self.shap_explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_row = shap_vals[0]
            top_indices = np.argsort(np.abs(shap_row))[::-1][:5]
            return [
                {
                    "feature": self.feature_names[i],
                    "value": float(X.iloc[0, i]),
                    "shap_impact": round(float(shap_row[i]), 4),
                }
                for i in top_indices
            ]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return []

    def _generate_reasons(
        self,
        transaction: dict,
        lgbm_score: float,
        graph_score: float,
        anomaly_score: float,
        shap_features: list[dict],
    ) -> list[str]:
        reasons = []

        # After isotonic calibration, scores above 0.40 represent strong fraud signal
        if lgbm_score > 0.40:
            reasons.append(f"High ML fraud probability ({lgbm_score:.0%})")

        if graph_score > 0.5:
            reasons.append(
                "Card appears in a high-risk device-sharing network "
                f"(graph risk: {graph_score:.0%})"
            )

        if anomaly_score > 0.7:
            reasons.append(
                f"Transaction pattern is statistically anomalous "
                f"(anomaly score: {anomaly_score:.0%})"
            )

        amount = transaction.get("TransactionAmt", 0)
        if amount > 5000:
            reasons.append(f"Unusually large transaction amount (${amount:,.0f})")

        for feat in shap_features[:2]:
            if abs(feat["shap_impact"]) > 0.05:
                direction = "high" if feat["shap_impact"] > 0 else "low"
                reasons.append(
                    f"Suspicious {feat['feature']} = {feat['value']:.2f} "
                    f"({direction} fraud impact)"
                )

        if not reasons:
            reasons.append("No specific high-risk indicators detected")

        return reasons

    @staticmethod
    def _classify_risk(score: float) -> str:
        """
        Map ensemble score → risk level.

        Thresholds for the calibrated-probability ensemble:
          CRITICAL  ≥ 0.65  (strong multi-component fraud signal)
          HIGH      ≥ 0.40  (moderate multi-component signal)
          MEDIUM    ≥ 0.20  (some fraud signal detected)
          LOW        < 0.20  (no significant fraud signal)
        """
        if score >= 0.65:
            return "CRITICAL"
        elif score >= 0.40:
            return "HIGH"
        elif score >= 0.20:
            return "MEDIUM"
        else:
            return "LOW"
