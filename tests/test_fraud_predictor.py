
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.fraud_predictor import FraudPrediction, FraudPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Minimal DataFrame that mimics the merged transaction/identity table."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "TransactionDT": np.random.randint(86400, 86400 * 30, n),
        "TransactionAmt": np.random.exponential(scale=100, size=n).clip(1, 5000),
        "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1": np.random.randint(1000, 5000, n),
        "card4": np.random.choice(["visa", "mastercard", None], n),
        "card6": np.random.choice(["debit", "credit", None], n),
        "addr1": np.random.randint(100, 500, n).astype(float),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "DeviceType": np.random.choice(["desktop", "mobile", None], n),
        "DeviceInfo": np.random.choice([f"Device_{i}" for i in range(1, 20)] + [None], n),
        "isFraud": np.random.choice([0, 1], n, p=[0.965, 0.035]),
        "C1": np.random.randint(1, 10, n).astype(float),
        "D1": np.random.randint(0, 365, n).astype(float),
        "V1": np.random.random(n),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_hour_extraction(self, sample_transactions):
        from src.preprocessing.clean_data import DataPreprocessor
        # Test the private helper directly
        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"

        df = prep._engineer_time_features(sample_transactions.copy())

        assert "tx_hour" in df.columns
        assert "tx_day_of_week" in df.columns
        assert "tx_hour_sin" in df.columns
        assert "tx_hour_cos" in df.columns
        assert df["tx_hour"].between(0, 24).all()
        assert df["tx_day_of_week"].between(0, 7).all()

    def test_log_amount(self, sample_transactions):
        from src.preprocessing.clean_data import DataPreprocessor
        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"
        df = prep._engineer_time_features(sample_transactions.copy())
        assert "tx_amount_log" in df.columns
        assert (df["tx_amount_log"] >= 0).all()

    def test_cyclical_encoding_bounds(self, sample_transactions):
        from src.preprocessing.clean_data import DataPreprocessor
        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"
        df = prep._engineer_time_features(sample_transactions.copy())
        # Sine and cosine should be in [-1, 1]
        assert df["tx_hour_sin"].between(-1.01, 1.01).all()
        assert df["tx_hour_cos"].between(-1.01, 1.01).all()


# ──────────────────────────────────────────────────────────────────────────────
# Behavioral Feature Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBehavioralFeatures:
    def test_card_stats_created(self, sample_transactions):
        from src.features.behavioral_features import BehavioralFeatureEngineer
        from src.preprocessing.clean_data import DataPreprocessor

        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"
        df = prep._engineer_time_features(sample_transactions.copy())

        eng = BehavioralFeatureEngineer()
        result = eng.fit_transform(df)

        assert "card_tx_count" in result.columns
        assert "card_tx_mean_amt" in result.columns
        assert "card_tx_std_amt" in result.columns
        assert "amount_zscore" in result.columns
        assert "card_amount_ratio" in result.columns

    def test_no_nulls_after_transform(self, sample_transactions):
        from src.features.behavioral_features import BehavioralFeatureEngineer
        from src.preprocessing.clean_data import DataPreprocessor

        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"
        df = prep._engineer_time_features(sample_transactions.copy())

        eng = BehavioralFeatureEngineer()
        result = eng.fit_transform(df)

        feature_cols = ["card_tx_count", "card_tx_mean_amt", "amount_zscore"]
        for col in feature_cols:
            assert result[col].notna().all(), f"NaNs found in {col}"

    def test_night_flag(self, sample_transactions):
        from src.features.behavioral_features import BehavioralFeatureEngineer
        from src.preprocessing.clean_data import DataPreprocessor

        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"
        df = prep._engineer_time_features(sample_transactions.copy())

        eng = BehavioralFeatureEngineer()
        result = eng.fit_transform(df)

        if "tx_is_night" in result.columns:
            assert result["tx_is_night"].isin([0, 1]).all()

    def test_weekend_flag(self, sample_transactions):
        from src.features.behavioral_features import BehavioralFeatureEngineer
        from src.preprocessing.clean_data import DataPreprocessor

        prep = DataPreprocessor.__new__(DataPreprocessor)
        prep.time_col = "TransactionDT"
        prep.amount_col = "TransactionAmt"
        df = prep._engineer_time_features(sample_transactions.copy())

        eng = BehavioralFeatureEngineer()
        result = eng.fit_transform(df)

        if "tx_is_weekend" in result.columns:
            assert result["tx_is_weekend"].isin([0, 1]).all()


# ──────────────────────────────────────────────────────────────────────────────
# Device Feature Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDeviceFeatures:
    def test_device_user_count_positive(self, sample_transactions):
        from src.features.device_features import DeviceFeatureEngineer
        eng = DeviceFeatureEngineer()
        result = eng.fit_transform(sample_transactions.copy())

        if "device_user_count" in result.columns:
            assert (result["device_user_count"] >= 0).all()

    def test_device_risk_score_bounds(self, sample_transactions):
        from src.features.device_features import DeviceFeatureEngineer
        eng = DeviceFeatureEngineer()
        result = eng.fit_transform(sample_transactions.copy())

        if "device_risk_score" in result.columns:
            assert result["device_risk_score"].between(0, 1).all(), \
                "device_risk_score should be in [0,1]"

    def test_device_fraud_rate_bounds(self, sample_transactions):
        from src.features.device_features import DeviceFeatureEngineer
        eng = DeviceFeatureEngineer()
        result = eng.fit_transform(sample_transactions.copy())

        if "device_fraud_rate" in result.columns:
            assert result["device_fraud_rate"].between(0, 1).all()

    def test_identity_mismatch_nonnegative(self, sample_transactions):
        from src.features.device_features import DeviceFeatureEngineer
        eng = DeviceFeatureEngineer()
        result = eng.fit_transform(sample_transactions.copy())

        if "identity_mismatch_score" in result.columns:
            assert (result["identity_mismatch_score"] >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# FraudPrediction Dataclass Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFraudPrediction:
    def _make_prediction(self, score: float, risk: str) -> FraudPrediction:
        return FraudPrediction(
            transaction_id="TX123456",
            fraud_score=score,
            risk_level=risk,
            lgbm_score=score,
            graph_score=score * 0.8,
            anomaly_score=score * 0.6,
            reasons=["test reason"],
            top_shap_features=[{"feature": "amount", "value": 100.0, "shap_impact": 0.05}],
            latency_ms=12.5,
        )

    def test_to_dict_keys(self):
        pred = self._make_prediction(0.85, "CRITICAL")
        d = pred.to_dict()
        assert "transaction_id" in d
        assert "fraud_score" in d
        assert "risk_level" in d
        assert "component_scores" in d
        assert "reasons" in d
        assert "top_shap_features" in d
        assert "latency_ms" in d

    def test_fraud_score_rounded(self):
        pred = self._make_prediction(0.123456789, "LOW")
        d = pred.to_dict()
        assert d["fraud_score"] == round(0.123456789, 4)

    def test_risk_level_preserved(self):
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            pred = self._make_prediction(0.5, level)
            assert pred.to_dict()["risk_level"] == level


# ──────────────────────────────────────────────────────────────────────────────
# Risk Classification Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskClassification:
    # Thresholds for calibrated-probability ensemble:
    #   LOW < 0.20  |  MEDIUM 0.20–0.40  |  HIGH 0.40–0.65  |  CRITICAL >= 0.65
    @pytest.mark.parametrize("score,expected", [
        (0.00, "LOW"),
        (0.10, "LOW"),
        (0.19, "LOW"),
        (0.20, "MEDIUM"),
        (0.30, "MEDIUM"),
        (0.39, "MEDIUM"),
        (0.40, "HIGH"),
        (0.52, "HIGH"),
        (0.64, "HIGH"),
        (0.65, "CRITICAL"),
        (0.80, "CRITICAL"),
        (1.00, "CRITICAL"),
    ])
    def test_classify_risk(self, score: float, expected: str):
        result = FraudPredictor._classify_risk(score)
        assert result == expected, f"score={score}: expected {expected}, got {result}"


# ──────────────────────────────────────────────────────────────────────────────
# Graph Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFraudGraph:
    def test_graph_builds_with_minimal_data(self, sample_transactions):
        from src.graph.build_graph import FraudGraphBuilder

        builder = FraudGraphBuilder.__new__(FraudGraphBuilder)
        builder.model_dir = __import__("pathlib").Path("/tmp")
        builder.graph_node_types = ["card1", "DeviceInfo", "P_emaildomain", "addr1"]
        builder.NODE_TYPES = {
            "card": "card1",
            "device": "DeviceInfo",
            "email": "P_emaildomain",
            "address": "addr1",
        }

        G = builder.build(sample_transactions)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() >= 0

    def test_graph_zero_features_for_unknown_node(self, sample_transactions):
        from src.graph.build_graph import FraudGraphBuilder
        zero = FraudGraphBuilder._zero_features()
        assert zero["graph_degree"] == 0
        assert zero["graph_pagerank"] == 0.0
        assert zero["graph_clustering"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly Model Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_fit_and_score(self, sample_transactions):
        from src.models.anomaly_model import AnomalyDetector

        detector = AnomalyDetector.__new__(AnomalyDetector)
        detector.contamination = 0.05
        detector.n_estimators = 50
        detector.max_samples = 100
        detector.random_state = 42
        detector.model = None

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
        detector.scaler = RobustScaler()
        detector._score_min = -1.0
        detector._score_max = 1.0

        X = sample_transactions[["TransactionAmt", "C1", "D1", "V1"]].fillna(0)
        detector.fit(X)

        scores = detector.score(X)
        assert len(scores) == len(X)
        assert all(0.0 <= s <= 1.0 for s in scores), "Scores should be in [0,1]"

    def test_select_anomaly_features_returns_subset(self, sample_transactions):
        from src.models.anomaly_model import AnomalyDetector

        # Add some expected columns
        df = sample_transactions.copy()
        df["amount_zscore"] = 0.0
        df["tx_hour"] = 12.0

        result = AnomalyDetector.select_anomaly_features(df)
        assert "TransactionAmt" in result.columns
        assert result.isna().sum().sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# API Schema Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAPISchemas:
    def test_transaction_request_valid(self):
        from src.api.app import TransactionRequest
        tx = TransactionRequest(
            TransactionAmt=500.0,
            card1=12345,
            DeviceType="desktop",
        )
        assert tx.TransactionAmt == 500.0

    def test_transaction_request_invalid_amount(self):
        from src.api.app import TransactionRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TransactionRequest(TransactionAmt=-1.0)

    def test_transaction_request_zero_amount(self):
        from src.api.app import TransactionRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TransactionRequest(TransactionAmt=0.0)

    def test_batch_request_max_size(self):
        from src.api.app import BatchRequest, TransactionRequest
        import pydantic
        txs = [TransactionRequest(TransactionAmt=float(i + 1)) for i in range(101)]
        with pytest.raises(pydantic.ValidationError):
            BatchRequest(transactions=txs)
