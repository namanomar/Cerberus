
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anomaly_model import AnomalyDetector
from src.models.calibrators import IsotonicCalibrator  # noqa: F401 — required for pickle deserialization
from src.features.behavioral_features import add_cross_entity_features


def _to_python(obj):
    """
    Recursively convert numpy scalars / arrays to plain Python types so that
    yaml.dump() writes clean floats/ints instead of
    ``!!python/object/apply:numpy.core.multiarray.scalar`` tags.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_model(model_dir: Path):
    """Load calibrated model if available (mirrors FraudPredictor behaviour),
    otherwise fall back to the raw LightGBM model."""
    cal_path = model_dir / "lgbm_calibrated_model.pkl"
    if cal_path.exists():
        logger.info("Loading calibrated LightGBM model for evaluation.")
        with open(cal_path, "rb") as f:
            return pickle.load(f)
    with open(model_dir / "lgbm_fraud_model.pkl", "rb") as f:
        return pickle.load(f)


def load_feature_names(model_dir: Path) -> list[str]:
    with open(model_dir / "feature_names.pkl", "rb") as f:
        return pickle.load(f)


def evaluate_lgbm(
    model,
    feature_names: list[str],
    df: pd.DataFrame,
    target: str,
    threshold: float,
) -> dict:
    """Full LightGBM evaluation."""
    non_feat = {"isFraud", "TransactionID", "TransactionDT", "tx_day", "tx_week"}
    X = df[[c for c in feature_names if c in df.columns and c not in non_feat]].fillna(0)

    # Align missing features
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names].fillna(0)

    y = df[target]
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)

    # Optimal threshold
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1_arr = 2 * precision * recall / (precision + recall + 1e-9)
    opt_idx = np.argmax(f1_arr[:-1])
    opt_threshold = float(thresholds[opt_idx])
    opt_f1 = float(f1_arr[opt_idx])

    report = classification_report(y, preds, target_names=["Legitimate", "Fraud"])
    logger.info(f"\nClassification Report (threshold={threshold}):\n{report}")

    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": round(roc, 6),
        "pr_auc": round(pr_auc, 6),
        "f1_at_threshold": round(f1, 6),
        "threshold": threshold,
        "optimal_threshold": round(opt_threshold, 6),
        "optimal_f1": round(opt_f1, 6),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_recall": round(tp / (tp + fn + 1e-9), 6),
        "fraud_precision": round(tp / (tp + fp + 1e-9), 6),
    }


def evaluate_anomaly(
    model: AnomalyDetector,
    df: pd.DataFrame,
    target: str,
) -> dict:
    """Anomaly detector evaluation at multiple thresholds."""
    X = AnomalyDetector.select_anomaly_features(df)
    y = df[target]

    scores = model.score(X)
    results = {}

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        results[f"f1_at_{thresh}"] = round(f1, 6)

    try:
        results["roc_auc"] = round(roc_auc_score(y, scores), 6)
        results["pr_auc"] = round(average_precision_score(y, scores), 6)
    except Exception:
        pass

    return results


def run_evaluation(config_path: str, threshold: float = 0.5) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["data"].get("models", "data/models"))
    target = cfg["preprocessing"]["target_col"]
    time_col = cfg["preprocessing"]["time_col"]
    val_frac = cfg["model"]["validation_frac"]

    # Load processed data
    logger.info("Loading processed data …")
    df = pd.read_parquet(processed_dir / "processed_train.parquet")

    # Add cross-entity interaction features (matches what LightGBMTrainer._load_data does)
    df = add_cross_entity_features(df)

    # Use same time-based val split as training
    df_sorted = df.sort_values(time_col)
    n_val = int(len(df_sorted) * val_frac)
    val_df = df_sorted.iloc[-n_val:]
    logger.info(f"Validation set: {len(val_df):,} rows | fraud rate: {val_df[target].mean():.4%}")

    # Load models
    lgbm_model = load_model(model_dir)
    feature_names = load_feature_names(model_dir)
    anomaly_model = AnomalyDetector.load(model_dir)

    # Evaluate
    logger.info("\n── LightGBM Evaluation ──")
    lgbm_metrics = evaluate_lgbm(lgbm_model, feature_names, val_df, target, threshold)

    logger.info("\n── Anomaly Detector Evaluation ──")
    anomaly_metrics = evaluate_anomaly(anomaly_model, val_df, target)

    report = {
        "lightgbm": lgbm_metrics,
        "anomaly": anomaly_metrics,
        "val_set_size": len(val_df),
        "val_fraud_rate": round(val_df[target].mean(), 6),
    }

    # Save report — convert numpy scalars to plain Python before dumping
    report_path = processed_dir / "evaluation_report.yaml"
    with open(report_path, "w") as f:
        yaml.dump(_to_python(report), f, default_flow_style=False)

    logger.success(f"\nEvaluation report saved → {report_path}")
    logger.info(f"\nKey metrics:")
    logger.info(f"  LightGBM ROC-AUC  : {lgbm_metrics['roc_auc']:.4f}")
    logger.info(f"  LightGBM PR-AUC   : {lgbm_metrics['pr_auc']:.4f}")
    logger.info(f"  Fraud Recall      : {lgbm_metrics['fraud_recall']:.4f}")
    logger.info(f"  Optimal Threshold  : {lgbm_metrics['optimal_threshold']:.4f}")
    logger.info(f"  Anomaly ROC-AUC   : {anomaly_metrics.get('roc_auc', 'N/A')}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection Models")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for LightGBM")
    args = parser.parse_args()

    run_evaluation(args.config, threshold=args.threshold)
