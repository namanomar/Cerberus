
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap
import yaml
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.behavioral_features import add_cross_entity_features
from src.models.calibrators import IsotonicCalibrator


class LightGBMTrainer:
    """
    Trains and evaluates a LightGBM binary classifier for fraud detection.

    Usage
    -----
    >>> trainer = LightGBMTrainer("config/config.yaml")
    >>> model, metrics = trainer.train()
    """

    NON_FEATURE_COLS = {
        "isFraud", "TransactionID", "TransactionDT",
        "tx_hour", "tx_day_of_week", "tx_day", "tx_week",
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model_cfg = self.cfg["model"]["lightgbm"]
        self.processed_dir = Path(self.cfg["data"]["processed_dir"])
        self.model_dir = Path(self.cfg["data"].get("models", "data/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.target = self.cfg["preprocessing"]["target_col"]
        self.id_col = self.cfg["preprocessing"]["id_col"]
        self.time_col = self.cfg["preprocessing"]["time_col"]
        self.val_frac = self.cfg["model"]["validation_frac"]
        self.early_stopping = self.cfg["model"]["early_stopping_rounds"]

        self.model: lgb.LGBMClassifier | None = None
        self.calibrated_model: IsotonicCalibrator | None = None
        self.feature_names: list[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> tuple[lgb.LGBMClassifier, dict]:
        """Full training pipeline. Returns (model, metrics_dict)."""
        df = self._load_data()
        X_train, X_val, y_train, y_val = self._split(df)

        logger.info(
            f"Train: {len(X_train):,} rows | "
            f"Val: {len(X_val):,} rows | "
            f"Fraud rate (train): {y_train.mean():.4%}"
        )

        # Compute scale_pos_weight to handle class imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(f"scale_pos_weight = {scale_pos_weight:.1f}")

        # Force AP as the only evaluation metric.
        # LightGBM's default 'binary_logloss' is appended to eval metrics and
        # is LAST in the list, so early_stopping monitors it.  Logloss rises
        # after round 1 when scale_pos_weight is high (the model trades
        # calibration for recall), causing instant early-stopping.
        # Setting metric="average_precision" in params prevents logloss from
        # appearing in the eval output entirely.
        params = {
            **self.model_cfg,
            "scale_pos_weight": scale_pos_weight,
            "metric": "average_precision",   # ← overrides default binary_logloss
        }
        self.model = lgb.LGBMClassifier(**params)

        # ── MLflow setup ──────────────────────────────────────────────────────
        tracking_uri = self.cfg["mlflow"]["tracking_uri"]
        # Only probe remote HTTP servers; sqlite:// and mlruns/ are always local
        if tracking_uri.startswith("http"):
            try:
                import urllib.request
                urllib.request.urlopen(tracking_uri, timeout=2)
            except Exception:
                logger.warning(
                    f"MLflow server at {tracking_uri} is not reachable. "
                    f"Falling back to local SQLite tracking."
                )
                tracking_uri = "sqlite:///mlruns/mlflow.db"

        # Ensure mlruns dir exists for SQLite backend
        import os
        os.makedirs("mlruns", exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        with mlflow.start_run(run_name="lightgbm-train"):
            mlflow.log_params(params)

            callbacks = [
                lgb.early_stopping(self.early_stopping, verbose=True),
                lgb.log_evaluation(50),
            ]

            # ── Train ──────────────────────────────────────────────────────────
            # metric="average_precision" is set in params above — do NOT pass
            # eval_metric here or LightGBM will append it a second time.
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )

            # ── Probability calibration (isotonic regression) ─────────────────
            # Maps raw LightGBM probabilities to better-calibrated posteriors.
            # Uses our own IsotonicCalibrator to avoid the deprecated
            # cv="prefit" string that was removed in recent scikit-learn versions.
            logger.info("Calibrating probability outputs (isotonic regression) …")
            self.calibrated_model = IsotonicCalibrator(self.model)
            self.calibrated_model.fit(X_val, y_val)
            logger.success("Calibration complete.")

            # Evaluate raw model first, then calibrated
            raw_metrics = self._evaluate(X_val, y_val, model=self.model)
            cal_metrics  = self._evaluate(X_val, y_val, model=self.calibrated_model)

            # Report both; log calibrated metrics to MLflow (they're better)
            metrics = cal_metrics
            mlflow.log_metrics({f"raw_{k}": v for k, v in raw_metrics.items()})
            mlflow.log_metrics({f"cal_{k}": v for k, v in cal_metrics.items()})
            mlflow.lightgbm.log_model(self.model.booster_, "lgbm_model")

            logger.info(f"Raw model metrics :  {raw_metrics}")
            logger.success(f"Calibrated metrics: {cal_metrics}")

            # SHAP analysis (uses raw model — TreeExplainer needs native booster)
            shap_values = self._compute_shap(X_val)
            self._log_shap_summary(shap_values, X_val)

            # Save artifacts
            self._save(metrics)

        return self.model, metrics

    def cross_validate(self, n_splits: int = 5) -> dict:
        """Stratified K-Fold cross-validation. Returns mean metrics."""
        df = self._load_data()
        X = df[self.feature_names]
        y = df[self.target]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"Fold {fold}/{n_splits}")
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
            model = lgb.LGBMClassifier(
                **{**self.model_cfg, "scale_pos_weight": neg / pos}
            )
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])

            proba = model.predict_proba(X_val)[:, 1]
            fold_metrics.append({
                "roc_auc": roc_auc_score(y_val, proba),
                "pr_auc": average_precision_score(y_val, proba),
            })

        mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics]))
                        for k in fold_metrics[0]}
        logger.success(f"CV Metrics: {mean_metrics}")
        return mean_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        path = self.processed_dir / "processed_train.parquet"
        logger.info(f"Loading processed data from {path} …")
        df = pd.read_parquet(path)

        # ── Cross-entity interaction features ─────────────────────────────
        # These 5 product/sum features (device×email, addr×device, total/max
        # entity risk, high-amt×risk) are added inline so the parquet doesn't
        # need to be regenerated.  IMPORTANT: the same call must exist in
        # FraudPredictor.predict() to keep training and inference consistent.
        n_before = len(df.columns)
        df = add_cross_entity_features(df)
        n_added = len(df.columns) - n_before
        logger.info(f"Added {n_added} cross-entity features.")

        self.feature_names = [
            c for c in df.columns
            if c not in self.NON_FEATURE_COLS
        ]
        return df

    def _split(self, df: pd.DataFrame):
        """
        Time-based split: last val_frac of transactions go to validation.
        This prevents leakage — real deployment always predicts future transactions.
        """
        df_sorted = df.sort_values(self.time_col)
        n_val = int(len(df_sorted) * self.val_frac)
        train = df_sorted.iloc[:-n_val]
        val = df_sorted.iloc[-n_val:]

        X_train = train[self.feature_names]
        X_val = val[self.feature_names]
        y_train = train[self.target]
        y_val = val[self.target]

        return X_train, X_val, y_train, y_val

    def _evaluate(self, X_val: pd.DataFrame, y_val: pd.Series, model=None) -> dict:
        """Evaluate a model (defaults to self.model).  Pass self.calibrated_model
        to compare calibrated vs raw probabilities."""
        m = model if model is not None else self.model
        proba = m.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)

        # Find optimal threshold via PR curve
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        opt_threshold = thresholds[np.argmax(f1_scores[:-1])]
        preds_opt = (proba >= opt_threshold).astype(int)

        return {
            "roc_auc": round(roc_auc_score(y_val, proba), 6),
            "pr_auc": round(average_precision_score(y_val, proba), 6),
            "f1_at_0.5": round(f1_score(y_val, preds), 6),
            "f1_optimal": round(f1_score(y_val, preds_opt), 6),
            "optimal_threshold": round(float(opt_threshold), 6),
            "fraud_recall_at_0.5": round(
                recall[np.searchsorted(thresholds, 0.5)], 6
            ) if any(thresholds >= 0.5) else 0.0,
        }

    def _compute_shap(self, X_val: pd.DataFrame) -> np.ndarray:
        logger.info("Computing SHAP values (sample of 5000 rows) …")
        sample = X_val.sample(min(5000, len(X_val)), random_state=42)
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(sample)

    def _log_shap_summary(self, shap_values, X_val: pd.DataFrame) -> None:
        """Log top 20 SHAP features to MLflow as a text artifact."""
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(20)

        shap_path = self.model_dir / "shap_importance.csv"
        importance_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(str(shap_path))
        logger.info(f"\nTop 10 SHAP features:\n{importance_df.head(10).to_string(index=False)}")

    def _save(self, metrics: dict) -> None:
        # Raw LightGBM (needed for SHAP and as fallback)
        model_path = self.model_dir / "lgbm_fraud_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Calibrated wrapper (used for actual scoring in inference)
        if self.calibrated_model is not None:
            cal_path = self.model_dir / "lgbm_calibrated_model.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump(self.calibrated_model, f)
            logger.success(f"Saved calibrated model → {cal_path}")

        feature_path = self.model_dir / "feature_names.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(self.feature_names, f)

        metrics_path = self.model_dir / "lgbm_metrics.yaml"
        with open(metrics_path, "w") as f:
            # Cast numpy scalars → Python floats so YAML is human-readable
            clean_metrics = {
                k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
            yaml.dump(clean_metrics, f)

        logger.success(f"Saved model → {model_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    trainer = LightGBMTrainer(config)
    model, metrics = trainer.train()
    print(f"\nFinal metrics: {metrics}")
