
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """
    Loads, merges, cleans, and prepares the IEEE-CIS fraud dataset.

    Usage
    -----
    >>> prep = DataPreprocessor("config/config.yaml")
    >>> df = prep.run(save=True)
    """

    # Columns that are always categorical (regardless of dtype)
    CATEGORICAL_COLS = [
        "ProductCD",
        "card4", "card6",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
        "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
        "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
        "id_36", "id_37", "id_38",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.raw_dir = Path(cfg["data"]["raw_dir"])
        self.processed_dir = Path(cfg["data"]["processed_dir"])
        self.transaction_file = cfg["data"]["transaction_file"]
        self.identity_file = cfg["data"]["identity_file"]
        self.target_col = cfg["preprocessing"]["target_col"]
        self.id_col = cfg["preprocessing"]["id_col"]
        self.time_col = cfg["preprocessing"]["time_col"]
        self.amount_col = cfg["preprocessing"]["amount_col"]
        self.cat_fill = cfg["preprocessing"]["cat_fill_value"]
        self.num_fill = cfg["preprocessing"]["num_fill_strategy"]
        self.drop_cols = cfg["preprocessing"].get("drop_cols", [])

        self.label_encoders: dict[str, LabelEncoder] = {}
        self.num_medians: dict[str, float] = {}

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, save: bool = True) -> pd.DataFrame:
        """Full preprocessing pipeline. Returns cleaned DataFrame."""
        logger.info("Loading raw data …")
        df = self._load_and_merge()

        logger.info(f"Raw shape: {df.shape}")

        logger.info("Engineering time features …")
        df = self._engineer_time_features(df)

        logger.info("Handling missing values …")
        df = self._handle_missing(df)

        logger.info("Encoding categoricals …")
        df = self._encode_categoricals(df)

        logger.info("Dropping unused columns …")
        df = self._drop_columns(df)

        logger.info(f"Final shape: {df.shape}")

        if save:
            out_path = self.processed_dir / "processed_train.parquet"
            df.to_parquet(out_path, index=False)
            logger.success(f"Saved processed data → {out_path}")

            encoder_path = self.processed_dir / "label_encoders.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoders, f)
            logger.success(f"Saved encoders → {encoder_path}")

            median_path = self.processed_dir / "num_medians.pkl"
            with open(median_path, "wb") as f:
                pickle.dump(self.num_medians, f)

        return df

    def transform_single(self, record: dict) -> pd.DataFrame:
        """
        Transform a single inference record using fitted encoders/medians.
        Call after `run()` or after loading saved encoders.
        """
        df = pd.DataFrame([record])
        df = self._engineer_time_features(df)
        df = self._impute_with_saved_stats(df)
        df = self._apply_saved_encoders(df)
        return df

    def load_encoders(self):
        """Load encoders saved from a previous `run()`."""
        encoder_path = self.processed_dir / "label_encoders.pkl"
        with open(encoder_path, "rb") as f:
            self.label_encoders = pickle.load(f)

        median_path = self.processed_dir / "num_medians.pkl"
        with open(median_path, "rb") as f:
            self.num_medians = pickle.load(f)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_and_merge(self) -> pd.DataFrame:
        tx = pd.read_csv(self.raw_dir / self.transaction_file)
        id_ = pd.read_csv(self.raw_dir / self.identity_file)
        df = tx.merge(id_, on=self.id_col, how="left")
        logger.info(
            f"Transactions: {len(tx):,} | "
            f"Identity matched: {id_[self.id_col].isin(tx[self.id_col]).sum():,}"
        )
        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose TransactionDT (seconds from reference) into useful time units."""
        dt = df[self.time_col]
        df["tx_hour"] = (dt / 3600) % 24
        df["tx_day_of_week"] = (dt / 86400) % 7
        df["tx_day"] = (dt / 86400).astype(int)
        df["tx_week"] = (dt / (86400 * 7)).astype(int)

        # Cyclical encoding for hour and day (avoids discontinuity at midnight/end-of-week)
        df["tx_hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24)
        df["tx_hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24)
        df["tx_dow_sin"] = np.sin(2 * np.pi * df["tx_day_of_week"] / 7)
        df["tx_dow_cos"] = np.cos(2 * np.pi * df["tx_day_of_week"] / 7)

        # Log-transform amount (right-skewed)
        df["tx_amount_log"] = np.log1p(df[self.amount_col])

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs: median for numeric, constant for categorical."""
        cat_cols = self._get_categorical_cols(df)
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in [self.target_col, self.id_col] and c not in cat_cols
        ]

        # Numeric: compute median on training set, store for inference
        for col in num_cols:
            median = df[col].median()
            self.num_medians[col] = median
            df[col] = df[col].fillna(median)

        # Categorical: fill with "Unknown"
        for col in cat_cols:
            df[col] = df[col].fillna(self.cat_fill).astype(str)

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode all categorical columns."""
        cat_cols = self._get_categorical_cols(df)
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
        return df

    def _get_categorical_cols(self, df: pd.DataFrame) -> list[str]:
        """Return categorical columns that actually exist in df."""
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        explicit_cats = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        return list(set(object_cols + explicit_cats))

    def _impute_with_saved_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply saved medians for inference-time imputation."""
        for col, median in self.num_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        cat_cols = self._get_categorical_cols(df)
        for col in cat_cols:
            df[col] = df[col].fillna(self.cat_fill).astype(str)
        return df

    def _apply_saved_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply saved label encoders; unseen labels map to -1."""
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                continue
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
        return df


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    prep = DataPreprocessor(config)
    df = prep.run(save=True)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
