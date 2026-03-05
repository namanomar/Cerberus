
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import entropy as scipy_entropy


class DeviceFeatureEngineer:
    """
    Generates device & identity network features.

    Features produced
    ─────────────────
    device_user_count          # unique cards seen on this device
    device_tx_count            # total transactions from this device
    device_fraud_rate          # historical fraud rate for this device
    device_risk_score          # composite risk (user_count × fraud_rate)
    email_user_count           # unique cards using this email domain
    email_domain_fraud_rate    # historical fraud rate per email domain
    addr_user_count            # unique cards at this address
    addr_fraud_rate            # historical fraud rate per address
    card_device_count          # how many devices this card has used
    card_email_count           # how many email domains this card used
    identity_mismatch_score    # how much identity info differs from card history
    device_type_risk           # encoded device type risk level
    """

    DEVICE_COL = "DeviceInfo"
    DEVICE_TYPE_COL = "DeviceType"
    CARD_COL = "card1"
    EMAIL_COL = "P_emaildomain"
    ADDR_COL = "addr1"
    TARGET_COL = "isFraud"

    # Risk weight for device types (tunable based on domain knowledge)
    DEVICE_TYPE_RISK = {
        "mobile": 0.4,
        "desktop": 0.2,
        "Unknown": 0.6,
    }

    def __init__(self):
        self._device_stats: pd.DataFrame | None = None
        self._email_stats: pd.DataFrame | None = None
        self._addr_stats: pd.DataFrame | None = None
        self._card_stats: pd.DataFrame | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing device-level fraud features …")
        self._build_device_stats(df)
        self._build_email_stats(df)
        self._build_addr_stats(df)
        self._build_card_stats(df)
        return self._add_features(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._device_stats is None:
            raise RuntimeError("Call fit_transform first.")
        return self._add_features(df)

    # ──────────────────────────────────────────────────────────────────────────
    # Stat builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_device_stats(self, df: pd.DataFrame) -> None:
        dev = self.DEVICE_COL
        card = self.CARD_COL
        target = self.TARGET_COL

        agg = df.groupby(dev).agg(
            device_user_count=(card, "nunique"),
            device_tx_count=(card, "count"),
        )

        # Fraud rate per device (if label available)
        if target in df.columns:
            fraud_rate = df.groupby(dev)[target].mean().rename("device_fraud_rate")
            agg = agg.join(fraud_rate)
        else:
            agg["device_fraud_rate"] = 0.0

        agg["device_risk_score"] = (
            agg["device_user_count"] * agg["device_fraud_rate"]
        ).clip(upper=10) / 10  # Normalise to [0, 1]

        self._device_stats = agg.reset_index()

    def _build_email_stats(self, df: pd.DataFrame) -> None:
        email = self.EMAIL_COL
        card = self.CARD_COL
        target = self.TARGET_COL

        if email not in df.columns:
            return

        agg = df.groupby(email).agg(
            email_user_count=(card, "nunique"),
        )
        if target in df.columns:
            fraud_rate = df.groupby(email)[target].mean().rename("email_domain_fraud_rate")
            agg = agg.join(fraud_rate)
        else:
            agg["email_domain_fraud_rate"] = 0.0

        self._email_stats = agg.reset_index()

    def _build_addr_stats(self, df: pd.DataFrame) -> None:
        addr = self.ADDR_COL
        card = self.CARD_COL
        target = self.TARGET_COL

        if addr not in df.columns:
            return

        agg = df.groupby(addr).agg(
            addr_user_count=(card, "nunique"),
        )
        if target in df.columns:
            fraud_rate = df.groupby(addr)[target].mean().rename("addr_fraud_rate")
            agg = agg.join(fraud_rate)
        else:
            agg["addr_fraud_rate"] = 0.0

        self._addr_stats = agg.reset_index()

    def _build_card_stats(self, df: pd.DataFrame) -> None:
        card = self.CARD_COL
        dev = self.DEVICE_COL
        email = self.EMAIL_COL

        agg_parts = []

        if dev in df.columns:
            dev_count = df.groupby(card)[dev].nunique().rename("card_device_count")
            agg_parts.append(dev_count)

        if email in df.columns:
            email_count = df.groupby(card)[email].nunique().rename("card_email_count")
            agg_parts.append(email_count)

        if agg_parts:
            self._card_stats = pd.concat(agg_parts, axis=1).reset_index()

    # ──────────────────────────────────────────────────────────────────────────
    # Feature adder
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_merge(df: pd.DataFrame, stats: "pd.DataFrame", on: str) -> "pd.DataFrame":
        """
        Merge stats onto df, dropping any pre-existing conflicting columns first.
        This prevents pandas from creating _x/_y suffixes when fit_transform is
        called on a DataFrame that already contains the feature columns
        (e.g. a parquet saved after a previous training run).
        """
        stat_cols = [c for c in stats.columns if c != on]
        cols_to_drop = [c for c in stat_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df.merge(stats, on=on, how="left")

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Device stats ────────────────────────────────────────────────────
        if self._device_stats is not None and self.DEVICE_COL in df.columns:
            df = self._safe_merge(df, self._device_stats, on=self.DEVICE_COL)
            for col in ["device_user_count", "device_tx_count",
                        "device_fraud_rate", "device_risk_score"]:
                df[col] = df[col].fillna(0)

        # ── Email stats ─────────────────────────────────────────────────────
        if self._email_stats is not None and self.EMAIL_COL in df.columns:
            df = self._safe_merge(df, self._email_stats, on=self.EMAIL_COL)
            df["email_user_count"] = df["email_user_count"].fillna(1)
            df["email_domain_fraud_rate"] = df["email_domain_fraud_rate"].fillna(0)

        # ── Address stats ───────────────────────────────────────────────────
        if self._addr_stats is not None and self.ADDR_COL in df.columns:
            df = self._safe_merge(df, self._addr_stats, on=self.ADDR_COL)
            df["addr_user_count"] = df["addr_user_count"].fillna(1)
            df["addr_fraud_rate"] = df["addr_fraud_rate"].fillna(0)

        # ── Card stats ──────────────────────────────────────────────────────
        if self._card_stats is not None:
            df = self._safe_merge(df, self._card_stats, on=self.CARD_COL)
            for col in ["card_device_count", "card_email_count"]:
                if col in df.columns:
                    df[col] = df[col].fillna(1)

        # ── Device type risk ────────────────────────────────────────────────
        if self.DEVICE_TYPE_COL in df.columns:
            df["device_type_risk"] = (
                df[self.DEVICE_TYPE_COL]
                .astype(str)
                .str.lower()
                .map(self.DEVICE_TYPE_RISK)
                .fillna(0.5)
            )
        else:
            df["device_type_risk"] = 0.5

        # ── Identity mismatch score ──────────────────────────────────────────
        # Cards using many devices / email domains suggest credential stuffing
        if "card_device_count" in df.columns and "card_email_count" in df.columns:
            df["identity_mismatch_score"] = (
                np.log1p(df["card_device_count"]) * 0.5
                + np.log1p(df["card_email_count"]) * 0.5
            )
        else:
            df["identity_mismatch_score"] = 0.0

        # ── High-risk device flag ─────────────────────────────────────────
        if "device_user_count" in df.columns:
            p95 = df["device_user_count"].quantile(0.95)
            df["device_is_high_risk"] = (df["device_user_count"] > p95).astype(int)
        else:
            df["device_is_high_risk"] = 0

        return df
