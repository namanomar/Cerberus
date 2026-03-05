
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class BehavioralFeatureEngineer:
    """
    Generates behavioral (user-level) aggregate features from transaction history.

    The engineer operates in two modes:
    - fit_transform(df)  : builds aggregation stats from training data
    - transform(df)      : applies stats to new data (inference / streaming)

    Features generated per card (proxy for user identity):
    ─────────────────────────────────────────────────────────
    card_tx_count            total transactions
    card_tx_mean_amt         mean transaction amount
    card_tx_std_amt          std of transaction amounts
    card_tx_max_amt          max transaction amount
    card_freq_product_*      product code distribution
    card_merchant_entropy    entropy of merchant distribution
    amount_zscore            per-card z-score of current amount
    time_since_last_tx       seconds since card's previous transaction
    tx_is_night              flag: 1 if hour in [22, 6]
    tx_is_weekend            flag: 1 if day_of_week >= 5
    card_hour_mean           typical transaction hour for this card
    card_amount_ratio        current_amount / card mean amount
    """

    CARD_COL = "card1"
    AMOUNT_COL = "TransactionAmt"
    TIME_COL = "TransactionDT"
    PRODUCT_COL = "ProductCD"

    def __init__(self):
        self._card_stats: pd.DataFrame | None = None
        self._product_dummies: list[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute stats from df and add behavioral features. Returns augmented df."""
        logger.info("Computing card-level behavioral aggregates …")
        self._compute_card_stats(df)
        return self._add_features(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-computed stats to new data (inference mode)."""
        if self._card_stats is None:
            raise RuntimeError("Call fit_transform first or load saved stats.")
        return self._add_features(df)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_card_stats(self, df: pd.DataFrame) -> None:
        """Build per-card aggregate statistics from training data."""
        card = self.CARD_COL
        amt = self.AMOUNT_COL

        agg = df.groupby(card)[amt].agg(
            card_tx_count="count",
            card_tx_mean_amt="mean",
            card_tx_std_amt="std",
            card_tx_max_amt="max",
            card_tx_min_amt="min",
        ).reset_index()

        # Median transaction hour per card (typical activity window)
        if "tx_hour" in df.columns:
            hour_med = df.groupby(card)["tx_hour"].median().rename("card_hour_mean")
            agg = agg.merge(hour_med, on=card, how="left")

        # Product code distribution per card (fraud often concentrates on one product)
        if self.PRODUCT_COL in df.columns:
            product_counts = (
                df.groupby([card, self.PRODUCT_COL])
                .size()
                .unstack(fill_value=0)
                .add_prefix("card_freq_product_")
            )
            product_counts = product_counts.div(product_counts.sum(axis=1), axis=0)
            self._product_dummies = product_counts.columns.tolist()
            agg = agg.merge(product_counts.reset_index(), on=card, how="left")

        # Merchant entropy: high entropy = using many merchants (lower risk)
        if self.PRODUCT_COL in df.columns:
            def _entropy(series: pd.Series) -> float:
                probs = series.value_counts(normalize=True)
                return float(-np.sum(probs * np.log(probs + 1e-9)))

            merchant_ent = (
                df.groupby(card)[self.PRODUCT_COL]
                .apply(_entropy)
                .rename("card_merchant_entropy")
            )
            agg = agg.merge(merchant_ent.reset_index(), on=card, how="left")

        self._card_stats = agg

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        card = self.CARD_COL
        amt = self.AMOUNT_COL

        # ── 1. Merge pre-computed card-level stats ─────────────────────────
        # Drop any columns that already exist in df AND in _card_stats to
        # avoid pandas creating _x/_y suffixed duplicates (e.g. when the
        # processed parquet was saved after a previous fit_transform call).
        stat_cols = [c for c in self._card_stats.columns if c != card]
        cols_to_drop = [c for c in stat_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.merge(self._card_stats, on=card, how="left")

        # Fill missing (new cards not seen in training)
        for col in stat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # ── 2. Amount z-score (how unusual is this amount for this card?) ──
        std = df["card_tx_std_amt"].replace(0, 1).fillna(1)
        mean = df["card_tx_mean_amt"].fillna(df[amt].mean())
        df["amount_zscore"] = (df[amt] - mean) / std

        # ── 3. Amount ratio ─────────────────────────────────────────────────
        df["card_amount_ratio"] = df[amt] / (mean + 1e-9)

        # ── 4. Time since last transaction per card ─────────────────────────
        if self.TIME_COL in df.columns:
            df = df.sort_values(self.TIME_COL)
            if "time_since_last_tx" not in df.columns:
                df["time_since_last_tx"] = (
                    df.groupby(card)[self.TIME_COL]
                    .diff()
                    .fillna(0)
                )
            # Velocity: transactions in last 1h / 24h / 7d windows
            for window_secs, label in [(3600, "1h"), (86400, "24h"), (604800, "7d")]:
                col_name = f"card_tx_count_{label}"
                if col_name not in df.columns:
                    df[col_name] = self._rolling_count(
                        df, card, self.TIME_COL, window_secs
                    )

        # ── 5. Time-of-day flags ────────────────────────────────────────────
        if "tx_hour" in df.columns:
            df["tx_is_night"] = df["tx_hour"].between(22, 24) | df["tx_hour"].between(0, 6)
            df["tx_is_night"] = df["tx_is_night"].astype(int)

        if "tx_day_of_week" in df.columns:
            df["tx_is_weekend"] = (df["tx_day_of_week"] >= 5).astype(int)

        return df

    @staticmethod
    def _rolling_count(
        df: pd.DataFrame,
        group_col: str,
        time_col: str,
        window_secs: int,
    ) -> pd.Series:
        """
        For each row, count how many prior transactions by the same group entity
        occurred within the last `window_secs` seconds.
        Requires df to be sorted by time_col.
        """
        result = np.zeros(len(df), dtype=np.int32)
        times = df[time_col].values
        groups = df[group_col].values

        group_indices: dict = {}
        for i, (g, t) in enumerate(zip(groups, times)):
            if g not in group_indices:
                group_indices[g] = []
            # Count how many previous tx for this group fall within window
            prev = group_indices[g]
            cutoff = t - window_secs
            count = sum(1 for pt in prev if pt >= cutoff)
            result[i] = count
            prev.append(t)

        return pd.Series(result, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level utility — call from BOTH training and inference
# ──────────────────────────────────────────────────────────────────────────────

def add_cross_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-entity fraud-risk interaction features.

    These products / sums of entity fraud rates capture joint risk signals
    that the model cannot easily derive from individual rate columns alone.
    For example: a device WITH a high fraud rate AND an email domain WITH a
    high fraud rate is far more suspicious than either feature in isolation.

    Requires columns produced by DeviceFeatureEngineer:
        device_fraud_rate, email_domain_fraud_rate, addr_fraud_rate

    Requires columns produced by BehavioralFeatureEngineer:
        card_amount_ratio   (= TransactionAmt / card_tx_mean_amt)

    All operations are guarded — missing columns are silently skipped so the
    function is safe to call in inference even if a column isn't available.
    """
    dev  = "device_fraud_rate"
    mail = "email_domain_fraud_rate"
    addr = "addr_fraud_rate"
    ratio = "card_amount_ratio"

    # ── Joint risk products ────────────────────────────────────────────────
    # device × email: both entities historically associated with fraud
    if dev in df.columns and mail in df.columns:
        df["device_x_email_risk"] = df[dev] * df[mail]

    # address × device: billing address and device both high-risk
    if addr in df.columns and dev in df.columns:
        df["addr_x_device_risk"] = df[addr] * df[dev]

    # ── Aggregate entity risk ──────────────────────────────────────────────
    entity_cols = [c for c in [dev, mail, addr] if c in df.columns]
    if entity_cols:
        # Total risk: the more high-risk entities, the more suspicious
        df["total_entity_risk"] = df[entity_cols].sum(axis=1)
        # Max risk: single worst entity (catches fraudsters who rotate identities)
        df["max_entity_risk"] = df[entity_cols].max(axis=1)

    # ── Amount × entity risk ──────────────────────────────────────────────
    # Large transaction on a high-risk entity is especially suspicious
    if "max_entity_risk" in df.columns and ratio in df.columns:
        df["high_amt_x_entity_risk"] = df[ratio] * df["max_entity_risk"]

    return df
