
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(BASE_DIR / config_path) as f:
        return yaml.safe_load(f)


def load_transactions(cfg: dict, n_rows: int = 50_000) -> pd.DataFrame:
    """Load a sample of transactions to replay."""
    raw_dir = BASE_DIR / cfg["data"]["raw_dir"]
    tx_path = raw_dir / cfg["data"]["transaction_file"]
    id_path = raw_dir / cfg["data"]["identity_file"]

    logger.info(f"Loading {n_rows:,} transactions from dataset …")
    tx = pd.read_csv(tx_path, nrows=n_rows)
    id_ = pd.read_csv(id_path)
    df = tx.merge(id_, on="TransactionID", how="left")

    # Fill NaNs for JSON serialisation
    df = df.fillna("null")
    logger.success(f"Loaded {len(df):,} transactions")
    return df


def make_producer(bootstrap_servers: str) -> "KafkaProducer | None":
    if not KAFKA_AVAILABLE:
        logger.warning("kafka-python not installed. Running in dry-run mode.")
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            acks="all",
            retries=3,
            compression_type="gzip",
            batch_size=16_384,
            linger_ms=10,
        )
        logger.success(f"Connected to Kafka at {bootstrap_servers}")
        return producer
    except NoBrokersAvailable:
        logger.error(f"No Kafka brokers available at {bootstrap_servers}. Running dry-run.")
        return None


def row_to_message(row: pd.Series) -> dict:
    """Convert a DataFrame row to a JSON-serialisable message."""
    msg = {}
    for k, v in row.items():
        if v == "null" or (isinstance(v, float) and np.isnan(v)):
            msg[k] = None
        elif isinstance(v, (np.integer,)):
            msg[k] = int(v)
        elif isinstance(v, (np.floating,)):
            msg[k] = float(v)
        else:
            msg[k] = v
    msg["_produced_at"] = time.time()
    return msg


def run_producer(
    tps: int = 10,
    duration: int = 0,
    config_path: str = "config/config.yaml",
    n_rows: int = 50_000,
    inject_fraud_spike: bool = True,
) -> None:
    """
    Main producer loop.

    Parameters
    ----------
    tps       : transactions per second
    duration  : seconds to run (0 = run indefinitely)
    n_rows    : how many dataset rows to load for replay
    """
    cfg = load_config(config_path)
    topic = cfg["streaming"]["topic_transactions"]
    bootstrap = cfg["streaming"]["kafka_bootstrap_servers"]

    df = load_transactions(cfg, n_rows)
    producer = make_producer(bootstrap)

    delay = 1.0 / max(tps, 1)
    rows = df.to_dict("records")
    total_sent = 0
    start_time = time.time()
    last_stats_time = start_time
    stats_interval = 10  # seconds

    logger.info(f"Starting stream: {tps} TPS → topic '{topic}' | duration={duration}s")

    idx = 0
    while True:
        if duration > 0 and (time.time() - start_time) >= duration:
            logger.info(f"Duration {duration}s reached. Stopping.")
            break

        row = rows[idx % len(rows)]
        msg = row_to_message(pd.Series(row))

        # Optionally inject a synthetic fraud spike every ~500 transactions
        if inject_fraud_spike and total_sent % 500 == 0 and total_sent > 0:
            msg["TransactionAmt"] = random.uniform(3000, 10000)
            msg["_synthetic_spike"] = True
            logger.debug("Injected synthetic fraud spike")

        tx_id = msg.get("TransactionID", idx)

        if producer is not None:
            try:
                future = producer.send(
                    topic,
                    key=str(tx_id),
                    value=msg,
                )
                future.get(timeout=5)  # Block to confirm delivery
            except KafkaError as e:
                logger.error(f"Kafka send error: {e}")
        else:
            # Dry-run: just log
            logger.debug(f"[DRY-RUN] Would send TX {tx_id}: ${msg.get('TransactionAmt', 0):.2f}")

        total_sent += 1
        idx += 1

        # Periodic stats
        now = time.time()
        if now - last_stats_time >= stats_interval:
            elapsed = now - start_time
            actual_tps = total_sent / elapsed
            logger.info(
                f"Stats: sent={total_sent:,} | "
                f"elapsed={elapsed:.0f}s | "
                f"actual_tps={actual_tps:.1f}"
            )
            last_stats_time = now

        time.sleep(delay)

    if producer is not None:
        producer.flush()
        producer.close()

    logger.success(f"Producer finished. Total sent: {total_sent:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Kafka Producer")
    parser.add_argument("--tps", type=int, default=10, help="Transactions per second")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0=forever)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--rows", type=int, default=50_000, help="Dataset rows to load")
    parser.add_argument("--no-spike", action="store_true", help="Disable fraud spike injection")
    args = parser.parse_args()

    run_producer(
        tps=args.tps,
        duration=args.duration,
        config_path=args.config,
        n_rows=args.rows,
        inject_fraud_spike=not args.no_spike,
    )
