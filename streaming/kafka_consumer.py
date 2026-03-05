

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml
from loguru import logger

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(BASE_DIR / config_path) as f:
        return yaml.safe_load(f)


class FraudAlertConsumer:
    """
    Kafka consumer that:
      1. Reads transaction messages
      2. Sends them to the fraud detection API
      3. Forwards HIGH/CRITICAL results to the alerts topic
      4. Writes alert log for offline analysis
    """

    ALERT_LOG_PATH = BASE_DIR / "data" / "processed" / "alert_log.jsonl"

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = load_config(config_path)
        self.bootstrap = cfg["streaming"]["kafka_bootstrap_servers"]
        self.tx_topic = cfg["streaming"]["topic_transactions"]
        self.alert_topic = cfg["streaming"]["topic_alerts"]
        self.consumer_group = cfg["streaming"]["consumer_group"]
        self.api_url = os.getenv("FRAUD_API_URL", "http://localhost:8000")

        self.consumer = None
        self.producer = None
        self.http_client = httpx.Client(timeout=10.0)

        self.stats = {
            "processed": 0,
            "high_risk": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    def connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.warning("kafka-python not available. Using dry-run mode.")
            return False

        try:
            self.consumer = KafkaConsumer(
                self.tx_topic,
                bootstrap_servers=self.bootstrap,
                group_id=self.consumer_group,
                auto_offset_reset="latest",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                max_poll_records=50,
                session_timeout_ms=30_000,
            )
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.success(f"Connected to Kafka at {self.bootstrap}")
            return True
        except NoBrokersAvailable:
            logger.error("No Kafka brokers available.")
            return False

    def score_transaction(self, tx: dict) -> dict | None:
        """Call the fraud API for a single transaction."""
        try:
            response = self.http_client.post(
                f"{self.api_url}/predict",
                json=tx,
            )
            if response.status_code == 200:
                return response.json()
            logger.warning(f"API returned {response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"API request failed: {e}")
        return None

    def handle_alert(self, prediction: dict, original_tx: dict) -> None:
        """Process a HIGH/CRITICAL fraud alert."""
        risk = prediction.get("risk_level", "UNKNOWN")
        score = prediction.get("fraud_score", 0)
        tx_id = prediction.get("transaction_id", "?")

        logger.warning(
            f"🚨 FRAUD ALERT | TX: {tx_id} | Risk: {risk} | Score: {score:.4f} | "
            f"Reasons: {prediction.get('reasons', [])}"
        )

        # Publish to alerts topic
        if self.producer is not None:
            alert_msg = {
                "alert_type": "fraud_detection",
                "transaction_id": tx_id,
                "fraud_score": score,
                "risk_level": risk,
                "reasons": prediction.get("reasons", []),
                "component_scores": prediction.get("component_scores", {}),
                "original_amount": original_tx.get("TransactionAmt"),
                "alert_timestamp": time.time(),
            }
            self.producer.send(self.alert_topic, value=alert_msg)

        # Write to alert log
        self._write_alert_log({**prediction, "original_tx": original_tx})
        self.stats["high_risk"] += 1

    def _write_alert_log(self, record: dict) -> None:
        self.ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ALERT_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _print_stats(self) -> None:
        elapsed = time.time() - self.stats["start_time"]
        tps = self.stats["processed"] / max(elapsed, 1)
        logger.info(
            f"Stats: processed={self.stats['processed']:,} | "
            f"high_risk={self.stats['high_risk']:,} | "
            f"errors={self.stats['errors']:,} | "
            f"TPS={tps:.1f} | "
            f"elapsed={elapsed:.0f}s"
        )

    def run(self, batch_size: int = 10) -> None:
        """Main consumer loop."""
        kafka_available = self.connect()
        last_stats = time.time()

        if not kafka_available:
            logger.info("Running in simulation mode (no Kafka). Press Ctrl+C to stop.")
            self._run_simulation()
            return

        logger.info(
            f"Consuming from topic '{self.tx_topic}' | "
            f"group='{self.consumer_group}' | "
            f"API='{self.api_url}'"
        )

        try:
            for message in self.consumer:
                tx = message.value

                prediction = self.score_transaction(tx)
                self.stats["processed"] += 1

                if prediction is None:
                    self.stats["errors"] += 1
                    continue

                risk = prediction.get("risk_level", "LOW")
                if risk in ("HIGH", "CRITICAL"):
                    self.handle_alert(prediction, tx)

                # Periodic stats
                if time.time() - last_stats >= 10:
                    self._print_stats()
                    last_stats = time.time()

        except KeyboardInterrupt:
            logger.info("Consumer stopped by user.")
        finally:
            self._print_stats()
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.flush()
                self.producer.close()
            self.http_client.close()

    def _run_simulation(self) -> None:
        """
        Simulation mode: generate fake transactions and score via API.
        Useful for testing without a running Kafka cluster.
        """
        import random

        logger.info("Simulation mode: generating synthetic transactions")
        try:
            while True:
                fake_tx = {
                    "TransactionID": random.randint(1_000_000, 9_999_999),
                    "TransactionDT": 86400.0 + random.randint(0, 3_600_000),
                    "TransactionAmt": random.choice(
                        [random.uniform(5, 500)] * 19 + [random.uniform(2000, 8000)]
                    ),
                    "ProductCD": random.choice(["W", "H", "C", "S", "R"]),
                    "card1": random.randint(1000, 9999),
                    "DeviceType": random.choice(["desktop", "mobile", None]),
                    "DeviceInfo": random.choice([f"Device_{i}" for i in range(1, 50)]),
                    "P_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com"]),
                }

                prediction = self.score_transaction(fake_tx)
                self.stats["processed"] += 1

                if prediction:
                    risk = prediction.get("risk_level", "LOW")
                    score = prediction.get("fraud_score", 0)
                    logger.info(
                        f"TX {fake_tx['TransactionID']} | "
                        f"${fake_tx['TransactionAmt']:.2f} | "
                        f"Risk: {risk} | Score: {score:.4f}"
                    )
                    if risk in ("HIGH", "CRITICAL"):
                        self.handle_alert(prediction, fake_tx)
                else:
                    logger.warning("API unavailable — skipping")

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Simulation stopped.")
            self._print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Kafka Consumer")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    consumer = FraudAlertConsumer(config_path=args.config)
    consumer.run(batch_size=args.batch_size)
