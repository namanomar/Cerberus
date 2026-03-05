

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.clean_data import DataPreprocessor
from src.features.behavioral_features import BehavioralFeatureEngineer
from src.features.device_features import DeviceFeatureEngineer
from src.models.train_lightgbm import LightGBMTrainer
from src.models.anomaly_model import AnomalyDetector
from src.graph.build_graph import FraudGraphBuilder
from src.graph.graph_embeddings import GraphEmbedder


def run_pipeline(config_path: str, skip_graph: bool = False) -> None:
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION TRAINING PIPELINE")
    logger.info("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["data"].get("models", "data/models"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    logger.info("\n[1/6] Data Preprocessing …")
    t0 = time.time()
    preprocessor = DataPreprocessor(config_path)
    df = preprocessor.run(save=True)
    logger.success(f"  Done in {time.time() - t0:.1f}s — shape: {df.shape}")

    # ── Step 2: Behavioral Features ───────────────────────────────────────────
    logger.info("\n[2/6] Behavioral Feature Engineering …")
    t0 = time.time()
    beh_eng = BehavioralFeatureEngineer()
    df = beh_eng.fit_transform(df)
    logger.success(f"  Done in {time.time() - t0:.1f}s — shape: {df.shape}")

    # ── Step 3: Device Features ───────────────────────────────────────────────
    logger.info("\n[3/6] Device Feature Engineering …")
    t0 = time.time()
    dev_eng = DeviceFeatureEngineer()
    df = dev_eng.fit_transform(df)
    logger.success(f"  Done in {time.time() - t0:.1f}s — shape: {df.shape}")

    # Save augmented dataset
    augmented_path = processed_dir / "processed_train.parquet"
    df.to_parquet(augmented_path, index=False)
    logger.info(f"  Saved augmented dataset → {augmented_path}")

    # ── Step 4: Graph + Embeddings ────────────────────────────────────────────
    if not skip_graph:
        logger.info("\n[4/6] Graph Construction + Node2Vec …")
        t0 = time.time()

        # Sample for graph (full dataset can be too large for Node2Vec)
        graph_sample = df.sample(min(100_000, len(df)), random_state=42)

        builder = FraudGraphBuilder(config_path)
        G = builder.build(graph_sample)
        builder.save_graph(G)

        graph_feats = builder.extract_node_features(graph_sample, G)
        graph_feats.to_parquet(processed_dir / "graph_features.parquet")

        embedder = GraphEmbedder(config_path)
        embedder.fit(G)
        embedder.save()

        logger.success(f"  Done in {time.time() - t0:.1f}s")
    else:
        logger.info("\n[4/6] Skipping graph construction (--skip-graph flag set)")

    # ── Step 5: LightGBM Training ─────────────────────────────────────────────
    logger.info("\n[5/6] LightGBM Training …")
    t0 = time.time()
    trainer = LightGBMTrainer(config_path)
    model, lgbm_metrics = trainer.train()
    logger.success(
        f"  Done in {time.time() - t0:.1f}s | "
        f"ROC-AUC={lgbm_metrics['roc_auc']:.4f} | "
        f"PR-AUC={lgbm_metrics['pr_auc']:.4f}"
    )

    # ── Step 6: Anomaly Detector ──────────────────────────────────────────────
    logger.info("\n[6/6] Anomaly Detector Training …")
    t0 = time.time()
    anomaly = AnomalyDetector(config_path)
    X_anom = AnomalyDetector.select_anomaly_features(df)
    anomaly.fit(X_anom)
    anomaly.save()
    logger.success(f"  Done in {time.time() - t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.success(f"PIPELINE COMPLETE in {total_time:.1f}s")
    logger.info(f"  LightGBM ROC-AUC  : {lgbm_metrics['roc_auc']:.4f}")
    logger.info(f"  LightGBM PR-AUC   : {lgbm_metrics['pr_auc']:.4f}")
    logger.info(f"  Optimal threshold  : {lgbm_metrics['optimal_threshold']:.4f}")
    logger.info(f"  Models saved to   : {model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Training Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip graph construction (faster iteration)")
    args = parser.parse_args()

    run_pipeline(args.config, skip_graph=args.skip_graph)
