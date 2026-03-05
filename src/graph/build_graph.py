

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from loguru import logger


class FraudGraphBuilder:
    """
    Builds and analyses the fraud entity graph from transaction data.

    Usage
    -----
    >>> builder = FraudGraphBuilder("config/config.yaml")
    >>> G = builder.build(df)
    >>> graph_features = builder.extract_node_features(df, G)
    """

    NODE_TYPES = {
        "card": "card1",
        "device": "DeviceInfo",
        "email": "P_emaildomain",
        "address": "addr1",
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model_dir = Path(cfg["data"].get("models", "data/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.graph_node_types = cfg["graph"]["node_types"]

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build an undirected entity graph from transaction data.
        Returns a NetworkX Graph with typed node attributes.
        """
        G = nx.Graph()

        logger.info("Building fraud entity graph …")

        card_col = self.NODE_TYPES["card"]

        for entity_type, col in self.NODE_TYPES.items():
            if col not in df.columns:
                continue

            values = df[col].dropna().astype(str).unique()
            for val in values:
                node_id = f"{entity_type}:{val}"
                G.add_node(node_id, type=entity_type, value=val)

            logger.info(f"  Added {len(values):,} {entity_type} nodes")

        # ── Edges: card ↔ device ─────────────────────────────────────────────
        if "DeviceInfo" in df.columns:
            pairs = (
                df[[card_col, "DeviceInfo"]]
                .dropna()
                .astype(str)
                .drop_duplicates()
            )
            for _, row in pairs.iterrows():
                G.add_edge(
                    f"card:{row[card_col]}",
                    f"device:{row['DeviceInfo']}",
                    edge_type="card_device",
                )
            logger.info(f"  Added {len(pairs):,} card-device edges")

        # ── Edges: card ↔ email ──────────────────────────────────────────────
        if "P_emaildomain" in df.columns:
            pairs = (
                df[[card_col, "P_emaildomain"]]
                .dropna()
                .astype(str)
                .drop_duplicates()
            )
            for _, row in pairs.iterrows():
                G.add_edge(
                    f"card:{row[card_col]}",
                    f"email:{row['P_emaildomain']}",
                    edge_type="card_email",
                )
            logger.info(f"  Added {len(pairs):,} card-email edges")

        # ── Edges: card ↔ address ────────────────────────────────────────────
        if "addr1" in df.columns:
            pairs = (
                df[[card_col, "addr1"]]
                .dropna()
                .astype(str)
                .drop_duplicates()
            )
            for _, row in pairs.iterrows():
                G.add_edge(
                    f"card:{row[card_col]}",
                    f"address:{row['addr1']}",
                    edge_type="card_address",
                )
            logger.info(f"  Added {len(pairs):,} card-address edges")

        # ── Co-device card edges (card1 ↔ card2 sharing same device) ─────────
        if "DeviceInfo" in df.columns:
            device_cards = (
                df[[card_col, "DeviceInfo"]]
                .dropna()
                .astype(str)
                .groupby("DeviceInfo")[card_col]
                .apply(list)
            )
            co_edges = 0
            for device, cards in device_cards.items():
                unique_cards = list(set(cards))
                if len(unique_cards) > 1:
                    for i in range(len(unique_cards)):
                        for j in range(i + 1, min(i + 5, len(unique_cards))):
                            G.add_edge(
                                f"card:{unique_cards[i]}",
                                f"card:{unique_cards[j]}",
                                edge_type="co_device",
                                shared_device=device,
                            )
                            co_edges += 1
            logger.info(f"  Added {co_edges:,} co-device card-card edges")

        logger.success(
            f"Graph built: {G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges"
        )
        return G

    def extract_node_features(
        self, df: pd.DataFrame, G: nx.Graph
    ) -> pd.DataFrame:
        """
        Compute graph-based features for each transaction's card node:
          - degree
          - pagerank
          - clustering_coefficient
          - connected_component_size
          - n_neighbors_of_type (device / email / address)
          - co_device_card_count (number of cards sharing the same device)

        Returns a DataFrame indexed by TransactionID.
        """
        logger.info("Extracting graph node features …")

        card_col = self.NODE_TYPES["card"]

        # Pre-compute graph metrics (expensive, do once)
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        clustering = nx.clustering(G)
        components = {
            node: len(c)
            for c in nx.connected_components(G)
            for node in c
        }

        records = []
        for _, row in df.iterrows():
            card_id = f"card:{row[card_col]}"

            if card_id not in G:
                records.append(self._zero_features())
                continue

            degree = G.degree(card_id)
            pr = pagerank.get(card_id, 0.0)
            cc = clustering.get(card_id, 0.0)
            comp_size = components.get(card_id, 1)

            neighbors = list(G.neighbors(card_id))
            n_device_neighbors = sum(1 for n in neighbors if n.startswith("device:"))
            n_email_neighbors = sum(1 for n in neighbors if n.startswith("email:"))
            n_card_neighbors = sum(1 for n in neighbors if n.startswith("card:"))

            records.append({
                "graph_degree": degree,
                "graph_pagerank": pr,
                "graph_clustering": cc,
                "graph_component_size": comp_size,
                "graph_n_device_neighbors": n_device_neighbors,
                "graph_n_email_neighbors": n_email_neighbors,
                "graph_n_co_device_cards": n_card_neighbors,
                "graph_is_hub": int(degree > np.percentile(
                    [G.degree(n) for n in G.nodes() if n.startswith("card:")], 95
                )),
            })

        out = pd.DataFrame(records, index=df.index)
        logger.success(f"Graph features shape: {out.shape}")
        return out

    def save_graph(self, G: nx.Graph) -> None:
        path = self.model_dir / "fraud_graph.pkl"
        with open(path, "wb") as f:
            pickle.dump(G, f)
        logger.success(f"Saved graph → {path}")

    @classmethod
    def load_graph(cls, model_dir: str | Path) -> nx.Graph:
        path = Path(model_dir) / "fraud_graph.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _zero_features() -> dict:
        return {
            "graph_degree": 0,
            "graph_pagerank": 0.0,
            "graph_clustering": 0.0,
            "graph_component_size": 1,
            "graph_n_device_neighbors": 0,
            "graph_n_email_neighbors": 0,
            "graph_n_co_device_cards": 0,
            "graph_is_hub": 0,
        }
