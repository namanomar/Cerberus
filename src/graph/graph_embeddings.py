

from __future__ import annotations

import io
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from loguru import logger

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False
    logger.warning("node2vec not installed. Using random walk fallback embeddings.")


@contextmanager
def _suppress_stderr():
    """
    Suppress stderr during gensim/Node2Vec Word2Vec training on Python 3.12+.

    Root cause: gensim's Cython `word2vec_inner.our_dot_float` raises a
    harmless exception on *every* worker thread teardown due to a CPython 3.12
    change in how tp_finalize interacts with Cython extension types.
    Bug tracker: https://github.com/piskvorky/gensim/issues/3390

    This context manager silences those hundreds of identical lines without
    suppressing any real error output — it restores stderr afterwards and
    re-emits anything that is NOT the known gensim noise.
    """
    GENSIM_NOISE = "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'"
    old_stderr = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        yield
    finally:
        sys.stderr = old_stderr
        captured = buf.getvalue()
        # Re-emit any lines that aren't the known gensim spam
        real_errors = [
            line for line in captured.splitlines()
            if line.strip() and GENSIM_NOISE not in line
        ]
        if real_errors:
            print("\n".join(real_errors), file=sys.stderr)


class GraphEmbedder:
    """
    Trains Node2Vec on the fraud graph and provides embeddings for
    each entity node (card, device, email, address).

    Usage
    -----
    >>> embedder = GraphEmbedder("config/config.yaml")
    >>> embedder.fit(G)
    >>> card_embs = embedder.get_card_embeddings(df)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        n2v_cfg = cfg["graph"]["node2vec"]
        self.dimensions = n2v_cfg["dimensions"]
        self.walk_length = n2v_cfg["walk_length"]
        self.num_walks = n2v_cfg["num_walks"]
        self.p = n2v_cfg["p"]
        self.q = n2v_cfg["q"]
        self.workers = n2v_cfg["workers"]
        self.window = n2v_cfg["window"]
        self.min_count = n2v_cfg["min_count"]
        self.batch_words = n2v_cfg["batch_words"]

        self.model_dir = Path(cfg["data"].get("models", "data/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings: dict[str, np.ndarray] = {}
        self._n2v_model = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, G: nx.Graph) -> "GraphEmbedder":
        """Train Node2Vec embeddings on graph G."""
        if not NODE2VEC_AVAILABLE:
            logger.warning("Using fallback random embeddings (install node2vec for real embeddings)")
            self._fit_fallback(G)
            return self

        # ── Python 3.12+ gensim Cython bug: workers > 1 floods stderr ────────
        # Force single-threaded on 3.12+ to avoid the spam entirely.
        # On 3.11 and below, multi-worker is fine.
        py_ver = sys.version_info
        workers = self.workers
        if py_ver >= (3, 12) and workers > 1:
            logger.warning(
                f"Python {py_ver.major}.{py_ver.minor} detected — forcing Node2Vec "
                f"workers=1 to avoid gensim Cython stderr spam (gensim issue #3390). "
                f"Training will be slightly slower but output will be clean."
            )
            workers = 1

        logger.info(
            f"Training Node2Vec: dim={self.dimensions}, "
            f"walk_length={self.walk_length}, num_walks={self.num_walks}, "
            f"workers={workers} …"
        )

        # Walk generation: run outside suppressor so tqdm bars are visible
        node2vec = Node2Vec(
            G,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=workers,
            quiet=False,          # shows "Computing transition probabilities" + walk bars
        )

        # Word2Vec fit: wrap in suppressor — this is the only source of the spam
        logger.info("Fitting Word2Vec on walks (stderr spam suppressed) …")
        with _suppress_stderr():
            self._n2v_model = node2vec.fit(
                window=self.window,
                min_count=self.min_count,
                batch_words=self.batch_words,
            )

        # Cache all node embeddings as numpy arrays
        for node in G.nodes():
            if node in self._n2v_model.wv:
                self._embeddings[node] = self._n2v_model.wv[node]

        logger.success(f"Trained embeddings for {len(self._embeddings):,} nodes")
        return self

    def get_embedding(self, node_id: str) -> np.ndarray:
        """Return embedding for a node, or zero vector if not found."""
        return self._embeddings.get(node_id, np.zeros(self.dimensions))

    def get_card_embeddings(self, df: pd.DataFrame, card_col: str = "card1") -> pd.DataFrame:
        """
        Return a DataFrame of card node embeddings,
        one row per transaction (matched by card_col).
        """
        emb_cols = [f"emb_card_{i}" for i in range(self.dimensions)]
        rows = []

        for card_val in df[card_col].astype(str):
            node_id = f"card:{card_val}"
            rows.append(self.get_embedding(node_id))

        return pd.DataFrame(rows, columns=emb_cols, index=df.index)

    def get_device_embeddings(
        self, df: pd.DataFrame, device_col: str = "DeviceInfo"
    ) -> pd.DataFrame:
        """Return device node embeddings per transaction."""
        emb_cols = [f"emb_device_{i}" for i in range(self.dimensions)]
        rows = []

        for dev_val in df[device_col].fillna("Unknown").astype(str):
            node_id = f"device:{dev_val}"
            rows.append(self.get_embedding(node_id))

        return pd.DataFrame(rows, columns=emb_cols, index=df.index)

    def get_all_embeddings_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get concatenated card + device embeddings per transaction.
        Dimensionality: 2 × self.dimensions
        """
        card_emb = self.get_card_embeddings(df)
        device_emb = self.get_device_embeddings(df)
        return pd.concat([card_emb, device_emb], axis=1)

    def save(self) -> None:
        emb_path = self.model_dir / "graph_embeddings.pkl"
        with open(emb_path, "wb") as f:
            pickle.dump(self._embeddings, f)

        meta_path = self.model_dir / "graph_embedder_meta.pkl"
        meta = {"dimensions": self.dimensions}
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.success(f"Saved embeddings → {emb_path}")

    def load(self) -> None:
        emb_path = self.model_dir / "graph_embeddings.pkl"
        with open(emb_path, "rb") as f:
            self._embeddings = pickle.load(f)

        meta_path = self.model_dir / "graph_embedder_meta.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.dimensions = meta["dimensions"]
        logger.success(f"Loaded {len(self._embeddings):,} embeddings (dim={self.dimensions})")

    # ──────────────────────────────────────────────────────────────────────────
    # Fallback (no node2vec installed)
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_fallback(self, G: nx.Graph) -> None:
        """
        Simple random-walk graph feature extraction as Node2Vec fallback.
        Produces degree-based structural features instead of dense embeddings.
        """
        rng = np.random.default_rng(42)
        for node in G.nodes():
            degree = G.degree(node)
            # Deterministic pseudo-embedding based on degree + hash
            seed_vec = rng.standard_normal(self.dimensions)
            # Scale by log degree to encode connectivity
            seed_vec = seed_vec * np.log1p(degree) / self.dimensions
            self._embeddings[node] = seed_vec.astype(np.float32)
