from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...config import ProjectConfig
    from ...data import PreparedDataset

LOGGER = logging.getLogger(__name__)


class Node2VecCorpus:
    def __init__(
        self,
        graph: nx.Graph,
        walk_length: int,
        num_walks: int,
        seed: int,
        return_p: float,
        inout_q: float,
    ) -> None:
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.seed = seed
        self.return_p = max(return_p, 1e-6)
        self.inout_q = max(inout_q, 1e-6)

    def __iter__(self):
        rng = random.Random(self.seed)
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            rng.shuffle(nodes)
            for node in nodes:
                yield self._walk(node, rng)

    def _walk(self, start: str, rng: random.Random) -> list[str]:
        walk = [start]
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            if len(walk) == 1:
                walk.append(rng.choice(neighbors))
                continue
            previous = walk[-2]
            weights = []
            for neighbor in neighbors:
                base_weight = float(self.graph[current][neighbor].get("weight", 1.0))
                if neighbor == previous:
                    bias = 1.0 / self.return_p
                elif self.graph.has_edge(previous, neighbor):
                    bias = 1.0
                else:
                    bias = 1.0 / self.inout_q
                weights.append(base_weight * bias)
            walk.append(_weighted_choice(neighbors, weights, rng))
        return walk


def _weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]


def compute_node2vec_embeddings(config: "ProjectConfig", prepared: "PreparedDataset") -> pd.DataFrame:
    cache_path = config.cache_dir / "node2vec_embeddings.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, low_memory=False)
        if _is_valid_embedding_cache(cached, prepared.users["user_id"], prefix="n2v_", expected_dim=config.node2vec_dimensions):
            return cached
        LOGGER.warning("Existing Node2Vec cache does not match the current prepared dataset. Recomputing embeddings.")

    try:
        from gensim.models import Word2Vec
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping Node2Vec because gensim is unavailable: %s", exc)
        return pd.DataFrame({"user_id": prepared.users["user_id"]})

    graph = nx.Graph()
    graph.add_nodes_from(prepared.users["user_id"].tolist())
    for row in prepared.graph_edges.itertuples(index=False):
        weight = 1.0
        if graph.has_edge(row.source_id, row.target_id):
            graph[row.source_id][row.target_id]["weight"] += weight
        else:
            graph.add_edge(row.source_id, row.target_id, weight=weight)

    corpus = Node2VecCorpus(
        graph=graph,
        walk_length=config.node2vec_walk_length,
        num_walks=config.node2vec_num_walks,
        seed=config.random_state,
        return_p=config.node2vec_return_p,
        inout_q=config.node2vec_inout_q,
    )
    model = Word2Vec(
        vector_size=config.node2vec_dimensions,
        window=config.node2vec_window,
        min_count=0,
        sg=1,
        workers=max(1, config.node2vec_workers),
        epochs=config.node2vec_epochs,
        seed=config.random_state,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    rows = []
    for user_id in prepared.users["user_id"].tolist():
        if user_id in model.wv:
            embedding = model.wv[user_id]
        else:
            embedding = np.zeros(config.node2vec_dimensions, dtype=np.float32)
        row = {"user_id": user_id}
        for index, value in enumerate(embedding.tolist()):
            row[f"n2v_{index}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def _is_valid_embedding_cache(
    frame: pd.DataFrame,
    expected_user_ids: pd.Series,
    prefix: str,
    expected_dim: int | None = None,
) -> bool:
    if "user_id" not in frame.columns:
        return False
    embedding_columns = [column for column in frame.columns if column.startswith(prefix)]
    if expected_dim is not None and len(embedding_columns) != expected_dim:
        return False
    expected_ids = set(expected_user_ids.astype(str).tolist())
    cached_ids = set(frame["user_id"].astype(str).tolist())
    if expected_ids != cached_ids:
        return False
    if frame["user_id"].duplicated().any():
        return False
    if embedding_columns and frame[embedding_columns].isna().any().any():
        return False
    return True
