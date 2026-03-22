from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .config import PipelineConfig
from .data import PreparedData

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphArtifacts:
    users: pd.DataFrame
    embeddings: pd.DataFrame
    network_summary: dict[str, float | int | dict[str, int]]


def enrich_with_graph_features(config: PipelineConfig, prepared: PreparedData) -> GraphArtifacts:
    graph = build_user_graph(prepared.graph_edges, graph_nodes=prepared.graph_user_ids)
    LOGGER.info("Built graph with %s nodes and %s directed edges.", graph.number_of_nodes(), graph.number_of_edges())

    graph_features, graph_summary = compute_graph_features(
        graph=graph,
        target_user_ids=prepared.users["user_id"].tolist(),
        betweenness_sample_k=config.betweenness_sample_k,
        harmonic_sample_sources=config.harmonic_sample_sources,
        seed=config.random_state,
    )

    undirected = graph.to_undirected()
    deepwalk_embeddings = train_deepwalk_embeddings(
        undirected,
        target_user_ids=prepared.users["user_id"].tolist(),
        dimensions=config.deepwalk_dimensions,
        walk_length=config.deepwalk_walk_length,
        num_walks=config.deepwalk_num_walks,
        window=config.deepwalk_window,
        epochs=config.deepwalk_epochs,
        seed=config.random_state,
    )
    node2vec_embeddings = train_node2vec_embeddings(
        undirected,
        target_user_ids=prepared.users["user_id"].tolist(),
        dimensions=config.node2vec_dimensions,
        walk_length=config.node2vec_walk_length,
        num_walks=config.node2vec_num_walks,
        window=config.node2vec_window,
        epochs=config.node2vec_epochs,
        seed=config.random_state,
        return_p=config.node2vec_return_p,
        inout_q=config.node2vec_inout_q,
    )
    embeddings = deepwalk_embeddings.merge(node2vec_embeddings, on="user_id", how="outer").fillna(0.0)

    users = prepared.users.merge(graph_features, on="user_id", how="left")
    graph_numeric_columns = [
        "graph_in_degree",
        "graph_out_degree",
        "graph_total_degree",
        "graph_pagerank",
        "graph_clustering",
        "graph_core_number",
        "graph_component_size",
        "graph_betweenness_approx",
        "graph_harmonic_approx",
        "graph_reciprocity",
    ]
    prepared.manifest["graph_numeric_columns"] = graph_numeric_columns
    prepared.manifest["deepwalk_embedding_columns"] = [column for column in deepwalk_embeddings.columns if column != "user_id"]
    prepared.manifest["node2vec_embedding_columns"] = [column for column in node2vec_embeddings.columns if column != "user_id"]
    prepared.manifest["embedding_columns"] = [column for column in embeddings.columns if column != "user_id"]
    for column in graph_numeric_columns:
        users[column] = users[column].fillna(0.0)

    users.to_csv(config.cache_dir / "users.csv", index=False)
    with (config.cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.manifest, handle, ensure_ascii=False, indent=2)

    summary = dict(prepared.network_summary)
    summary.update(graph_summary)
    with (config.cache_dir / "network_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    dump_embedding_payload(config.cache_dir / "deepwalk_embeddings.joblib", deepwalk_embeddings)
    dump_embedding_payload(config.cache_dir / "node2vec_embeddings.joblib", node2vec_embeddings)
    dump_embedding_payload(config.cache_dir / "graph_embeddings.joblib", embeddings)

    return GraphArtifacts(users=users, embeddings=embeddings, network_summary=summary)


def dump_embedding_payload(path, embeddings: pd.DataFrame) -> None:
    payload = {
        "user_id": embeddings["user_id"].astype(str).tolist(),
        "embeddings": embeddings.drop(columns=["user_id"]).to_numpy(dtype=np.float32),
        "columns": [column for column in embeddings.columns if column != "user_id"],
    }
    joblib.dump(payload, path)


def build_user_graph(graph_edges: pd.DataFrame, graph_nodes: Iterable[str] | None) -> nx.DiGraph:
    graph = nx.DiGraph()
    if graph_nodes is not None:
        graph.add_nodes_from(graph_nodes)
    for row in graph_edges.itertuples(index=False):
        source = row.source_id
        target = row.target_id
        weight = float(row.weight)
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight, relations=str(getattr(row, "relations", "")))
    return graph


def compute_graph_features(
    graph: nx.DiGraph,
    target_user_ids: list[str],
    betweenness_sample_k: int,
    harmonic_sample_sources: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if graph.number_of_nodes() == 0:
        empty_df = pd.DataFrame({"user_id": target_user_ids})
        return empty_df, {"density": 0.0, "average_clustering": 0.0, "reciprocity": 0.0}

    undirected = graph.to_undirected()
    target_set = set(target_user_ids)

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    total_degree = dict(undirected.degree())
    pagerank = nx.pagerank(graph, alpha=0.85, weight="weight")
    clustering = nx.clustering(undirected, weight="weight")
    core_number = nx.core_number(undirected) if undirected.number_of_edges() > 0 else {node: 0 for node in graph.nodes()}

    betweenness_k = min(max(1, betweenness_sample_k), max(1, undirected.number_of_nodes() - 1))
    betweenness = nx.betweenness_centrality(undirected, k=betweenness_k, seed=seed, normalized=True)
    harmonic = approximate_harmonic_centrality(undirected, sample_size=harmonic_sample_sources, seed=seed)

    reciprocity_values = nx.reciprocity(graph, nodes=graph.nodes())
    reciprocity_values = {
        node: float(value) if value is not None else 0.0 for node, value in reciprocity_values.items()
    }

    component_size: dict[str, int] = {}
    for component in nx.connected_components(undirected):
        size = len(component)
        for node in component:
            component_size[node] = size

    records = []
    for user_id in target_user_ids:
        records.append(
            {
                "user_id": user_id,
                "graph_in_degree": float(in_degree.get(user_id, 0)),
                "graph_out_degree": float(out_degree.get(user_id, 0)),
                "graph_total_degree": float(total_degree.get(user_id, 0)),
                "graph_pagerank": float(pagerank.get(user_id, 0.0)),
                "graph_clustering": float(clustering.get(user_id, 0.0)),
                "graph_core_number": float(core_number.get(user_id, 0.0)),
                "graph_component_size": float(component_size.get(user_id, 1)),
                "graph_betweenness_approx": float(betweenness.get(user_id, 0.0)),
                "graph_harmonic_approx": float(harmonic.get(user_id, 0.0)),
                "graph_reciprocity": float(reciprocity_values.get(user_id, 0.0)),
            }
        )

    graph_summary = {
        "density": float(nx.density(undirected)),
        "average_clustering": float(nx.average_clustering(undirected)) if undirected.number_of_edges() else 0.0,
        "reciprocity": float(nx.overall_reciprocity(graph) or 0.0) if graph.number_of_edges() else 0.0,
        "largest_component_size": int(max(component_size.values(), default=0)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "target_nodes": int(len(target_set)),
    }
    return pd.DataFrame(records), graph_summary


def approximate_harmonic_centrality(graph: nx.Graph, sample_size: int, seed: int) -> dict[str, float]:
    nodes = list(graph.nodes())
    if not nodes:
        return {}

    rng = random.Random(seed)
    if sample_size <= 0 or sample_size >= len(nodes):
        sources = nodes
    else:
        sources = rng.sample(nodes, sample_size)

    scores = defaultdict(float)
    for source in sources:
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target, distance in lengths.items():
            if source == target or distance <= 0:
                continue
            scores[target] += 1.0 / float(distance)

    scale = len(nodes) / max(1, len(sources))
    normalizer = max(1, len(nodes) - 1)
    return {node: float(scale * scores.get(node, 0.0) / normalizer) for node in nodes}


class RandomWalkCorpus:
    def __iter__(self):
        raise NotImplementedError


class DeepWalkCorpus(RandomWalkCorpus):
    def __init__(self, graph: nx.Graph, num_walks: int, walk_length: int, seed: int):
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.seed = seed
        self.nodes = list(graph.nodes())
        self.adjacency = {node: list(graph.neighbors(node)) for node in self.nodes}

    def __iter__(self):
        rng = random.Random(self.seed)
        nodes = list(self.nodes)
        for _ in range(self.num_walks):
            rng.shuffle(nodes)
            for node in nodes:
                yield self.generate_walk(node, rng)

    def generate_walk(self, start_node: str, rng: random.Random) -> list[str]:
        walk = [str(start_node)]
        current = start_node
        for _ in range(self.walk_length - 1):
            neighbors = self.adjacency.get(current)
            if not neighbors:
                break
            current = rng.choice(neighbors)
            walk.append(str(current))
        return walk


class Node2VecCorpus(DeepWalkCorpus):
    def __init__(
        self,
        graph: nx.Graph,
        num_walks: int,
        walk_length: int,
        seed: int,
        return_p: float,
        inout_q: float,
    ):
        super().__init__(graph=graph, num_walks=num_walks, walk_length=walk_length, seed=seed)
        self.return_p = max(return_p, 1e-6)
        self.inout_q = max(inout_q, 1e-6)

    def generate_walk(self, start_node: str, rng: random.Random) -> list[str]:
        walk = [str(start_node)]
        previous = None
        current = start_node
        for _ in range(self.walk_length - 1):
            neighbors = self.adjacency.get(current)
            if not neighbors:
                break
            if previous is None:
                current = rng.choice(neighbors)
            else:
                weights = [self.transition_weight(previous, current, neighbor) for neighbor in neighbors]
                current = weighted_choice(neighbors, weights, rng)
            walk.append(str(current))
            previous = walk[-2]
        return walk

    def transition_weight(self, previous_node: str, current_node: str, next_node: str) -> float:
        previous_node = str(previous_node)
        current_node = str(current_node)
        next_node = str(next_node)
        if next_node == previous_node:
            return 1.0 / self.return_p
        if self.graph.has_edge(next_node, previous_node) or self.graph.has_edge(previous_node, next_node):
            return 1.0
        return 1.0 / self.inout_q


def weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(max(weight, 0.0) for weight in weights)
    if total <= 0:
        return rng.choice(items)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(weight, 0.0)
        if cumulative >= threshold:
            return item
    return items[-1]


def train_deepwalk_embeddings(
    graph: nx.Graph,
    target_user_ids: list[str],
    dimensions: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    seed: int,
) -> pd.DataFrame:
    corpus = DeepWalkCorpus(graph=graph, num_walks=num_walks, walk_length=walk_length, seed=seed)
    return train_random_walk_embeddings(
        corpus=corpus,
        graph=graph,
        target_user_ids=target_user_ids,
        dimensions=dimensions,
        window=window,
        epochs=epochs,
        seed=seed,
        prefix="dw",
    )


def train_node2vec_embeddings(
    graph: nx.Graph,
    target_user_ids: list[str],
    dimensions: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    seed: int,
    return_p: float,
    inout_q: float,
) -> pd.DataFrame:
    corpus = Node2VecCorpus(
        graph=graph,
        num_walks=num_walks,
        walk_length=walk_length,
        seed=seed,
        return_p=return_p,
        inout_q=inout_q,
    )
    return train_random_walk_embeddings(
        corpus=corpus,
        graph=graph,
        target_user_ids=target_user_ids,
        dimensions=dimensions,
        window=window,
        epochs=epochs,
        seed=seed,
        prefix="n2v",
    )


def train_random_walk_embeddings(
    corpus: RandomWalkCorpus,
    graph: nx.Graph,
    target_user_ids: list[str],
    dimensions: int,
    window: int,
    epochs: int,
    seed: int,
    prefix: str,
) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        data = np.zeros((len(target_user_ids), dimensions), dtype=np.float32)
        frame = pd.DataFrame(data, columns=[f"{prefix}_{idx}" for idx in range(dimensions)])
        frame.insert(0, "user_id", target_user_ids)
        return frame

    workers = max(1, (os.cpu_count() or 1) - 1)
    model = Word2Vec(
        sentences=corpus,
        vector_size=dimensions,
        window=window,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    columns = [f"{prefix}_{idx}" for idx in range(dimensions)]
    zero_vector = np.zeros(dimensions, dtype=np.float32)
    rows = []
    for user_id in target_user_ids:
        vector = model.wv[user_id] if user_id in model.wv else zero_vector
        rows.append([user_id, *vector.tolist()])
    return pd.DataFrame(rows, columns=["user_id", *columns])
