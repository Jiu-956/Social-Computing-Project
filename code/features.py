from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
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
    embeddings = train_deepwalk_embeddings(
        graph.to_undirected(),
        target_user_ids=prepared.users["user_id"].tolist(),
        dimensions=config.deepwalk_dimensions,
        walk_length=config.deepwalk_walk_length,
        num_walks=config.deepwalk_num_walks,
        window=config.deepwalk_window,
        epochs=config.deepwalk_epochs,
        seed=config.random_state,
    )

    users = prepared.users.merge(graph_features, on="user_id", how="left")

    prepared.manifest["graph_numeric_columns"] = [
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
    prepared.manifest["embedding_columns"] = [column for column in embeddings.columns if column != "user_id"]
    for column in prepared.manifest["graph_numeric_columns"]:
        users[column] = users[column].fillna(0.0)

    users.to_csv(config.cache_dir / "users.csv", index=False)
    with (config.cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.manifest, handle, ensure_ascii=False, indent=2)

    summary = dict(prepared.network_summary)
    summary.update(graph_summary)
    with (config.cache_dir / "network_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    embeddings_payload = {
        "user_id": embeddings["user_id"].tolist(),
        "embeddings": embeddings.drop(columns=["user_id"]).to_numpy(dtype=np.float32),
        "columns": [column for column in embeddings.columns if column != "user_id"],
    }
    joblib.dump(embeddings_payload, config.cache_dir / "deepwalk_embeddings.joblib")

    return GraphArtifacts(users=users, embeddings=embeddings, network_summary=summary)


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
            graph.add_edge(source, target, weight=weight)
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
                yield self._generate_walk(node, rng)

    def _generate_walk(self, start_node: str, rng: random.Random) -> list[str]:
        walk = [str(start_node)]
        current = start_node
        for _ in range(self.walk_length - 1):
            neighbors = self.adjacency.get(current)
            if not neighbors:
                break
            current = rng.choice(neighbors)
            walk.append(str(current))
        return walk


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
    if graph.number_of_nodes() == 0:
        data = np.zeros((len(target_user_ids), dimensions), dtype=np.float32)
        frame = pd.DataFrame(data, columns=[f"dw_{idx}" for idx in range(dimensions)])
        frame.insert(0, "user_id", target_user_ids)
        return frame

    corpus = RandomWalkCorpus(graph=graph, num_walks=num_walks, walk_length=walk_length, seed=seed)
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

    rows = []
    columns = [f"dw_{idx}" for idx in range(dimensions)]
    zero_vector = np.zeros(dimensions, dtype=np.float32)
    for user_id in target_user_ids:
        vector = model.wv[user_id] if user_id in model.wv else zero_vector
        rows.append([user_id, *vector.tolist()])
    return pd.DataFrame(rows, columns=["user_id", *columns])
