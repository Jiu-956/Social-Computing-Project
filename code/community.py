from __future__ import annotations

import json
import logging

import networkx as nx
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def run_community_analysis(
    config: PipelineConfig,
    split: str = "test",
    threshold: float = 0.8,
    use_ground_truth: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(config.cache_dir / "users.csv")
    predictions = pd.read_csv(config.tables_dir / "classification_predictions.csv")
    best_info = json.loads((config.models_dir / "best_experiment.json").read_text(encoding="utf-8"))
    predictions = predictions[predictions["experiment"] == best_info["best_experiment"]]

    split_df = users[users["split"] == split].copy()
    split_predictions = predictions[predictions["split"] == split].copy()
    split_df = split_df.merge(
        split_predictions[["user_id", "bot_probability", "prediction"]],
        on="user_id",
        how="left",
    )

    if use_ground_truth:
        candidate_df = split_df[split_df["label_id"] == 1].copy()
    else:
        candidate_df = split_df[split_df["bot_probability"].fillna(0.0) >= threshold].copy()
        if len(candidate_df) < 10:
            candidate_df = split_df.nlargest(min(200, len(split_df)), "bot_probability").copy()

    graph_edges = pd.read_csv(config.cache_dir / "graph_edges.csv")
    graph = nx.Graph()
    candidate_ids = set(candidate_df["user_id"].astype(str))
    for row in graph_edges.itertuples(index=False):
        source_id = str(row.source_id)
        target_id = str(row.target_id)
        if source_id in candidate_ids and target_id in candidate_ids:
            graph.add_edge(source_id, target_id, weight=float(row.weight))

    overview_metrics = {
        "split": split,
        "threshold": float(threshold),
        "use_ground_truth": bool(use_ground_truth),
        "candidate_count": int(len(candidate_df)),
        "candidate_with_edges": int(graph.number_of_nodes()),
        "isolated_candidate_count": int(max(0, len(candidate_df) - graph.number_of_nodes())),
        "subgraph_edge_count": int(graph.number_of_edges()),
        "connected_component_count": int(nx.number_connected_components(graph)) if graph.number_of_nodes() > 0 else 0,
    }

    if graph.number_of_nodes() == 0:
        empty_assignments = pd.DataFrame(columns=["method", "user_id", "community_id", "size", "bot_ratio", "average_probability", "density"])
        empty_summary = pd.DataFrame(columns=["method", "community_count", "modularity", "largest_community_size", "average_community_size", "average_bot_ratio"])
        empty_assignments.to_csv(config.tables_dir / "community_assignments.csv", index=False)
        empty_summary.to_csv(config.tables_dir / "community_summary.csv", index=False)
        with (config.tables_dir / "community_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({**overview_metrics, "methods": {}}, handle, ensure_ascii=False, indent=2)
        return empty_assignments, empty_summary

    methods = {
        "louvain": nx.community.louvain_communities(graph, seed=config.random_state, weight="weight"),
        "greedy_modularity": list(nx.community.greedy_modularity_communities(graph, weight="weight")),
    }

    assignment_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    method_metrics: dict[str, dict[str, float | int]] = {}
    candidate_lookup = candidate_df.set_index("user_id")

    for method_name, communities in methods.items():
        if not communities:
            continue
        communities = [set(community) for community in communities if community]
        partition = [community for community in communities if len(community) > 0]
        if not partition:
            continue

        modularity = float(nx.community.modularity(graph, partition, weight="weight"))
        community_sizes = [len(community) for community in partition]
        bot_ratios = []
        community_densities = []

        for community_id, community in enumerate(sorted(partition, key=len, reverse=True)):
            member_ids = sorted(community)
            subgraph = graph.subgraph(member_ids).copy()
            member_df = candidate_lookup.loc[member_ids].copy()
            bot_ratio = float(member_df["label_id"].mean()) if len(member_df) else 0.0
            density = float(nx.density(subgraph)) if subgraph.number_of_nodes() > 1 else 0.0
            avg_probability = float(member_df["bot_probability"].fillna(0.0).mean())
            bot_ratios.append(bot_ratio)
            community_densities.append(density)

            for user_id in member_ids:
                assignment_rows.append(
                    {
                        "method": method_name,
                        "user_id": user_id,
                        "community_id": int(community_id),
                        "size": int(len(member_ids)),
                        "bot_ratio": bot_ratio,
                        "average_probability": avg_probability,
                        "density": density,
                    }
                )

        summary_rows.append(
            {
                "method": method_name,
                "community_count": int(len(partition)),
                "modularity": modularity,
                "largest_community_size": int(max(community_sizes, default=0)),
                "average_community_size": float(sum(community_sizes) / max(1, len(community_sizes))),
                "average_bot_ratio": float(sum(bot_ratios) / max(1, len(bot_ratios))),
                "average_density": float(sum(community_densities) / max(1, len(community_densities))),
            }
        )
        method_metrics[method_name] = {
            "community_count": int(len(partition)),
            "modularity": modularity,
            "largest_community_size": int(max(community_sizes, default=0)),
        }

    assignments_df = pd.DataFrame(assignment_rows).sort_values(["method", "community_id", "user_id"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("modularity", ascending=False).reset_index(drop=True)

    assignments_df.to_csv(config.tables_dir / "community_assignments.csv", index=False)
    summary_df.to_csv(config.tables_dir / "community_summary.csv", index=False)
    with (config.tables_dir / "community_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({**overview_metrics, "methods": method_metrics}, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Community analysis completed with methods: %s", ", ".join(summary_df["method"].tolist()))
    return assignments_df, summary_df
