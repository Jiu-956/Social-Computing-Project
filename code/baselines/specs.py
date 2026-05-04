from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExperimentSpec:
    name: str
    family: str
    estimator: str
    numeric_columns: tuple[str, ...]
    text_mode: str = "none"
    text_column: str = "combined_text"


FEATURE_COLUMNS: tuple[str, ...] = (
    "followers_count",
    "following_count",
    "tweet_count",
    "like_count",
    "listed_count",
    "account_creation_year",
    "account_creation_month",
    "account_creation_day",
    "account_creation_hour",
    "has_location",
    "has_description",
    "has_url",
    "has_banner",
    "is_verified",
    "has_extended_profile",
    "default_profile",
)

FEATURE_CAT_COLUMNS: tuple[str, ...] = (
    "default_profile_color",
    "profile_location",
    "lang",
)

GRAPH_COLUMNS: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_degree",
    "reciprocal_ratio",
    "avg_neighbor_degree",
    "clustering_coefficient",
    "page_rank",
    "triangles",
    "core_number",
    "is_bridges",
)


def _make_specs(
    feature_cols: tuple[str, ...],
    graph_cols: tuple[str, ...],
    node2vec_cols: tuple[str, ...],
    transformer_cols: tuple[str, ...],
) -> list[ExperimentSpec]:
    feature_all = feature_cols + FEATURE_CAT_COLUMNS
    specs: list[ExperimentSpec] = [
        ExperimentSpec(
            name="feature_only_logistic_regression",
            family="feature_only",
            estimator="logreg",
            numeric_columns=feature_all,
        ),
        ExperimentSpec(
            name="feature_only_random_forest",
            family="feature_only",
            estimator="rf",
            numeric_columns=feature_all,
        ),
        ExperimentSpec(
            name="text_only_tfidf_logistic_regression",
            family="text_only",
            estimator="logreg",
            numeric_columns=(),
            text_mode="tfidf",
        ),
        ExperimentSpec(
            name="graph_only_structure_random_forest",
            family="graph_only",
            estimator="rf",
            numeric_columns=graph_cols,
        ),
        ExperimentSpec(
            name="feature_text_tfidf_logistic_regression",
            family="feature_text",
            estimator="logreg",
            numeric_columns=feature_all,
            text_mode="tfidf",
        ),
        ExperimentSpec(
            name="feature_graph_random_forest",
            family="feature_graph",
            estimator="rf",
            numeric_columns=feature_all + graph_cols,
        ),
    ]
    if node2vec_cols:
        specs.extend([
            ExperimentSpec(
                name="graph_only_node2vec_logistic_regression",
                family="graph_only",
                estimator="logreg",
                numeric_columns=node2vec_cols,
            ),
            ExperimentSpec(
                name="feature_graph_node2vec_logistic_regression",
                family="feature_graph",
                estimator="logreg",
                numeric_columns=feature_all + graph_cols + node2vec_cols,
            ),
            ExperimentSpec(
                name="feature_text_graph_tfidf_node2vec_logistic_regression",
                family="feature_text_graph",
                estimator="logreg",
                numeric_columns=feature_all + graph_cols + node2vec_cols,
                text_mode="tfidf",
            ),
        ])
    if transformer_cols:
        specs.extend([
            ExperimentSpec(
                name="text_only_transformer_logistic_regression",
                family="text_only",
                estimator="logreg",
                numeric_columns=transformer_cols,
            ),
            ExperimentSpec(
                name="feature_text_transformer_logistic_regression",
                family="feature_text",
                estimator="logreg",
                numeric_columns=feature_all + transformer_cols,
            ),
        ])
    return specs
