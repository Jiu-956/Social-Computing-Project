from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from code.config import ProjectConfig
from code.data import prepare_dataset
from code.experiments import run_experiments
from code.reporting import generate_report
from code.visualization import generate_visualizations


class SmokeTest(unittest.TestCase):
    def test_end_to_end_pipeline_on_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            output_dir = root / "artifacts"
            data_dir.mkdir()

            self._write_dataset(data_dir)
            config = ProjectConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                max_tweets_per_user=2,
                tfidf_max_features=100,
                tfidf_min_df=1,
                use_transformer=False,
                run_node2vec=True,
                node2vec_dimensions=8,
                node2vec_walk_length=6,
                node2vec_num_walks=3,
                node2vec_window=3,
                node2vec_epochs=2,
                node2vec_workers=1,
                run_gnn=True,
                gnn_hidden_dim=16,
                gnn_epochs=6,
                gnn_patience=3,
                visualization_sample_size=50,
                random_state=7,
            )

            prepare_dataset(config)
            outputs = run_experiments(config)
            generate_visualizations(config)
            report_path = generate_report(config)

            self.assertTrue((output_dir / "cache" / "users.csv").exists())
            self.assertTrue((output_dir / "tables" / "experiment_metrics.csv").exists())
            self.assertTrue((output_dir / "figures" / "model_comparison.png").exists())
            self.assertTrue(report_path.exists())
            families = set(outputs["metrics"]["family"])
            self.assertIn("feature_only", families)
            self.assertIn("text_only", families)
            self.assertIn("graph_only", families)
            self.assertIn("feature_text", families)
            self.assertIn("feature_graph", families)
            self.assertIn("feature_text_graph", families)

    def _write_dataset(self, data_dir: Path) -> None:
        labels = pd.DataFrame(
            [
                {"id": "u1", "label": "human"},
                {"id": "u2", "label": "bot"},
                {"id": "u3", "label": "human"},
                {"id": "u4", "label": "bot"},
                {"id": "u5", "label": "human"},
                {"id": "u6", "label": "bot"},
                {"id": "u7", "label": "human"},
                {"id": "u8", "label": "bot"},
            ]
        )
        labels.to_csv(data_dir / "label.csv", index=False)

        splits = pd.DataFrame(
            [
                {"id": "u1", "split": "train"},
                {"id": "u2", "split": "train"},
                {"id": "u3", "split": "train"},
                {"id": "u4", "split": "train"},
                {"id": "u5", "split": "val"},
                {"id": "u6", "split": "val"},
                {"id": "u7", "split": "test"},
                {"id": "u8", "split": "test"},
                {"id": "u9", "split": "support"},
                {"id": "u10", "split": "support"},
            ]
        )
        splits.to_csv(data_dir / "split.csv", index=False)

        edges = pd.DataFrame(
            [
                {"source_id": "u1", "relation": "follow", "target_id": "u2"},
                {"source_id": "u2", "relation": "follow", "target_id": "u4"},
                {"source_id": "u3", "relation": "friend", "target_id": "u1"},
                {"source_id": "u4", "relation": "friend", "target_id": "u2"},
                {"source_id": "u5", "relation": "follow", "target_id": "u2"},
                {"source_id": "u6", "relation": "friend", "target_id": "u4"},
                {"source_id": "u7", "relation": "follow", "target_id": "u1"},
                {"source_id": "u8", "relation": "friend", "target_id": "u2"},
                {"source_id": "u9", "relation": "follow", "target_id": "u2"},
                {"source_id": "u10", "relation": "friend", "target_id": "u4"},
                {"source_id": "u1", "relation": "post", "target_id": "t1"},
                {"source_id": "u2", "relation": "post", "target_id": "t2"},
                {"source_id": "u3", "relation": "post", "target_id": "t3"},
                {"source_id": "u4", "relation": "post", "target_id": "t4"},
                {"source_id": "u5", "relation": "post", "target_id": "t5"},
                {"source_id": "u6", "relation": "post", "target_id": "t6"},
                {"source_id": "u7", "relation": "post", "target_id": "t7"},
                {"source_id": "u8", "relation": "post", "target_id": "t8"},
            ]
        )
        edges.to_csv(data_dir / "edge.csv", index=False)

        nodes = [
            self._user("u1", followers=10, following=8, tweets=12, description="sports family sunshine", verified=True),
            self._user("u2", followers=2, following=40, tweets=90, description="win crypto bonus now limited", verified=False),
            self._user("u3", followers=15, following=5, tweets=20, description="music books and coffee", verified=False),
            self._user("u4", followers=1, following=55, tweets=120, description="cheap promo bot sale fast", verified=False),
            self._user("u5", followers=12, following=6, tweets=18, description="local news and school", verified=False),
            self._user("u6", followers=1, following=60, tweets=140, description="airdrop signal buy now", verified=False),
            self._user("u7", followers=11, following=7, tweets=16, description="travel photos and food", verified=False),
            self._user("u8", followers=2, following=58, tweets=110, description="coupon click limited offer", verified=False),
            self._user("u9", followers=3, following=20, tweets=30, description="support node alpha", verified=False),
            self._user("u10", followers=4, following=25, tweets=35, description="support node beta", verified=False),
            {"id": "t1", "text": "great basketball game tonight"},
            {"id": "t2", "text": "free crypto reward click now"},
            {"id": "t3", "text": "love reading on rainy weekends"},
            {"id": "t4", "text": "massive giveaway act fast"},
            {"id": "t5", "text": "community event this afternoon"},
            {"id": "t6", "text": "buy followers instant traffic"},
            {"id": "t7", "text": "new restaurant review downtown"},
            {"id": "t8", "text": "promo link discount offer"},
        ]
        (data_dir / "node.json").write_text(json.dumps(nodes), encoding="utf-8")

    def _user(self, user_id: str, followers: int, following: int, tweets: int, description: str, verified: bool) -> dict[str, object]:
        return {
            "id": user_id,
            "created_at": "Mon Jan 01 00:00:00 +0000 2024",
            "description": description,
            "location": "Shanghai",
            "name": user_id.upper(),
            "protected": "False",
            "profile_image_url": "",
            "public_metrics": {
                "followers_count": followers,
                "following_count": following,
                "tweet_count": tweets,
                "listed_count": 1,
            },
            "url": "",
            "username": user_id,
            "verified": str(verified),
        }


if __name__ == "__main__":
    unittest.main()
