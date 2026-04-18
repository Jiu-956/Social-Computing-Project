from __future__ import annotations

from dafbot.data.build_attr_features import build_attr_features
from dafbot.data.build_dynamic_snapshots import build_dynamic_snapshots
from dafbot.data.build_text_inputs import build_text_inputs_and_features
from dafbot.data.preprocess_users import build_user_table


def run_preprocess_pipeline(config: dict[str, object]):
    user_table = build_user_table(config)
    build_attr_features(config, user_table)
    build_text_inputs_and_features(config, user_table)
    build_dynamic_snapshots(config, user_table)
    return user_table
