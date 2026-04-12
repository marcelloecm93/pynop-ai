"""Integration tests — verify NeMo accepts generated configs via RailsConfig.from_path()."""

import pytest
from nemoguardrails import RailsConfig

from pynop.guards.nemo_builder import NeMoConfigBuilder


class TestRailsConfigAcceptsGenerated:
    def test_jailbreak_config(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.build(str(tmp_path))

        config = RailsConfig.from_path(str(tmp_path))
        assert config is not None

    def test_content_safety_config(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("content_safety")
        builder.build(str(tmp_path))

        config = RailsConfig.from_path(str(tmp_path))
        assert config is not None

    def test_pii_config(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("pii")
        builder.build(str(tmp_path))

        config = RailsConfig.from_path(str(tmp_path))
        assert config is not None

    def test_topic_control_config(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("topic_control", {"allowed": ["coding", "data science"], "denied": ["politics"]})
        builder.build(str(tmp_path))

        config = RailsConfig.from_path(str(tmp_path))
        assert config is not None

    def test_multi_rail_config(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.add_rail("content_safety")
        builder.add_rail("pii")
        builder.build(str(tmp_path))

        config = RailsConfig.from_path(str(tmp_path))
        assert config is not None

    def test_all_rails_combined(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.add_rail("content_safety")
        builder.add_rail("pii")
        builder.add_rail("topic_control", {"allowed": ["coding"], "denied": ["violence"]})
        builder.build(str(tmp_path))

        config = RailsConfig.from_path(str(tmp_path))
        assert config is not None
