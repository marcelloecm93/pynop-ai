"""Tests for NeMoConfigBuilder."""

import os

import pytest
import yaml

from pynop.guards.nemo_builder import KNOWN_RAILS, NeMoConfigBuilder


class TestAddRail:
    def test_add_known_built_in_rail(self):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        assert len(builder._rails) == 1

    def test_add_parameterized_rail(self):
        builder = NeMoConfigBuilder()
        builder.add_rail("topic_control", {"allowed": ["coding"]})
        assert len(builder._rails) == 1

    def test_unknown_rail_raises(self):
        builder = NeMoConfigBuilder()
        with pytest.raises(ValueError, match="unknown rail"):
            builder.add_rail("nonexistent")

    def test_parameterized_rail_without_params_raises(self):
        builder = NeMoConfigBuilder()
        with pytest.raises(ValueError, match="requires parameters"):
            builder.add_rail("topic_control")


class TestBuild:
    def test_builds_config_yml(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.build(str(tmp_path))

        config_path = tmp_path / "config.yml"
        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        assert "rails" in config
        assert "self check input" in config["rails"]["input"]["flows"]

    def test_jailbreak_adds_input_flow(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        assert config["rails"]["input"]["flows"] == ["self check input"]
        assert any(p["task"] == "self_check_input" for p in config["prompts"])

    def test_content_safety_adds_output_flow(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("content_safety")
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        assert config["rails"]["output"]["flows"] == ["self check output"]

    def test_pii_adds_output_flow(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("pii")
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        assert "self check output" in config["rails"]["output"]["flows"]

    def test_multi_rail_deduplicates_flows(self, tmp_path):
        """content_safety and pii both use 'self check output' — should appear once."""
        builder = NeMoConfigBuilder()
        builder.add_rail("content_safety")
        builder.add_rail("pii")
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        flows = config["rails"]["output"]["flows"]
        assert flows.count("self check output") == 1

    def test_topic_control_generates_co_file(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("topic_control", {"allowed": ["coding", "data science"], "denied": ["politics"]})
        builder.build(str(tmp_path))

        co_path = tmp_path / "topic_control.co"
        assert co_path.exists()
        content = co_path.read_text()
        assert "coding, data science" in content
        assert "politics" in content

    def test_topic_control_adds_check_topic_flow(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("topic_control", {"allowed": ["coding"]})
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        assert "check topic" in config["rails"]["input"]["flows"]

    def test_build_creates_output_dir(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        out = str(tmp_path / "nested" / "dir")
        builder.build(out)
        assert os.path.isfile(os.path.join(out, "config.yml"))

    def test_build_idempotent(self, tmp_path):
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.build(str(tmp_path))
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        assert config["rails"]["input"]["flows"] == ["self check input"]

    def test_models_key_is_empty_list(self, tmp_path):
        """Skeleton should have models: [] since LLM is shared via LLMRails(llm=...)."""
        builder = NeMoConfigBuilder()
        builder.add_rail("jailbreak")
        builder.build(str(tmp_path))

        config = yaml.safe_load((tmp_path / "config.yml").read_text())
        assert config["models"] == []
