"""Tests for Garak PipelineGenerator adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pynop.eval.garak_adapter import PipelineGenerator
from pynop.exceptions import GuardRejection
from pynop.types import PipelineResult


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.run = AsyncMock()
    return pipeline


class TestPipelineGenerator:
    def test_creation(self, mock_pipeline):
        gen = PipelineGenerator(mock_pipeline)
        assert gen.generator_family_name == "pynop"
        assert gen.name == "SafetyPipeline"

    def test_is_garak_generator(self, mock_pipeline):
        from garak.generators.base import Generator

        gen = PipelineGenerator(mock_pipeline)
        assert isinstance(gen, Generator)

    def test_call_model_returns_pipeline_output(self, mock_pipeline):
        mock_pipeline.run.return_value = PipelineResult(
            output="Safe response", trace_id=None, guarded=True
        )
        gen = PipelineGenerator(mock_pipeline)
        result = gen._call_model("Hello")
        assert result == ["Safe response"]
        mock_pipeline.run.assert_called_once_with("Hello")

    def test_call_model_catches_guard_rejection(self, mock_pipeline):
        mock_pipeline.run.side_effect = GuardRejection("Invalid input detected")
        gen = PipelineGenerator(mock_pipeline)
        result = gen._call_model("Jailbreak attempt")
        assert result == ["[REJECTED] Invalid input detected"]

    def test_call_model_returns_list_of_one(self, mock_pipeline):
        mock_pipeline.run.return_value = PipelineResult(
            output="Response", trace_id=None, guarded=False
        )
        gen = PipelineGenerator(mock_pipeline)
        result = gen._call_model("Test", generations_this_call=1)
        assert isinstance(result, list)
        assert len(result) == 1
