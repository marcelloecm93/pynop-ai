"""Tests for Giskard PipelineModel adapter."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from giskard.models.base.model import BaseModel

from pynop.eval.giskard_adapter import create_pipeline_model
from pynop.exceptions import GuardRejection
from pynop.types import PipelineResult


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.run = AsyncMock()
    return pipeline


class TestCreatePipelineModel:
    def test_returns_giskard_model(self, mock_pipeline):
        model = create_pipeline_model(mock_pipeline)
        assert isinstance(model, BaseModel)

    def test_name(self, mock_pipeline):
        model = create_pipeline_model(mock_pipeline)
        assert model.name == "pynop-SafetyPipeline"

    def test_feature_names(self, mock_pipeline):
        model = create_pipeline_model(mock_pipeline)
        assert model.feature_names == ["question"]

    def test_predict_calls_pipeline(self, mock_pipeline):
        mock_pipeline.run.return_value = PipelineResult(
            output="Safe response", trace_id=None, guarded=True
        )
        model = create_pipeline_model(mock_pipeline)
        df = pd.DataFrame({"question": ["Hello", "World"]})
        result = model.predict_df(df)
        assert isinstance(result, np.ndarray)
        assert list(result) == ["Safe response", "Safe response"]
        assert mock_pipeline.run.call_count == 2

    def test_predict_catches_guard_rejection(self, mock_pipeline):
        mock_pipeline.run.side_effect = GuardRejection("Invalid input detected")
        model = create_pipeline_model(mock_pipeline)
        df = pd.DataFrame({"question": ["Bad input"]})
        result = model.predict_df(df)
        assert result[0] == "[REJECTED] Invalid input detected"
