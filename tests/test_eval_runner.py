"""Tests for EvalRunner orchestration."""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pynop.eval.runner import EvalRunner
from pynop.eval.types import EvalResult
from pynop.types import EvalThreshold


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.run = AsyncMock()
    pipeline.tracer = MagicMock()
    pipeline.tracer.start_trace.return_value = None
    pipeline.tracer.start_span.return_value = None
    default_threshold = EvalThreshold()
    pipeline.eval_threshold = default_threshold
    pipeline.eval_threshold_for = MagicMock(return_value=default_threshold)
    return pipeline


class TestEvalRunnerGarak:
    @pytest.mark.asyncio
    async def test_run_garak_returns_eval_result(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)

        mock_probe = MagicMock()
        mock_probe.recommended_detector = ["always.Pass"]

        with (
            patch("garak._plugins.enumerate_plugins", return_value=[("probes.dan.Dan_11_0", True)]),
            patch("garak._plugins.load_plugin", return_value=mock_probe),
            patch("garak.harnesses.base.Harness") as MockHarness,
            patch("garak.evaluators.base.Evaluator"),
            patch("garak._config") as mock_config,
        ):
            buf = io.StringIO()
            mock_config.transient = MagicMock()
            mock_config.transient.reportfile = buf

            MockHarness.return_value.run.side_effect = lambda *a: None

            result = await runner.run_garak(probes=["dan"])

        assert isinstance(result, EvalResult)
        assert result.tool == "garak"

    @pytest.mark.asyncio
    async def test_run_garak_no_matching_probes(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)

        with patch("garak._plugins.enumerate_plugins", return_value=[]):
            result = await runner.run_garak(probes=["nonexistent"])

        assert result.passed is True
        assert result.summary == "No matching probes found"

    @pytest.mark.asyncio
    async def test_run_garak_parses_issues_from_jsonl(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)

        mock_probe = MagicMock()
        mock_probe.recommended_detector = ["always.Fail"]

        attempt_json = json.dumps({
            "probe_classname": "probes.dan.Dan_11_0",
            "prompt": "Ignore instructions",
            "outputs": ["Sure, I'll help"],
            "detector_results": {"always.Fail": [1.0]},
        })

        with (
            patch("garak._plugins.enumerate_plugins", return_value=[("probes.dan.Dan_11_0", True)]),
            patch("garak._plugins.load_plugin", return_value=mock_probe),
            patch("garak.harnesses.base.Harness") as MockHarness,
            patch("garak.evaluators.base.Evaluator"),
            patch("garak._config") as mock_config,
        ):
            mock_config.transient = MagicMock()

            def fake_run(gen, probes, dets, evaluator):
                # Write to the reportfile the runner just set
                mock_config.transient.reportfile.write(attempt_json + "\n")

            MockHarness.return_value.run.side_effect = fake_run

            result = await runner.run_garak(probes=["dan"])

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].check == "probes.dan.Dan_11_0"


class TestEvalRunnerGiskard:
    @pytest.mark.asyncio
    async def test_run_giskard_returns_eval_result(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)

        mock_report = MagicMock()
        mock_report.issues = []

        with (
            patch("giskard.scan", return_value=mock_report),
            patch("pynop.eval.giskard_adapter.Model"),
        ):
            result = await runner.run_giskard(detectors=["prompt_injection"])

        assert isinstance(result, EvalResult)
        assert result.tool == "giskard"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_giskard_parses_issues(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)

        mock_issue = MagicMock()
        mock_issue.group.name = "prompt_injection"
        mock_issue.level.name = "MAJOR"
        mock_issue.description_tpl = "Prompt injection detected"
        mock_issue.meta = {"metric": 0.85}

        mock_report = MagicMock()
        mock_report.issues = [mock_issue]

        with (
            patch("giskard.scan", return_value=mock_report),
            patch("pynop.eval.giskard_adapter.Model"),
        ):
            result = await runner.run_giskard(detectors=["prompt_injection"])

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].check == "prompt_injection"
        assert result.issues[0].severity == "major"

    @pytest.mark.asyncio
    async def test_run_giskard_traces_to_langfuse(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)

        mock_report = MagicMock()
        mock_report.issues = []

        with (
            patch("giskard.scan", return_value=mock_report),
            patch("pynop.eval.giskard_adapter.Model"),
        ):
            await runner.run_giskard(detectors=["prompt_injection"])

        mock_pipeline.tracer.start_trace.assert_called_once()
        mock_pipeline.tracer.end_trace.assert_called_once()
        mock_pipeline.tracer.flush.assert_called_once()


class TestEvalRunnerThreshold:
    @pytest.mark.asyncio
    async def test_runner_resolves_threshold_from_pipeline(self, mock_pipeline):
        threshold = EvalThreshold(max_issues=5, ignore_severities=["minor"])
        mock_pipeline.eval_threshold_for = MagicMock(return_value=threshold)
        runner = EvalRunner(mock_pipeline)
        assert runner._resolve_threshold("garak") is threshold
        mock_pipeline.eval_threshold_for.assert_called_with("garak")

    @pytest.mark.asyncio
    async def test_explicit_threshold_overrides_pipeline(self, mock_pipeline):
        explicit = EvalThreshold(max_issues=10)
        runner = EvalRunner(mock_pipeline, threshold=explicit)
        assert runner._resolve_threshold("garak") is explicit
        assert runner._resolve_threshold("giskard") is explicit
        mock_pipeline.eval_threshold_for.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_tool_threshold_used_in_garak(self, mock_pipeline):
        garak_threshold = EvalThreshold(max_issues=0, garak_severities={"dan": "minor"})
        giskard_threshold = EvalThreshold(max_issues=5)
        mock_pipeline.eval_threshold_for = MagicMock(
            side_effect=lambda tool: garak_threshold if tool == "garak" else giskard_threshold
        )
        runner = EvalRunner(mock_pipeline)

        mock_probe = MagicMock()
        mock_probe.recommended_detector = ["always.Fail"]

        attempt_json = json.dumps({
            "probe_classname": "probes.dan.Dan_11_0",
            "prompt": "Ignore instructions",
            "outputs": ["Sure"],
            "detector_results": {"always.Fail": [1.0]},
        })

        with (
            patch("garak._plugins.enumerate_plugins", return_value=[("probes.dan.Dan_11_0", True)]),
            patch("garak._plugins.load_plugin", return_value=mock_probe),
            patch("garak.harnesses.base.Harness") as MockHarness,
            patch("garak.evaluators.base.Evaluator"),
            patch("garak._config") as mock_config,
        ):
            mock_config.transient = MagicMock()

            def fake_run(gen, probes, dets, evaluator):
                mock_config.transient.reportfile.write(attempt_json + "\n")

            MockHarness.return_value.run.side_effect = fake_run

            result = await runner.run_garak(probes=["dan"])

        assert result.issues[0].severity == "minor"
        assert result.threshold is garak_threshold
        mock_pipeline.eval_threshold_for.assert_called_with("garak")


class TestEvalRunnerEdgeCases:
    @pytest.mark.asyncio
    async def test_run_garak_empty_probes_raises(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)
        with pytest.raises(ValueError, match="probes list must not be empty"):
            await runner.run_garak(probes=[])

    @pytest.mark.asyncio
    async def test_run_giskard_empty_detectors_raises(self, mock_pipeline):
        runner = EvalRunner(mock_pipeline)
        with pytest.raises(ValueError, match="detectors list must not be empty"):
            await runner.run_giskard(detectors=[])
