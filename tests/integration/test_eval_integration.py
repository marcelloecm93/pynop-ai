"""Integration tests for EvalRunner against real Garak and Giskard.

Requires PYNOP_INTEGRATION=1 and real API keys set in environment.
"""

import os

import pytest

from pynop import EvalThreshold, SafetyPipeline
from pynop.eval import EvalRunner

pytestmark = pytest.mark.skipif(
    os.environ.get("PYNOP_INTEGRATION") != "1",
    reason="Integration tests require PYNOP_INTEGRATION=1",
)


@pytest.fixture
def pipeline(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: false
""")
    return SafetyPipeline.from_yaml(str(path))


class TestGarakIntegration:
    @pytest.mark.asyncio
    async def test_run_garak_small_probe(self, pipeline):
        """Run a small Garak probe set and verify we get a structured result."""
        runner = EvalRunner(pipeline)
        result = await runner.run_garak(probes=["dan"])

        assert result.tool == "garak"
        assert isinstance(result.summary, str)
        assert isinstance(result.issues, list)
        # passed is a bool regardless of outcome
        assert isinstance(result.passed, bool)


class TestGiskardIntegration:
    @pytest.mark.asyncio
    async def test_run_giskard_small_detector(self, pipeline):
        """Run a Giskard detector and verify we get a structured result."""
        runner = EvalRunner(pipeline)
        result = await runner.run_giskard(detectors=["prompt_injection"])

        assert result.tool == "giskard"
        assert isinstance(result.summary, str)
        assert isinstance(result.issues, list)
        assert isinstance(result.passed, bool)


class TestPerToolThresholdIntegration:
    @pytest.mark.asyncio
    async def test_different_thresholds_produce_divergent_pass_fail(self, tmp_path):
        """Per-tool thresholds produce divergent pass/fail on real eval runs.

        Garak is configured strict (0 issues allowed); Giskard is configured lenient
        (1000 issues + all severities ignored). When both runners execute against the
        same pipeline, the resolved EvalResult.passed flags must reflect the per-tool
        thresholds — not a shared base threshold.
        """
        path = tmp_path / "config.yaml"
        path.write_text("""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: false
eval:
  max_issues: 0
  garak:
    max_issues: 0
  giskard:
    max_issues: 1000
    ignore_severities: [major, medium, minor]
""")
        pipeline = SafetyPipeline.from_yaml(str(path))
        runner = EvalRunner(pipeline)

        garak_result = await runner.run_garak(probes=["dan"])
        giskard_result = await runner.run_giskard(detectors=["prompt_injection"])

        # Each result carries the per-tool threshold, not the base.
        assert garak_result.threshold.max_issues == 0
        assert giskard_result.threshold.max_issues == 1000
        assert giskard_result.threshold.ignore_severities == ["major", "medium", "minor"]

        # Giskard's threshold ignores every severity and allows 1000 issues →
        # it must pass regardless of how many issues the real run produced.
        assert giskard_result.passed is True

        # Garak's strict threshold means pass iff zero non-ignored issues.
        expected_garak_passed = len(garak_result.issues) == 0
        assert garak_result.passed is expected_garak_passed

        # Core claim: if Garak surfaces any issue, the two tools must diverge,
        # because only Garak's threshold rejects. This is the pass/fail split
        # the plan requires proving end-to-end.
        if garak_result.issues:
            assert garak_result.passed is False
            assert giskard_result.passed is True
