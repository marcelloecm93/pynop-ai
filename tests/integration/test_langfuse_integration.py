"""Integration tests for Langfuse tracing.

Requires PYNOP_INTEGRATION=1 and real Langfuse + OpenAI keys.
"""

import os
import time

import pytest

from pynop import LatencyBenchmark, SafetyPipeline

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PYNOP_INTEGRATION") != "1",
        reason="Integration tests require PYNOP_INTEGRATION=1",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY",
    ),
    pytest.mark.skipif(
        not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")),
        reason="Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
    ),
]


@pytest.fixture
def traced_pipeline(tmp_path):
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
  enabled: true
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
""")
    return SafetyPipeline.from_yaml(str(path))


class TestLangfuseIntegration:
    @pytest.mark.asyncio
    async def test_trace_created_and_readable(self, traced_pipeline):
        """A pipeline run creates a trace readable from the Langfuse API."""
        result = await traced_pipeline.run("Say hello in one word.")
        traced_pipeline.tracer.flush()

        assert result.trace_id is not None

        # Allow Langfuse to ingest the trace
        time.sleep(2)

        client = traced_pipeline.tracer.client
        trace = client.trace.get(result.trace_id)

        assert trace.id == result.trace_id
        assert trace.observations is not None
        assert len(trace.observations) > 0

        # Should have at least an llm_call span
        span_names = [obs.name for obs in trace.observations]
        assert "llm_call" in span_names


class TestLatencyBenchmarkIntegration:
    @pytest.mark.asyncio
    async def test_benchmark_produces_report(self, traced_pipeline, tmp_path):
        """LatencyBenchmark produces a report with real Langfuse data."""
        # Build a second pipeline with the same config
        path = tmp_path / "config_b.yaml"
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
  enabled: true
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
""")
        pipeline_b = SafetyPipeline.from_yaml(str(path))

        benchmark = LatencyBenchmark(traced_pipeline, pipeline_b, label_a="A", label_b="B")
        report = await benchmark.run(["Say hello in one word."])

        # Allow Langfuse ingestion
        time.sleep(2)

        assert report.label_a == "A"
        assert report.label_b == "B"
        assert report.total_a.count >= 1
        assert report.total_b.count >= 1
        assert report.total_a.p50 > 0
        assert report.total_b.p50 > 0
