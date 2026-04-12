"""Tests for LatencyBenchmark."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pynop.benchmark import LatencyBenchmark, SpanStats, _compute_stats, _percentile


class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 99) == 5.0

    def test_even_distribution(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == 3.0

    def test_p95_interpolation(self):
        data = list(range(1, 101))
        p95 = _percentile([float(x) for x in data], 95)
        assert 95.0 <= p95 <= 96.0


class TestComputeStats:
    def test_basic_stats(self):
        stats = _compute_stats("guard", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.name == "guard"
        assert stats.count == 5
        assert stats.p50 == 3.0


class TestLatencyBenchmark:
    def _make_pipeline(self, trace_ids, latencies_per_trace):
        """Create a mock pipeline that returns trace IDs and has a Langfuse client."""
        pipeline = MagicMock()
        pipeline.tracer = MagicMock()

        call_count = 0
        async def mock_run(prompt):
            nonlocal call_count
            result = MagicMock()
            result.trace_id = trace_ids[call_count] if call_count < len(trace_ids) else None
            call_count += 1
            return result

        pipeline.run = mock_run

        # Mock the Langfuse client
        client = MagicMock()
        pipeline.tracer.client = client

        def mock_get_trace(trace_id):
            trace = MagicMock()
            trace_data = latencies_per_trace.get(trace_id, {})
            trace.latency = trace_data.get("total")
            observations = []
            for span_name, span_latency in trace_data.get("spans", {}).items():
                obs = MagicMock()
                obs.name = span_name
                obs.latency = span_latency
                observations.append(obs)
            trace.observations = observations
            return trace

        client.trace.get.side_effect = mock_get_trace
        return pipeline

    @pytest.mark.asyncio
    async def test_benchmark_produces_report(self):
        pipeline_a = self._make_pipeline(
            trace_ids=["t1", "t2"],
            latencies_per_trace={
                "t1": {"total": 0.5, "spans": {"input_guard": 0.1, "llm_call": 0.3, "output_guard": 0.1}},
                "t2": {"total": 0.6, "spans": {"input_guard": 0.12, "llm_call": 0.35, "output_guard": 0.13}},
            },
        )
        pipeline_b = self._make_pipeline(
            trace_ids=["t3", "t4"],
            latencies_per_trace={
                "t3": {"total": 0.8, "spans": {"input_guard": 0.2, "llm_call": 0.4, "output_guard": 0.2}},
                "t4": {"total": 0.9, "spans": {"input_guard": 0.22, "llm_call": 0.45, "output_guard": 0.23}},
            },
        )

        benchmark = LatencyBenchmark(pipeline_a, pipeline_b, label_a="fast", label_b="strict")
        report = await benchmark.run(["prompt1", "prompt2"])

        assert report.label_a == "fast"
        assert report.label_b == "strict"
        assert len(report.stats_a) == 3  # input_guard, llm_call, output_guard
        assert len(report.stats_b) == 3
        assert report.total_a.count == 2
        assert report.total_b.count == 2
        # Strict config should have higher latency
        assert report.total_b.p50 > report.total_a.p50

    @pytest.mark.asyncio
    async def test_benchmark_empty_prompts_raises(self):
        pipeline_a = MagicMock()
        pipeline_b = MagicMock()
        benchmark = LatencyBenchmark(pipeline_a, pipeline_b)
        with pytest.raises(ValueError, match="prompts list must not be empty"):
            await benchmark.run([])

    @pytest.mark.asyncio
    async def test_benchmark_no_langfuse_raises(self):
        pipeline_a = MagicMock()
        pipeline_a.tracer.client = None
        pipeline_a.run = AsyncMock(return_value=MagicMock(trace_id="t1"))

        pipeline_b = MagicMock()
        pipeline_b.tracer.client = MagicMock()
        pipeline_b.run = AsyncMock(return_value=MagicMock(trace_id="t2"))

        benchmark = LatencyBenchmark(pipeline_a, pipeline_b)
        with pytest.raises(ValueError, match="Langfuse enabled"):
            await benchmark.run(["prompt1"])

    @pytest.mark.asyncio
    async def test_benchmark_spans_sorted_by_name(self):
        pipeline = self._make_pipeline(
            trace_ids=["t1"],
            latencies_per_trace={
                "t1": {"total": 1.0, "spans": {"z_guard": 0.5, "a_guard": 0.3}},
            },
        )
        benchmark = LatencyBenchmark(pipeline, pipeline)
        report = await benchmark.run(["prompt1"])

        span_names = [s.name for s in report.stats_a]
        assert span_names == ["a_guard", "z_guard"]
