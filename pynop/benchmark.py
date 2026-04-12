"""Guard latency comparison benchmark."""

from dataclasses import dataclass, field

from langfuse import Langfuse


@dataclass
class SpanStats:
    """Latency statistics for one span name across a benchmark run.

    Attributes:
        name: Span name (e.g. ``"llm_call"``, ``"input_guard"``, or ``"total"``).
        count: Number of samples observed.
        p50: 50th percentile latency in seconds.
        p95: 95th percentile latency in seconds.
        p99: 99th percentile latency in seconds.
        values: Raw latency samples (excluded from ``repr``).
    """

    name: str
    count: int
    p50: float
    p95: float
    p99: float
    values: list[float] = field(repr=False)


@dataclass
class BenchmarkReport:
    """Side-by-side latency comparison of two pipeline configurations.

    Attributes:
        label_a: Label for the first pipeline (defaults to ``"A"``).
        label_b: Label for the second pipeline (defaults to ``"B"``).
        stats_a: Per-span latency stats for pipeline A.
        stats_b: Per-span latency stats for pipeline B.
        total_a: Aggregate (full pipeline) latency stats for A.
        total_b: Aggregate (full pipeline) latency stats for B.
    """

    label_a: str
    label_b: str
    stats_a: list[SpanStats]
    stats_b: list[SpanStats]
    total_a: SpanStats
    total_b: SpanStats


def _percentile(data: list[float], pct: float) -> float:
    """Compute a percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _compute_stats(name: str, latencies: list[float]) -> SpanStats:
    """Compute p50/p95/p99 for a list of latencies."""
    return SpanStats(
        name=name,
        count=len(latencies),
        p50=_percentile(latencies, 50),
        p95=_percentile(latencies, 95),
        p99=_percentile(latencies, 99),
        values=latencies,
    )


def _extract_span_latencies(trace) -> dict[str, list[float]]:
    """Extract per-span-name latencies from a Langfuse trace."""
    latencies: dict[str, list[float]] = {}
    for obs in trace.observations or []:
        if obs.latency is not None and obs.name:
            latencies.setdefault(obs.name, []).append(obs.latency)
    return latencies


class LatencyBenchmark:
    """Compare per-span latency between two pipeline configurations.

    Runs the same prompt set through both pipelines, then fetches each trace from
    Langfuse and computes per-span and total p50/p95/p99 latencies. Both pipelines
    must have Langfuse tracing enabled — the benchmark reads span timings from
    Langfuse rather than instrumenting timers directly.

    Args:
        pipeline_a: First ``SafetyPipeline`` (typically the baseline).
        pipeline_b: Second ``SafetyPipeline`` (the candidate to compare).
        label_a: Friendly label for pipeline A in the report.
        label_b: Friendly label for pipeline B in the report.
    """

    def __init__(self, pipeline_a, pipeline_b, label_a: str = "A", label_b: str = "B"):
        self._pipeline_a = pipeline_a
        self._pipeline_b = pipeline_b
        self._label_a = label_a
        self._label_b = label_b

    async def run(self, prompts: list[str]) -> BenchmarkReport:
        """Run all prompts through both pipelines and produce a ``BenchmarkReport``.

        Args:
            prompts: Non-empty list of prompts to run through both pipelines.

        Raises:
            ValueError: If ``prompts`` is empty, or if either pipeline's tracer
                does not have a Langfuse client (tracing must be enabled).
        """
        if not prompts:
            raise ValueError("benchmark: prompts list must not be empty")

        trace_ids_a = await self._run_pipeline(self._pipeline_a, prompts)
        trace_ids_b = await self._run_pipeline(self._pipeline_b, prompts)

        client_a = self._get_langfuse_client(self._pipeline_a)
        client_b = self._get_langfuse_client(self._pipeline_b)

        spans_a, total_a = self._collect_stats(client_a, trace_ids_a)
        spans_b, total_b = self._collect_stats(client_b, trace_ids_b)

        return BenchmarkReport(
            label_a=self._label_a,
            label_b=self._label_b,
            stats_a=spans_a,
            stats_b=spans_b,
            total_a=total_a,
            total_b=total_b,
        )

    async def _run_pipeline(self, pipeline, prompts: list[str]) -> list[str]:
        """Run all prompts and collect trace IDs."""
        trace_ids = []
        for prompt in prompts:
            result = await pipeline.run(prompt)
            if result.trace_id:
                trace_ids.append(result.trace_id)
        pipeline.tracer.flush()
        return trace_ids

    def _get_langfuse_client(self, pipeline) -> Langfuse:
        """Get the Langfuse client from a pipeline's tracer."""
        client = pipeline.tracer.client
        if client is None:
            raise ValueError("benchmark: pipeline tracer must have Langfuse enabled")
        return client

    def _collect_stats(
        self, client: Langfuse, trace_ids: list[str]
    ) -> tuple[list[SpanStats], SpanStats]:
        """Fetch traces and compute per-span and total stats."""
        all_span_latencies: dict[str, list[float]] = {}
        total_latencies: list[float] = []

        for trace_id in trace_ids:
            trace = client.trace.get(trace_id)
            if trace.latency is not None:
                total_latencies.append(trace.latency)
            for name, values in _extract_span_latencies(trace).items():
                all_span_latencies.setdefault(name, []).extend(values)

        spans = [
            _compute_stats(name, latencies)
            for name, latencies in sorted(all_span_latencies.items())
        ]
        return spans, _compute_stats("total", total_latencies)
