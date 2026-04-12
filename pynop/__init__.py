"""pynop — LLM safety pipeline.

Public API surface (v1.0 contract):

  Core:        SafetyPipeline, PipelineResult, GuardRejection
  Guards:      Guard (protocol), GuardResult, NeMoConfigBuilder
  Eval:        EvalThreshold (via pynop), EvalRunner / EvalResult / EvalIssue (via pynop.eval)
  Benchmark:   LatencyBenchmark, BenchmarkReport, SpanStats
"""

from pynop.benchmark import BenchmarkReport, LatencyBenchmark, SpanStats
from pynop.exceptions import GuardRejection
from pynop.guards.base import Guard
from pynop.guards.nemo_builder import NeMoConfigBuilder
from pynop.pipeline import SafetyPipeline
from pynop.types import EvalThreshold, GuardResult, PipelineResult

__all__ = [
    "SafetyPipeline",
    "PipelineResult",
    "GuardRejection",
    "Guard",
    "GuardResult",
    "NeMoConfigBuilder",
    "EvalThreshold",
    "LatencyBenchmark",
    "BenchmarkReport",
    "SpanStats",
]
