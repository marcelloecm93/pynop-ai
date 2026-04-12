"""EvalRunner — orchestrates evaluation runs."""

import asyncio
import io
import json
import os

from pynop.eval.types import EvalIssue, EvalResult
from pynop.types import EvalThreshold


class EvalRunner:
    """Run Garak and Giskard evaluations against a ``SafetyPipeline``.

    Both tools probe the *full* pipeline (input guards + LLM + output guards) so
    results reflect production behavior. Each run is recorded as a Langfuse trace
    when the pipeline has tracing enabled.

    Args:
        pipeline: The ``SafetyPipeline`` to evaluate.
        threshold: Optional explicit ``EvalThreshold`` that overrides both the
            pipeline's base threshold and any per-tool thresholds. Most callers
            should leave this unset and rely on per-tool config from YAML.
    """

    def __init__(self, pipeline, threshold: EvalThreshold | None = None):
        self._pipeline = pipeline
        self._explicit_threshold = threshold

    def _resolve_threshold(self, tool: str) -> EvalThreshold:
        """Resolve threshold for a tool. Explicit override wins, then per-tool, then base."""
        if self._explicit_threshold is not None:
            return self._explicit_threshold
        return self._pipeline.eval_threshold_for(tool)

    async def run_garak(self, probes: list[str]) -> EvalResult:
        """Run Garak probes against the pipeline.

        Args:
            probes: List of probe family names (e.g. ["dan", "promptinject"]).
        """
        if not probes:
            raise ValueError("eval.garak: probes list must not be empty")
        threshold = self._resolve_threshold("garak")
        result = None
        trace = self._pipeline.tracer.start_trace(prompt=f"garak eval: {probes}")
        try:
            result = await asyncio.to_thread(self._run_garak_sync, probes, trace, threshold)
        finally:
            self._pipeline.tracer.end_trace(trace, output=result.summary if result else None)
            self._pipeline.tracer.flush()
        return result

    async def run_giskard(self, detectors: list[str]) -> EvalResult:
        """Run Giskard detectors against the pipeline.

        Args:
            detectors: List of detector tags (e.g. ["hallucination", "prompt_injection"]).
        """
        if not detectors:
            raise ValueError("eval.giskard: detectors list must not be empty")
        threshold = self._resolve_threshold("giskard")
        result = None
        trace = self._pipeline.tracer.start_trace(prompt=f"giskard eval: {detectors}")
        try:
            result = await asyncio.to_thread(self._run_giskard_sync, detectors, trace, threshold)
        finally:
            self._pipeline.tracer.end_trace(trace, output=result.summary if result else None)
            self._pipeline.tracer.flush()
        return result

    def _run_garak_sync(self, probes: list[str], trace, threshold: EvalThreshold) -> EvalResult:
        from garak import _config, _plugins
        from garak.evaluators.base import Evaluator
        from garak.harnesses.base import Harness

        from pynop.eval.garak_adapter import PipelineGenerator

        if not _config.loaded:
            _config.load_base_config()

        generator = PipelineGenerator(self._pipeline)

        # Expand probe family names to probe instances
        all_probes = _plugins.enumerate_plugins("probes")
        probe_instances = []
        for family in probes:
            prefix = f"probes.{family}."
            for class_name, active in all_probes:
                if class_name.startswith(prefix) and active:
                    module_name, cls_name = class_name.rsplit(".", 1)
                    mod = _plugins.load_plugin(class_name, break_on_fail=False)
                    if mod:
                        probe_instances.append(mod)

        if not probe_instances:
            return EvalResult(
                summary="No matching probes found",
                issues=[],
                trace_id=trace.id if trace else None,
                tool="garak",
                threshold=threshold,
            )

        # Use default detectors from each probe
        detector_instances = []
        for p in probe_instances:
            for d_name in p.recommended_detector:
                det = _plugins.load_plugin(f"detectors.{d_name}", break_on_fail=False)
                if det and det not in detector_instances:
                    detector_instances.append(det)

        if not detector_instances:
            detector_instances = [_plugins.load_plugin("detectors.always.Fail")]

        # Capture results to in-memory buffers (no disk I/O)
        report_buffer = io.StringIO()
        _config.transient.reportfile = report_buffer
        _config.transient.report_filename = os.devnull
        _config.transient.hitlogfile = io.StringIO()

        evaluator = Evaluator()
        harness = Harness()
        harness.run(generator, probe_instances, detector_instances, evaluator)

        # Parse JSONL results
        report_buffer.seek(0)
        issues = []
        for line in report_buffer:
            line = line.strip()
            if not line:
                continue
            attempt = json.loads(line)
            detector_results = attempt.get("detector_results", {})
            for detector_name, scores in detector_results.items():
                failures = [s for s in scores if s > 0.5]
                if failures:
                    probe_classname = attempt.get("probe_classname", "unknown")
                    span = self._pipeline.tracer.start_span(trace, name=f"garak:{probe_classname}")
                    issue = EvalIssue(
                        check=probe_classname,
                        severity=threshold.garak_severity(probe_classname),
                        description=f"Detector {detector_name} flagged {len(failures)}/{len(scores)} outputs",
                        details={
                            "detector": detector_name,
                            "prompt": attempt.get("prompt", ""),
                            "scores": scores,
                        },
                    )
                    issues.append(issue)
                    self._pipeline.tracer.end_span(span, metadata={"issue": issue.description})

        summary = f"{len(issues)} issues found across {len(probe_instances)} probes"
        return EvalResult(
            summary=summary,
            issues=issues,
            trace_id=trace.id if trace else None,
            tool="garak",
            threshold=threshold,
        )

    def _run_giskard_sync(self, detectors: list[str], trace, threshold: EvalThreshold) -> EvalResult:
        import giskard

        from pynop.eval.giskard_adapter import create_pipeline_model

        model = create_pipeline_model(self._pipeline)
        report = giskard.scan(model, only=detectors, raise_exceptions=False)

        issues = []
        for gisk_issue in report.issues:
            span = self._pipeline.tracer.start_span(trace, name=f"giskard:{gisk_issue.group.name}")
            issue = EvalIssue(
                check=gisk_issue.group.name,
                severity=gisk_issue.level.name.lower(),
                description=gisk_issue.description_tpl,
                details=gisk_issue.meta,
            )
            issues.append(issue)
            self._pipeline.tracer.end_span(span, metadata={"issue": issue.description})

        summary = f"{len(issues)} issues found across {len(detectors)} detectors"
        return EvalResult(
            summary=summary,
            issues=issues,
            trace_id=trace.id if trace else None,
            tool="giskard",
            threshold=threshold,
        )
