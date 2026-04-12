"""Evaluation data types."""

from dataclasses import dataclass, field

from pynop.types import EvalThreshold


@dataclass
class EvalIssue:
    """A single issue detected by Garak or Giskard during an evaluation run.

    Attributes:
        check: Probe or detector name (e.g. ``"probes.dan.Dan_11_0"``,
            ``"prompt_injection"``).
        severity: One of ``"major"``, ``"medium"``, ``"minor"``. Garak issues
            are mapped via ``EvalThreshold.garak_severities``; Giskard issues
            inherit Giskard's own severity classification.
        description: Human-readable summary of what was detected.
        details: Tool-specific metadata (prompt, scores, group, etc.).
    """

    check: str
    severity: str
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of a single ``EvalRunner.run_garak`` or ``run_giskard`` call.

    Attributes:
        summary: One-line summary string (e.g. ``"3 issues found across 5 probes"``).
        issues: List of ``EvalIssue`` records for everything the tool flagged.
        trace_id: Langfuse trace ID for this eval run, or ``None`` if tracing is off.
        tool: ``"garak"`` or ``"giskard"`` — identifies which tool produced the result.
        threshold: ``EvalThreshold`` used to compute ``passed``. Set automatically
            from the pipeline's per-tool config.
    """

    summary: str
    issues: list[EvalIssue]
    trace_id: str | None
    tool: str
    threshold: EvalThreshold = field(default_factory=EvalThreshold)

    @property
    def passed(self) -> bool:
        """``True`` if the issue count (after severity filtering) is within ``threshold.max_issues``."""
        return self.threshold.is_passed(self.issues)
