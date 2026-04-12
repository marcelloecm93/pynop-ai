"""Pipeline data types."""

from dataclasses import dataclass, field
from typing import Literal

DEFAULT_REASK_INSTRUCTION = "Your previous response was rejected: {reason}. Please try again."


@dataclass
class GuardResult:
    """Result of a single guard validation.

    Attributes:
        passed: ``True`` if the guard accepted the text, ``False`` if it rejected.
        reason: Human-readable rejection reason. ``None`` when ``passed`` is ``True``.
        modified: Reserved for future use — currently always ``False``.
    """

    passed: bool
    reason: str | None
    modified: bool = False


@dataclass
class GuardConfig:
    """Per-guard failure strategy configuration."""

    on_guard_fail: Literal["reject", "return_canned", "include_reason", "reask"] = "reject"
    max_reask: int = 2
    reask_instruction: str = DEFAULT_REASK_INSTRUCTION


@dataclass
class GuardSlot:
    """Configuration for an input or output guard slot."""

    guards: list = field(default_factory=list)
    guard_configs: list[GuardConfig] = field(default_factory=list)
    on_guard_fail: Literal["reject", "return_canned", "include_reason", "reask"] = "reject"
    on_guard_error: Literal["reject", "pass"] = "reject"
    canned_response: str | None = None


@dataclass
class ReaskSignal:
    """Returned by _run_guards when a reask-strategy guard fails."""

    guard_index: int
    reask_instruction: str
    max_reask: int
    reason: str | None


VALID_SEVERITIES = {"major", "medium", "minor"}


@dataclass
class EvalThreshold:
    """Configurable pass/fail threshold for evaluation results.

    Used by ``EvalResult.passed`` to decide whether an eval run is acceptable.
    The default is zero-tolerance — any issue fails.

    Attributes:
        max_issues: Maximum number of issues allowed before failing (after
            severity filtering). Default ``0``.
        ignore_severities: Severity levels to exclude from the issue count.
            Allowed values: ``"major"``, ``"medium"``, ``"minor"``.
        garak_severities: Map of Garak probe family names to severity levels
            (e.g. ``{"dan": "major", "glitch": "minor"}``). Probes not listed
            default to ``"major"``.

    Raises:
        ValueError: If ``garak_severities`` contains an invalid severity level.
    """

    max_issues: int = 0
    ignore_severities: list[str] = field(default_factory=list)
    garak_severities: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for probe, sev in self.garak_severities.items():
            if sev not in VALID_SEVERITIES:
                raise ValueError(
                    f"eval.garak_severities: '{sev}' for probe '{probe}' is not valid. "
                    f"Must be one of {VALID_SEVERITIES}"
                )

    def is_passed(self, issues: list) -> bool:
        """Check if issues pass the threshold after severity filtering."""
        remaining = [i for i in issues if i.severity not in self.ignore_severities]
        return len(remaining) <= self.max_issues

    def garak_severity(self, probe_classname: str) -> str:
        """Map a Garak probe classname to a severity level.

        Extracts the probe family (e.g. "probes.dan.Dan_11_0" -> "dan")
        and looks it up in garak_severities. Defaults to "major".
        """
        parts = probe_classname.split(".")
        family = parts[1] if len(parts) >= 2 else probe_classname
        return self.garak_severities.get(family, "major")


@dataclass
class PipelineResult:
    """Result of a successful ``SafetyPipeline.run`` call.

    Attributes:
        output: The LLM response text (or the configured ``canned_response`` when a
            guard rejected with the ``return_canned`` strategy).
        trace_id: Langfuse trace ID for this run, or ``None`` if tracing was disabled.
        guarded: ``True`` if at least one guard ran on the input or output slot.
    """

    output: str
    trace_id: str | None
    guarded: bool
