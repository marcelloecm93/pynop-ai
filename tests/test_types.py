"""Tests for PipelineResult and GuardResult dataclasses."""

import pytest

from pynop.types import (
    DEFAULT_REASK_INSTRUCTION,
    EvalThreshold,
    GuardConfig,
    GuardResult,
    GuardSlot,
    PipelineResult,
    ReaskSignal,
)
from pynop.eval.types import EvalIssue


class TestGuardResult:
    def test_passing_result(self):
        result = GuardResult(passed=True, reason=None)
        assert result.passed is True
        assert result.reason is None

    def test_failing_result_with_reason(self):
        result = GuardResult(passed=False, reason="PII detected")
        assert result.passed is False
        assert result.reason == "PII detected"

    def test_modified_defaults_to_false(self):
        result = GuardResult(passed=True, reason=None)
        assert result.modified is False

    def test_modified_can_be_set_true(self):
        result = GuardResult(passed=True, reason=None, modified=True)
        assert result.modified is True


class TestPipelineResult:
    def test_successful_result(self):
        result = PipelineResult(
            output="Hello world",
            trace_id="trace-123",
            guarded=True,
        )
        assert result.output == "Hello world"
        assert result.trace_id == "trace-123"
        assert result.guarded is True

    def test_result_without_tracing(self):
        result = PipelineResult(output="Hello", trace_id=None, guarded=False)
        assert result.trace_id is None
        assert result.guarded is False


class TestGuardConfig:
    def test_defaults(self):
        cfg = GuardConfig()
        assert cfg.on_guard_fail == "reject"
        assert cfg.max_reask == 2
        assert cfg.reask_instruction == DEFAULT_REASK_INSTRUCTION

    def test_reask_config(self):
        cfg = GuardConfig(
            on_guard_fail="reask",
            max_reask=3,
            reask_instruction="Fix this: {reason}",
        )
        assert cfg.on_guard_fail == "reask"
        assert cfg.max_reask == 3
        assert "{reason}" in cfg.reask_instruction


class TestReaskSignal:
    def test_fields(self):
        signal = ReaskSignal(
            guard_index=1,
            reask_instruction="Try again: {reason}",
            max_reask=2,
            reason="toxic content",
        )
        assert signal.guard_index == 1
        assert signal.max_reask == 2
        assert signal.reason == "toxic content"


class TestEvalThreshold:
    def test_defaults(self):
        t = EvalThreshold()
        assert t.max_issues == 0
        assert t.ignore_severities == []
        assert t.garak_severities == {}

    def test_is_passed_with_max_issues(self):
        t = EvalThreshold(max_issues=2)
        issues = [
            EvalIssue(check="a", severity="major", description="x"),
            EvalIssue(check="b", severity="major", description="y"),
        ]
        assert t.is_passed(issues) is True
        issues.append(EvalIssue(check="c", severity="major", description="z"))
        assert t.is_passed(issues) is False

    def test_is_passed_with_ignore_severities(self):
        t = EvalThreshold(max_issues=0, ignore_severities=["minor"])
        issues = [
            EvalIssue(check="a", severity="minor", description="x"),
            EvalIssue(check="b", severity="minor", description="y"),
        ]
        assert t.is_passed(issues) is True
        issues.append(EvalIssue(check="c", severity="major", description="z"))
        assert t.is_passed(issues) is False

    def test_garak_severity_extracts_family_and_maps(self):
        t = EvalThreshold(garak_severities={"dan": "major", "glitch": "minor"})
        assert t.garak_severity("probes.dan.Dan_11_0") == "major"
        assert t.garak_severity("probes.glitch.Glitch_1") == "minor"

    def test_garak_severity_defaults_to_major(self):
        t = EvalThreshold()
        assert t.garak_severity("probes.unknown.Foo") == "major"

    def test_invalid_garak_severity_raises(self):
        with pytest.raises(ValueError, match="is not valid"):
            EvalThreshold(garak_severities={"dan": "catastrophic"})


class TestGuardSlotWithConfigs:
    def test_guard_configs_default_empty(self):
        slot = GuardSlot()
        assert slot.guard_configs == []

    def test_guard_configs_stored(self):
        configs = [GuardConfig(on_guard_fail="reask"), GuardConfig(on_guard_fail="reject")]
        slot = GuardSlot(guard_configs=configs)
        assert len(slot.guard_configs) == 2
        assert slot.guard_configs[0].on_guard_fail == "reask"
