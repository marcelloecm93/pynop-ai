"""Tests for EvalResult and EvalIssue dataclasses."""

from pynop.eval.types import EvalIssue, EvalResult
from pynop.types import EvalThreshold


class TestEvalIssue:
    def test_construction(self):
        issue = EvalIssue(
            check="dan", severity="major", description="Jailbreak detected"
        )
        assert issue.check == "dan"
        assert issue.severity == "major"
        assert issue.description == "Jailbreak detected"
        assert issue.details == {}

    def test_construction_with_details(self):
        issue = EvalIssue(
            check="dan",
            severity="major",
            description="Jailbreak detected",
            details={"probe": "dan.Dan_11_0"},
        )
        assert issue.details == {"probe": "dan.Dan_11_0"}


class TestEvalResult:
    def test_passed_when_no_issues(self):
        result = EvalResult(
            summary="0 issues found", issues=[], trace_id=None, tool="garak"
        )
        assert result.passed is True

    def test_failed_when_issues_present(self):
        issue = EvalIssue(
            check="dan", severity="major", description="Jailbreak detected"
        )
        result = EvalResult(
            summary="1 issue found",
            issues=[issue],
            trace_id="trace-123",
            tool="garak",
        )
        assert result.passed is False
        assert result.trace_id == "trace-123"
        assert result.tool == "garak"

    def test_passed_uses_threshold(self):
        threshold = EvalThreshold(max_issues=1)
        issue = EvalIssue(check="a", severity="major", description="x")
        result = EvalResult(
            summary="1 issue",
            issues=[issue],
            trace_id=None,
            tool="garak",
            threshold=threshold,
        )
        assert result.passed is True

    def test_passed_with_ignore_severities(self):
        threshold = EvalThreshold(max_issues=0, ignore_severities=["minor"])
        issues = [
            EvalIssue(check="a", severity="minor", description="x"),
            EvalIssue(check="b", severity="major", description="y"),
        ]
        result = EvalResult(
            summary="2 issues", issues=issues, trace_id=None, tool="garak",
            threshold=threshold,
        )
        assert result.passed is False  # major issue remains

    def test_backward_compat_no_threshold_is_zero_tolerance(self):
        issue = EvalIssue(check="a", severity="minor", description="x")
        result = EvalResult(
            summary="1 issue", issues=[issue], trace_id=None, tool="garak"
        )
        assert result.passed is False  # default threshold: max_issues=0
