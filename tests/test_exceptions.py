"""Tests for pynop exceptions."""

import pytest

from pynop.exceptions import GuardRejection


class TestGuardRejection:
    def test_default_message(self):
        exc = GuardRejection()
        assert str(exc) == "Invalid input detected"

    def test_is_exception(self):
        assert issubclass(GuardRejection, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(GuardRejection, match="Invalid input detected"):
            raise GuardRejection()

    def test_reason_defaults_to_none(self):
        exc = GuardRejection()
        assert exc.reason is None

    def test_reason_can_be_set(self):
        exc = GuardRejection(reason="PII detected")
        assert exc.reason == "PII detected"
        assert str(exc) == "Invalid input detected"
