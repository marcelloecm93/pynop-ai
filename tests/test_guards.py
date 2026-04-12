"""Tests for guard interface and GuardrailsAIGuard."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pynop.guards.base import Guard
from pynop.guards.guardrails_ai import GuardrailsAIGuard
from pynop.guards.nemo import NeMoGuard
from pynop.types import GuardResult


class TestGuardProtocol:
    """Verify the Guard protocol contract."""

    @pytest.mark.asyncio
    async def test_custom_guard_satisfies_protocol(self):
        """Any class with async validate(str) -> GuardResult is a valid Guard."""

        class StubGuard:
            async def validate(self, text: str) -> GuardResult:
                return GuardResult(passed=True, reason=None)

        guard = StubGuard()
        result = await guard.validate("hello")
        assert result.passed is True


class TestGuardrailsAIGuard:
    def test_create_from_config(self):
        config = {
            "validators": [
                {"name": "DetectPII", "on_fail": "exception"}
            ]
        }
        guard = GuardrailsAIGuard.from_config(config)
        assert guard is not None

    @pytest.mark.asyncio
    async def test_validate_clean_input_passes(self):
        config = {
            "validators": [
                {"name": "DetectPII", "on_fail": "exception"}
            ]
        }
        guard = GuardrailsAIGuard.from_config(config)

        mock_result = MagicMock()
        mock_result.validation_passed = True
        with patch.object(guard, "_guard") as mock_guard:
            mock_guard.validate.return_value = mock_result
            result = await guard.validate("Tell me about Python.")
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_pii_input_fails(self):
        config = {
            "validators": [
                {"name": "DetectPII", "on_fail": "exception"}
            ]
        }
        guard = GuardrailsAIGuard.from_config(config)

        mock_result = MagicMock()
        mock_result.validation_passed = False
        mock_result.error = "PII detected: email"
        with patch.object(guard, "_guard") as mock_guard:
            mock_guard.validate.return_value = mock_result
            result = await guard.validate("My email is test@example.com")
            assert result.passed is False
            assert "PII" in result.reason

    @pytest.mark.asyncio
    async def test_validate_returns_guard_result_type(self):
        config = {
            "validators": [
                {"name": "DetectPII", "on_fail": "exception"}
            ]
        }
        guard = GuardrailsAIGuard.from_config(config)

        mock_result = MagicMock()
        mock_result.validation_passed = True
        with patch.object(guard, "_guard") as mock_guard:
            mock_guard.validate.return_value = mock_result
            result = await guard.validate("Hello")
            assert isinstance(result, GuardResult)


class TestNeMoGuardProtocol:
    def test_nemo_guard_satisfies_protocol(self, tmp_path):
        with patch("pynop.guards.nemo.RailsConfig") as mock_cfg_cls, \
             patch("pynop.guards.nemo.LLMRails"):
            mock_cfg_cls.from_path.return_value = MagicMock()
            guard = NeMoGuard(config_path=str(tmp_path))
        assert isinstance(guard, Guard)
