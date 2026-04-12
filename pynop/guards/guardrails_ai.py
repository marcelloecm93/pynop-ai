"""Guardrails-AI guard implementation."""

import asyncio

from guardrails import Guard
from guardrails import hub
from guardrails.errors import ValidationError
from guardrails.types import OnFailAction

from pynop.types import GuardResult


class GuardrailsAIGuard:
    """Wraps Guardrails-AI validators."""

    def __init__(self, validators_config: list[dict]):
        self._guard = Guard()
        validators = []
        for v in validators_config:
            cls = getattr(hub, v["name"])
            action = OnFailAction(v.get("on_fail", "noop"))
            kwargs = {k: val for k, val in v.items() if k not in ("name", "on_fail")}
            validators.append(cls(on_fail=action, **kwargs))
        self._guard = self._guard.use(*validators)

    @classmethod
    def from_config(cls, config: dict) -> "GuardrailsAIGuard":
        """Create a guard from a config dict."""
        return cls(validators_config=config["validators"])

    async def validate(self, text: str) -> GuardResult:
        """Validate text against configured validators."""
        try:
            result = await asyncio.to_thread(self._guard.validate, text)
        except ValidationError as e:
            return GuardResult(passed=False, reason=str(e))

        if result.validation_passed:
            return GuardResult(passed=True, reason=None)
        return GuardResult(passed=False, reason=str(result.error))
