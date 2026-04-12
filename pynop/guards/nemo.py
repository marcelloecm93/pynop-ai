"""NeMo Guardrails guard implementation."""

import os
import shutil
import tempfile

from langchain_core.language_models import BaseChatModel
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus

from pynop.guards.nemo_builder import NeMoConfigBuilder, _parse_rail_entry
from pynop.types import GuardResult


class NeMoGuard:
    """Wraps NeMo Guardrails LLMRails.check_async() as a Guard."""

    def __init__(self, config_path: str, role: str = "user", llm: BaseChatModel | None = None):
        rails_config = RailsConfig.from_path(config_path)
        self._rails = LLMRails(config=rails_config, llm=llm)
        self._role = role

    @classmethod
    def from_config(cls, cfg: dict, yaml_dir: str, role: str = "user", llm: BaseChatModel | None = None) -> "NeMoGuard":
        """Create a NeMoGuard from config_path or inline rails declarations."""
        if "rails" in cfg:
            return cls._from_rails(cfg["rails"], role=role, llm=llm)

        config_path = os.path.join(yaml_dir, cfg["config_path"])
        return cls(config_path=config_path, role=role, llm=llm)

    @classmethod
    def _from_rails(cls, rails: list, role: str = "user", llm: BaseChatModel | None = None) -> "NeMoGuard":
        """Build NeMo config from inline rail declarations."""
        builder = NeMoConfigBuilder()
        for rail in rails:
            name, params = _parse_rail_entry(rail)
            builder.add_rail(name, params)

        tmp_dir = tempfile.mkdtemp(prefix="pynop_nemo_")
        try:
            builder.build(tmp_dir)
            return cls(config_path=tmp_dir, role=role, llm=llm)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def validate(self, text: str) -> GuardResult:
        """Validate text using NeMo rails."""
        messages = [{"role": self._role, "content": text}]
        result = await self._rails.check_async(messages)

        if result.status == RailStatus.BLOCKED:
            return GuardResult(passed=False, reason=result.rail or str(result))
        if result.status == RailStatus.MODIFIED:
            return GuardResult(passed=True, reason=None, modified=True)
        return GuardResult(passed=True, reason=None)
