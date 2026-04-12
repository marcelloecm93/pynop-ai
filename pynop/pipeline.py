"""SafetyPipeline — main entry point."""

import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from pynop.config import load_config
from pynop.exceptions import GuardRejection
from pynop.guards.guardrails_ai import GuardrailsAIGuard
from pynop.tracing import Tracer
from pynop.types import (
    DEFAULT_REASK_INSTRUCTION,
    EvalThreshold,
    GuardConfig,
    GuardResult,
    GuardSlot,
    PipelineResult,
    ReaskSignal,
)

logger = logging.getLogger(__name__)


def _build_llm(llm_config: dict) -> BaseChatModel:
    """Build a LangChain chat model from config."""
    provider = llm_config.get("provider", "openai")
    model = llm_config["model"]
    api_key = llm_config.get("api_key")

    if provider in ("openai", "local"):
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model, "api_key": api_key}
        if provider == "local":
            kwargs["base_url"] = llm_config["base_url"]
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, api_key=api_key)

    raise ValueError(f"llm: unknown provider '{provider}'")


class SafetyPipeline:
    """Async pipeline: input guards → LLM call → output guards, all traced via Langfuse.

    Construct via :meth:`from_yaml` to load a YAML config, or instantiate directly with
    pre-built components (useful when injecting a custom :class:`BaseChatModel`).

    Public surface:
        - :meth:`from_yaml` — factory from a YAML config file (with optional env profile)
        - :meth:`run` — main entry point; runs guards + LLM and returns a ``PipelineResult``
        - :attr:`tracer` — read-only access to the underlying ``Tracer``
        - :attr:`eval_threshold` — read-only base ``EvalThreshold``
        - :meth:`eval_threshold_for` — resolve the per-tool threshold (used by ``EvalRunner``)

    Args:
        llm_config: Provider/model dict (e.g. ``{"provider": "openai", "model": "gpt-4o"}``).
        input_slot: ``GuardSlot`` configured with the input guards and rejection strategy.
        output_slot: ``GuardSlot`` configured with the output guards and rejection strategy.
        tracer: ``Tracer`` instance (use ``Tracer(enabled=False)`` to disable tracing).
        llm: Optional pre-built ``BaseChatModel``. If omitted, one is constructed from
            ``llm_config`` via the LangChain provider corresponding to ``provider:``.
        eval_threshold: Default ``EvalThreshold`` used by ``EvalRunner`` when no per-tool
            override is set. Defaults to zero-tolerance.
        eval_tool_thresholds: Optional ``{"garak": EvalThreshold, "giskard": EvalThreshold}``
            map of per-tool overrides.
    """

    def __init__(
        self,
        llm_config: dict,
        input_slot: GuardSlot,
        output_slot: GuardSlot,
        tracer: Tracer,
        llm: BaseChatModel | None = None,
        eval_threshold: EvalThreshold | None = None,
        eval_tool_thresholds: dict[str, EvalThreshold] | None = None,
    ):
        self._llm_config = llm_config
        self._input_slot = input_slot
        self._output_slot = output_slot
        self._tracer = tracer
        self._llm = llm or _build_llm(llm_config)
        self._eval_threshold = eval_threshold or EvalThreshold()
        self._eval_tool_thresholds = eval_tool_thresholds or {}

    @property
    def tracer(self) -> Tracer:
        """Public read-only access to the tracer."""
        return self._tracer

    @property
    def eval_threshold(self) -> EvalThreshold:
        """Public read-only access to the base eval threshold."""
        return self._eval_threshold

    def eval_threshold_for(self, tool: str) -> EvalThreshold:
        """Resolve the eval threshold for a specific tool.

        Returns the per-tool threshold if configured, otherwise the base threshold.
        """
        return self._eval_tool_thresholds.get(tool, self._eval_threshold)

    @classmethod
    def from_yaml(cls, path: str, env: str | None = None) -> "SafetyPipeline":
        """Create a ``SafetyPipeline`` from a YAML config file.

        Args:
            path: Filesystem path to the YAML config.
            env: Environment profile name to apply (e.g. ``"dev"``, ``"prod"``).
                Falls back to the ``PYNOP_ENV`` environment variable.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the config is malformed, missing required keys, or
                references unset environment variables.
        """
        config = load_config(path, env=env)
        yaml_dir = os.path.dirname(os.path.abspath(path))

        llm = _build_llm(config["llm"])

        input_slot = _build_guard_slot(config["guards"].get("input", {}), yaml_dir, role="user", llm=llm)
        output_slot = _build_guard_slot(config["guards"].get("output", {}), yaml_dir, role="assistant", llm=llm)

        tracing_cfg = config.get("tracing", {})
        tracer = Tracer(
            enabled=tracing_cfg.get("enabled", False),
            public_key=tracing_cfg.get("public_key"),
            secret_key=tracing_cfg.get("secret_key"),
            base_url=tracing_cfg.get("base_url"),
        )

        eval_cfg = config.get("eval", {})
        eval_threshold = EvalThreshold(
            max_issues=eval_cfg.get("max_issues", 0),
            ignore_severities=eval_cfg.get("ignore_severities", []),
            garak_severities=eval_cfg.get("garak_severities", {}),
        )

        # Per-tool threshold overrides (inherit from base, override with tool-specific)
        eval_tool_thresholds = {}
        for tool_name in ("garak", "giskard"):
            tool_cfg = eval_cfg.get(tool_name)
            if tool_cfg is not None:
                garak_severities = {}
                if tool_name == "garak":
                    garak_severities = tool_cfg.get(
                        "severities", eval_cfg.get("garak_severities", {})
                    )
                eval_tool_thresholds[tool_name] = EvalThreshold(
                    max_issues=tool_cfg.get("max_issues", eval_threshold.max_issues),
                    ignore_severities=tool_cfg.get(
                        "ignore_severities", eval_threshold.ignore_severities
                    ),
                    garak_severities=garak_severities,
                )

        return cls(
            llm_config=config["llm"],
            input_slot=input_slot,
            output_slot=output_slot,
            tracer=tracer,
            llm=llm,
            eval_threshold=eval_threshold,
            eval_tool_thresholds=eval_tool_thresholds,
        )

    async def run(self, prompt: str, **kwargs) -> PipelineResult:
        """Run the pipeline end-to-end.

        Order: input guards → LLM call → output guards. On guard failure, the
        configured rejection strategy applies (``reject``, ``return_canned``,
        ``include_reason``, or ``reask`` for output guards).

        Args:
            prompt: Non-empty user prompt. Wrapped as a single ``HumanMessage``.
            **kwargs: Forwarded to the underlying LangChain ``ainvoke`` call
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            ``PipelineResult`` with ``output``, ``trace_id``, and ``guarded`` fields.

        Raises:
            ValueError: If ``prompt`` is empty or whitespace-only.
            GuardRejection: If a guard rejects under a ``reject`` or
                ``include_reason`` strategy, or after exhausting ``max_reask``.
        """
        if not prompt or not prompt.strip():
            raise ValueError("pipeline.run: prompt must be a non-empty string")
        trace = self._tracer.start_trace(prompt=prompt)
        try:
            early_result = await self._run_guards(self._input_slot, prompt, trace, "input_guard")
            if early_result is not None:
                self._tracer.end_trace(trace, output=early_result.output)
                return early_result

            messages: list[BaseMessage] = [HumanMessage(content=prompt)]
            reask_counters: dict[int, int] = {}
            reask_attempt = 0

            while True:
                output, response = await self._call_and_trace_llm(
                    messages, trace, reask_attempt, **kwargs
                )

                guard_result = await self._run_guards(
                    self._output_slot, output, trace, "output_guard"
                )

                if guard_result is None:
                    break

                if isinstance(guard_result, ReaskSignal):
                    idx = guard_result.guard_index
                    reask_counters[idx] = reask_counters.get(idx, 0) + 1
                    if reask_counters[idx] > guard_result.max_reask:
                        raise GuardRejection(
                            f"output_guard: rejected after {guard_result.max_reask} reask attempt(s)",
                            reason=guard_result.reason,
                        )

                    instruction = guard_result.reask_instruction.format(
                        reason=guard_result.reason or "Unknown reason"
                    )
                    messages.append(AIMessage(content=output))
                    messages.append(HumanMessage(content=instruction))
                    reask_attempt += 1
                    continue

                # PipelineResult (e.g. return_canned)
                self._tracer.end_trace(trace, output=guard_result.output)
                return guard_result

            self._tracer.end_trace(trace, output=output)
        except GuardRejection:
            self._tracer.end_trace(trace, output=None)
            raise
        finally:
            self._tracer.flush()

        return PipelineResult(
            output=output,
            trace_id=trace.id if trace else None,
            guarded=bool(self._input_slot.guards or self._output_slot.guards),
        )

    async def _call_and_trace_llm(
        self, messages: list[BaseMessage], trace, reask_attempt: int, **kwargs
    ) -> tuple[str, AIMessage]:
        """Call the LLM, trace the span, return (output_text, response)."""
        span = self._tracer.start_span(trace, name="llm_call", as_type="generation")
        response = await self._call_llm(messages, **kwargs)
        output = response.content
        model_name = (
            getattr(self._llm, "model_name", None)
            or getattr(self._llm, "model", None)
            or self._llm_config.get("model")
        )
        metadata = {"model": model_name, "reask_attempt": reask_attempt}
        if response.response_metadata and (
            token_usage := response.response_metadata.get("token_usage")
        ):
            metadata["usage"] = token_usage
        self._tracer.end_span(span, metadata=metadata)
        return output, response

    async def _run_guards(
        self, slot: GuardSlot, text, trace, span_name
    ) -> PipelineResult | ReaskSignal | None:
        """Run guards in a slot. Returns None to continue, PipelineResult or ReaskSignal on failure."""
        span = self._tracer.start_span(trace, name=span_name, as_type="guardrail")
        for i, guard in enumerate(slot.guards):
            try:
                result = await guard.validate(text)
            except Exception as exc:
                if slot.on_guard_error == "pass":
                    logger.warning("Guard %s raised %s — skipping", guard, exc)
                    error_span = self._tracer.start_span(trace, name=f"{span_name}_error", as_type="guardrail")
                    self._tracer.end_span(error_span, metadata={"error": True, "exception": str(exc)})
                    continue
                result = GuardResult(passed=False, reason=str(exc))

            if not isinstance(result, GuardResult):
                logger.warning("Guard %s returned %s instead of GuardResult — treating as failure", guard, type(result).__name__)
                result = GuardResult(passed=False, reason=f"guard returned {type(result).__name__} instead of GuardResult")

            if not result.passed:
                self._tracer.end_span(span, metadata={"passed": False, "reason": result.reason})
                if i < len(slot.guard_configs):
                    guard_cfg = slot.guard_configs[i]
                else:
                    guard_cfg = GuardConfig(on_guard_fail=slot.on_guard_fail)
                on_fail = guard_cfg.on_guard_fail
                return self._apply_rejection(slot, span_name, on_fail, guard_cfg, result.reason, trace.id if trace else None, i)

        self._tracer.end_span(span, metadata={"passed": True})
        return None

    def _apply_rejection(
        self,
        slot: GuardSlot,
        slot_name: str,
        on_fail: str,
        guard_cfg: GuardConfig,
        reason: str | None,
        trace_id: str | None,
        guard_index: int,
    ) -> PipelineResult | ReaskSignal:
        """Apply the configured rejection strategy."""
        if on_fail == "reask":
            return ReaskSignal(
                guard_index=guard_index,
                reask_instruction=guard_cfg.reask_instruction,
                max_reask=guard_cfg.max_reask,
                reason=reason,
            )
        if on_fail == "return_canned":
            return PipelineResult(output=slot.canned_response, trace_id=trace_id, guarded=True)
        if on_fail == "include_reason":
            message = f"{slot_name}: {reason}" if reason else f"{slot_name}: rejected"
            raise GuardRejection(message, reason=reason)
        raise GuardRejection(f"{slot_name}: rejected")

    async def _call_llm(self, messages: list[BaseMessage], **kwargs):
        """Call the LLM via LangChain."""
        return await self._llm.ainvoke(messages, **kwargs)


def _build_guard_slot(slot_config: dict, yaml_dir: str, role: str = "user", llm: BaseChatModel | None = None) -> GuardSlot:
    """Build a GuardSlot from a config dict."""
    from pynop.guards.nemo import NeMoGuard

    slot_default_on_fail = slot_config.get("on_guard_fail", "reject")
    raw_guards = slot_config.get("guards", [])
    guards = []
    guard_configs = []

    for cfg in raw_guards:
        if cfg["type"] == "guardrails_ai":
            guards.append(GuardrailsAIGuard.from_config(cfg))
        elif cfg["type"] == "nemo":
            guards.append(NeMoGuard.from_config(cfg, yaml_dir, role=role, llm=llm))

        on_fail = cfg.get("on_guard_fail", slot_default_on_fail)
        guard_configs.append(GuardConfig(
            on_guard_fail=on_fail,
            max_reask=cfg.get("max_reask", 2),
            reask_instruction=cfg.get("reask_instruction", DEFAULT_REASK_INSTRUCTION),
        ))

    on_guard_error = slot_config.get("on_guard_error", "reject")
    canned_response = slot_config.get("canned_response")

    if slot_default_on_fail == "return_canned" and not canned_response:
        raise ValueError("guards: canned_response is required when on_guard_fail is 'return_canned'")

    return GuardSlot(
        guards=guards,
        guard_configs=guard_configs,
        on_guard_fail=slot_default_on_fail,
        on_guard_error=on_guard_error,
        canned_response=canned_response,
    )
