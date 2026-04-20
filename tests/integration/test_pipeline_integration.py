"""Integration tests for SafetyPipeline against real services.

Requires PYNOP_INTEGRATION=1 and real API keys set in environment.
"""

import os

import pytest

from pynop import SafetyPipeline
from pynop.exceptions import GuardRejection

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PYNOP_INTEGRATION") != "1",
        reason="Integration tests require PYNOP_INTEGRATION=1",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Integration tests require OPENAI_API_KEY to be set",
    ),
]

_needs_langfuse = pytest.mark.skipif(
    not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")),
    reason="Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
)


@pytest.fixture
def integration_config(tmp_path):
    """Minimal config using real OpenAI API."""
    path = tmp_path / "config.yaml"
    path.write_text("""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: false
""")
    return str(path)


@pytest.fixture
def traced_config(tmp_path):
    """Config with Langfuse tracing enabled."""
    path = tmp_path / "config.yaml"
    path.write_text("""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: true
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
""")
    return str(path)


@pytest.fixture
def guarded_config(tmp_path):
    """Config with Guardrails-AI PII detection."""
    path = tmp_path / "config.yaml"
    path.write_text("""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
guards:
  input:
    on_guard_fail: reject
    guards:
      - type: guardrails_ai
        validators:
          - name: DetectPII
            on_fail: exception
            pii_entities:
              - EMAIL_ADDRESS
              - PHONE_NUMBER
  output:
    guards: []
tracing:
  enabled: false
""")
    return str(path)


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_basic_pipeline_run(self, integration_config):
        pipeline = SafetyPipeline.from_yaml(integration_config)
        result = await pipeline.run("Say hello in one word.")
        assert result.output
        assert isinstance(result.output, str)

    @_needs_langfuse
    @pytest.mark.asyncio
    async def test_pipeline_with_tracing(self, traced_config):
        pipeline = SafetyPipeline.from_yaml(traced_config)
        result = await pipeline.run("Say hello in one word.")
        assert result.output
        assert result.trace_id is not None

    @pytest.mark.asyncio
    async def test_guarded_pipeline_passes_clean_input(self, guarded_config):
        pipeline = SafetyPipeline.from_yaml(guarded_config)
        result = await pipeline.run("What is the capital of France?")
        assert result.output

    @pytest.mark.asyncio
    async def test_guarded_pipeline_rejects_pii(self, guarded_config):
        pipeline = SafetyPipeline.from_yaml(guarded_config)
        with pytest.raises(GuardRejection):
            await pipeline.run("My email is john@example.com and my phone is 555-1234.")


class TestEnvironmentProfileIntegration:
    @pytest.mark.asyncio
    async def test_env_profile_switches_behavior(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: false
eval:
  max_issues: 0
environments:
  dev:
    eval:
      max_issues: 10
      ignore_severities: [minor]
""")
        prod = SafetyPipeline.from_yaml(str(path))
        dev = SafetyPipeline.from_yaml(str(path), env="dev")

        assert prod.eval_threshold.max_issues == 0
        assert dev.eval_threshold.max_issues == 10
        assert dev.eval_threshold.ignore_severities == ["minor"]
