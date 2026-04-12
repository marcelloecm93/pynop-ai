"""Shared fixtures for pynop tests."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest


SAMPLE_CONFIG_YAML = """\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key-123

guards:
  input:
    on_guard_fail: reject
    on_guard_error: reject
    guards:
      - type: guardrails_ai
        validators:
          - name: DetectPII
            on_fail: exception
  output:
    on_guard_fail: reject
    on_guard_error: reject
    guards:
      - type: guardrails_ai
        validators:
          - name: DetectPII
            on_fail: exception

tracing:
  enabled: true
  provider: langfuse
  public_key: pk-test
  secret_key: sk-test
"""

MINIMAL_CONFIG_YAML = """\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key-123

guards:
  input:
    guards: []
  output:
    guards: []

tracing:
  enabled: false
"""

ENV_VAR_CONFIG_YAML = """\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${TEST_OPENAI_KEY}

guards:
  input:
    guards: []
  output:
    guards: []

tracing:
  enabled: true
  provider: langfuse
  public_key: ${TEST_LANGFUSE_PK}
  secret_key: ${TEST_LANGFUSE_SK}
"""


@pytest.fixture
def sample_config_path(tmp_path):
    """Write sample config YAML to a temp file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(SAMPLE_CONFIG_YAML)
    return str(path)


@pytest.fixture
def minimal_config_path(tmp_path):
    """Config with no guards and tracing disabled."""
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_CONFIG_YAML)
    return str(path)


@pytest.fixture
def env_var_config_path(tmp_path):
    """Config with env var references."""
    path = tmp_path / "config.yaml"
    path.write_text(ENV_VAR_CONFIG_YAML)
    return str(path)


@pytest.fixture
def mock_openai_response():
    """A mock LangChain AIMessage response."""
    response = MagicMock()
    response.content = "This is a safe LLM response."
    response.response_metadata = {
        "token_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
        },
        "model_name": "gpt-4o-mini",
    }
    return response
