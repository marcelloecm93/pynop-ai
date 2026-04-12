"""Tests for the LangChain provider factory and multi-backend support."""

from unittest.mock import MagicMock, patch

import pytest

from pynop.pipeline import _build_llm, SafetyPipeline


class TestBuildLlm:
    def test_openai_provider(self):
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_llm({"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-test"})
        mock_cls.assert_called_once_with(model="gpt-4o-mini", api_key="sk-test")

    def test_anthropic_provider(self):
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_llm({"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "sk-ant-test"})
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514", api_key="sk-ant-test")

    def test_google_provider(self):
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_llm({"provider": "google", "model": "gemini-2.0-flash", "api_key": "goog-test"})
        mock_cls.assert_called_once_with(model="gemini-2.0-flash", api_key="goog-test")

    def test_local_provider(self):
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_llm({"provider": "local", "model": "llama3", "api_key": "not-needed", "base_url": "http://localhost:11434/v1"})
        mock_cls.assert_called_once_with(model="llama3", api_key="not-needed", base_url="http://localhost:11434/v1")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="llm: unknown provider 'foobar'"):
            _build_llm({"provider": "foobar", "model": "x", "api_key": "y"})

    def test_defaults_to_openai(self):
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_llm({"model": "gpt-4o-mini", "api_key": "sk-test"})
        mock_cls.assert_called_once()


class TestPipelinePrebuiltLlm:
    def test_prebuilt_llm_overrides_config(self, minimal_config_path):
        """A pre-built BaseChatModel passed to __init__ is used instead of config."""
        with patch("langfuse.Langfuse"):
            pipeline = SafetyPipeline.from_yaml(minimal_config_path)

        custom_llm = MagicMock()
        pipeline._llm = custom_llm
        assert pipeline._llm is custom_llm

    def test_constructor_accepts_llm_param(self):
        """SafetyPipeline.__init__ accepts an optional llm parameter."""
        from pynop.tracing import Tracer
        from pynop.types import GuardSlot

        custom_llm = MagicMock()
        with patch("langfuse.Langfuse"):
            pipeline = SafetyPipeline(
                llm_config={"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
                input_slot=GuardSlot(),
                output_slot=GuardSlot(),
                tracer=Tracer(enabled=False),
                llm=custom_llm,
            )
        assert pipeline._llm is custom_llm


class TestConfigProviderValidation:
    def test_unknown_provider_rejected_at_config(self, tmp_path):
        from pynop.config import load_config

        path = tmp_path / "config.yaml"
        path.write_text(
            "llm:\n  provider: cohere\n  model: x\n  api_key: y\n"
            "guards:\n  input:\n    guards: []\n  output:\n    guards: []\n"
            "tracing:\n  enabled: false\n"
        )
        with pytest.raises(ValueError, match="llm.provider:.*is not valid"):
            load_config(str(path))

    def test_local_without_base_url_rejected(self, tmp_path):
        from pynop.config import load_config

        path = tmp_path / "config.yaml"
        path.write_text(
            "llm:\n  provider: local\n  model: llama3\n  api_key: x\n"
            "guards:\n  input:\n    guards: []\n  output:\n    guards: []\n"
            "tracing:\n  enabled: false\n"
        )
        with pytest.raises(ValueError, match="base_url is required"):
            load_config(str(path))

    def test_valid_providers_accepted(self, tmp_path):
        from pynop.config import load_config

        for provider in ("openai", "anthropic", "google"):
            path = tmp_path / f"config_{provider}.yaml"
            path.write_text(
                f"llm:\n  provider: {provider}\n  model: x\n  api_key: y\n"
                "guards:\n  input:\n    guards: []\n  output:\n    guards: []\n"
                "tracing:\n  enabled: false\n"
            )
            config = load_config(str(path))
            assert config["llm"]["provider"] == provider
