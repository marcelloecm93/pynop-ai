"""Tests for YAML config loading."""

import os

import pytest

from pynop.config import load_config


class TestLoadConfig:
    def test_loads_valid_config(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["model"] == "gpt-4o-mini"

    def test_loads_guard_definitions(self, sample_config_path):
        config = load_config(sample_config_path)
        guards = config["guards"]["input"]["guards"]
        assert len(guards) == 1
        assert guards[0]["type"] == "guardrails_ai"
        assert guards[0]["validators"][0]["name"] == "DetectPII"

    def test_loads_tracing_config(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config["tracing"]["enabled"] is True
        assert config["tracing"]["provider"] == "langfuse"

    def test_minimal_config(self, minimal_config_path):
        config = load_config(minimal_config_path)
        assert config["guards"]["input"]["guards"] == []
        assert config["guards"]["output"]["guards"] == []
        assert config["tracing"]["enabled"] is False

    def test_env_var_substitution(self, env_var_config_path, monkeypatch):
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-real-key")
        monkeypatch.setenv("TEST_LANGFUSE_PK", "pk-real")
        monkeypatch.setenv("TEST_LANGFUSE_SK", "sk-real")

        config = load_config(env_var_config_path)
        assert config["llm"]["api_key"] == "sk-real-key"
        assert config["tracing"]["public_key"] == "pk-real"
        assert config["tracing"]["secret_key"] == "sk-real"

    def test_missing_env_var_raises(self, env_var_config_path):
        # Ensure the env vars are NOT set
        for key in ("TEST_OPENAI_KEY", "TEST_LANGFUSE_PK", "TEST_LANGFUSE_SK"):
            os.environ.pop(key, None)

        with pytest.raises(ValueError, match=r"config: environment variable \$\{TEST_OPENAI_KEY\} is not set"):
            load_config(env_var_config_path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_missing_required_keys_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("foo: bar\n")
        with pytest.raises(ValueError):
            load_config(str(path))


class TestReaskConfigValidation:
    def _write_config(self, tmp_path, guards_yaml):
        path = tmp_path / "config.yaml"
        path.write_text(f"""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
{guards_yaml}
tracing:
  enabled: false
""")
        return str(path)

    def test_reask_on_output_guard_is_valid(self, tmp_path):
        config = load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards: []
  output:
    guards:
      - type: guardrails_ai
        on_guard_fail: reask
        max_reask: 3
        validators:
          - name: ToxicLanguage
            on_fail: exception
"""))
        assert config is not None

    def test_reask_on_input_guard_raises(self, tmp_path):
        with pytest.raises(ValueError, match="reask.*not allowed on input"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: guardrails_ai
        on_guard_fail: reask
        validators:
          - name: DetectPII
            on_fail: exception
  output:
    guards: []
"""))

    def test_max_reask_less_than_1_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_reask must be an integer >= 1"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards: []
  output:
    guards:
      - type: guardrails_ai
        on_guard_fail: reask
        max_reask: 0
        validators:
          - name: ToxicLanguage
            on_fail: exception
"""))

    def test_reask_instruction_without_reason_raises(self, tmp_path):
        with pytest.raises(ValueError, match="reask_instruction must contain"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards: []
  output:
    guards:
      - type: guardrails_ai
        on_guard_fail: reask
        reask_instruction: "Try again please"
        validators:
          - name: ToxicLanguage
            on_fail: exception
"""))

    def test_slot_default_inherited_by_guard(self, tmp_path):
        """Guard without on_guard_fail inherits slot default."""
        config = load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards: []
  output:
    on_guard_fail: include_reason
    guards:
      - type: guardrails_ai
        validators:
          - name: DetectPII
            on_fail: exception
"""))
        # Guard doesn't specify on_guard_fail — should inherit slot default
        guard = config["guards"]["output"]["guards"][0]
        assert guard.get("on_guard_fail") is None  # not set on raw config
        # Actual inheritance happens in _build_guard_slot


class TestNemoRailsConfigValidation:
    def _write_config(self, tmp_path, guards_yaml):
        path = tmp_path / "config.yaml"
        path.write_text(f"""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
{guards_yaml}
tracing:
  enabled: false
""")
        return str(path)

    def test_valid_inline_rails(self, tmp_path):
        config = load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        rails:
          - jailbreak
  output:
    guards: []
"""))
        assert config is not None

    def test_config_path_and_rails_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        config_path: ./some_dir
        rails:
          - jailbreak
  output:
    guards: []
"""))

    def test_nemo_guard_requires_config_path_or_rails(self, tmp_path):
        with pytest.raises(ValueError, match="requires config_path or rails"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
  output:
    guards: []
"""))

    def test_unknown_rail_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown rail"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        rails:
          - nonexistent_rail
  output:
    guards: []
"""))

    def test_topic_control_without_params_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires parameters"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        rails:
          - topic_control
  output:
    guards: []
"""))

    def test_topic_control_without_allowed_or_denied_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires 'allowed' or 'denied'"):
            load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        rails:
          - topic_control:
              other_param: foo
  output:
    guards: []
"""))

    def test_valid_topic_control_with_allowed(self, tmp_path):
        config = load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        rails:
          - topic_control:
              allowed: [coding, data science]
  output:
    guards: []
"""))
        assert config is not None

    def test_valid_multi_rail_config(self, tmp_path):
        config = load_config(self._write_config(tmp_path, """\
guards:
  input:
    guards:
      - type: nemo
        rails:
          - jailbreak
          - topic_control:
              allowed: [coding]
              denied: [politics]
  output:
    guards:
      - type: nemo
        rails:
          - content_safety
          - pii
"""))
        assert config is not None


class TestEvalConfigValidation:
    def _write_config(self, tmp_path, extra=""):
        path = tmp_path / "config.yaml"
        path.write_text(f"""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: false
{extra}
""")
        return str(path)

    def test_eval_section_parsed(self, tmp_path):
        config = load_config(self._write_config(tmp_path, """\
eval:
  max_issues: 5
  ignore_severities: [minor]
  garak_severities:
    dan: major
    glitch: minor
"""))
        assert config["eval"]["max_issues"] == 5
        assert config["eval"]["ignore_severities"] == ["minor"]
        assert config["eval"]["garak_severities"]["dan"] == "major"

    def test_eval_section_optional(self, tmp_path):
        config = load_config(self._write_config(tmp_path))
        assert "eval" not in config

    def test_eval_negative_max_issues_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_issues must be a non-negative integer"):
            load_config(self._write_config(tmp_path, """\
eval:
  max_issues: -1
"""))

    def test_eval_invalid_ignore_severity_raises(self, tmp_path):
        with pytest.raises(ValueError, match="is not valid"):
            load_config(self._write_config(tmp_path, """\
eval:
  ignore_severities: [high]
"""))

    def test_eval_invalid_garak_severity_raises(self, tmp_path):
        with pytest.raises(ValueError, match="is not valid"):
            load_config(self._write_config(tmp_path, """\
eval:
  garak_severities:
    dan: critical
"""))


class TestEnvironmentOverlay:
    def _write_config(self, tmp_path, environments="", eval_section=""):
        path = tmp_path / "config.yaml"
        path.write_text(f"""\
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
guards:
  input:
    guards: []
  output:
    guards: []
tracing:
  enabled: true
  public_key: pk-test
  secret_key: sk-test
{eval_section}
{environments}
""")
        return str(path)

    def test_env_overlay_replaces_section(self, tmp_path):
        config = load_config(self._write_config(tmp_path, environments="""\
environments:
  dev:
    tracing:
      enabled: false
"""), env="dev")
        assert config["tracing"]["enabled"] is False
        assert "public_key" not in config["tracing"]  # section replaced entirely

    def test_missing_section_falls_through_to_base(self, tmp_path):
        config = load_config(self._write_config(tmp_path, environments="""\
environments:
  dev:
    tracing:
      enabled: false
"""), env="dev")
        assert config["llm"]["model"] == "gpt-4o-mini"  # not overridden

    def test_pynop_env_var_selection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYNOP_ENV", "dev")
        config = load_config(self._write_config(tmp_path, environments="""\
environments:
  dev:
    tracing:
      enabled: false
"""))
        assert config["tracing"]["enabled"] is False

    def test_explicit_env_overrides_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYNOP_ENV", "dev")
        config = load_config(self._write_config(tmp_path, environments="""\
environments:
  dev:
    tracing:
      enabled: false
  staging:
    tracing:
      enabled: true
      public_key: pk-staging
      secret_key: sk-staging
"""), env="staging")
        assert config["tracing"]["enabled"] is True

    def test_unknown_env_uses_base(self, tmp_path):
        config = load_config(self._write_config(tmp_path, environments="""\
environments:
  dev:
    tracing:
      enabled: false
"""), env="nonexistent")
        assert config["tracing"]["enabled"] is True  # base config

    def test_no_environments_section_works(self, tmp_path):
        config = load_config(self._write_config(tmp_path), env="dev")
        assert config["tracing"]["enabled"] is True  # base config unchanged

    def test_env_overlay_with_eval_section(self, tmp_path):
        config = load_config(self._write_config(tmp_path,
            eval_section="""\
eval:
  max_issues: 0
""",
            environments="""\
environments:
  dev:
    eval:
      max_issues: 10
      ignore_severities: [minor]
"""), env="dev")
        assert config["eval"]["max_issues"] == 10
        assert config["eval"]["ignore_severities"] == ["minor"]

    def test_environments_key_stripped_from_result(self, tmp_path):
        config = load_config(self._write_config(tmp_path, environments="""\
environments:
  dev:
    tracing:
      enabled: false
"""), env="dev")
        assert "environments" not in config
