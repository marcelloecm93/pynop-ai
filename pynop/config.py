"""YAML config loading with env var substitution."""

import os
import re
from pathlib import Path

import yaml

from pynop.guards.nemo_builder import KNOWN_RAILS, PARAMETERIZED_RAILS, _parse_rail_entry


from pynop.types import VALID_SEVERITIES

ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")

REQUIRED_KEYS = {"llm", "guards", "tracing"}
OPTIONAL_KEYS = {"eval", "environments"}
VALID_PROVIDERS = {"openai", "anthropic", "google", "local"}
VALID_ON_GUARD_FAIL = {"reject", "return_canned", "include_reason", "reask"}


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} references with environment variable values."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise ValueError(f"config: environment variable ${{{var_name}}} is not set")
        return val

    return ENV_VAR_PATTERN.sub(replacer, value)


def _walk_and_substitute(obj):
    """Recursively substitute env vars in all string values."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


def load_config(path: str, env: str | None = None) -> dict:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML config file.
        env: Environment profile name. Falls back to PYNOP_ENV env var.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config: file must contain a YAML mapping")

    missing = REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"config: missing required keys: {missing}")

    # Apply environment overlay before env var substitution
    env = env or os.environ.get("PYNOP_ENV")
    if env and "environments" in config:
        config = _apply_environment_overlay(config, env)

    # Remove environments from the resolved config — not needed downstream
    config.pop("environments", None)

    config = _walk_and_substitute(config)

    _validate_resolved_config(config)

    return config


def _apply_environment_overlay(config: dict, env: str) -> dict:
    """Apply section-level overlay from an environment profile."""
    environments = config.get("environments", {})
    if env not in environments:
        return config

    profile = environments[env]
    if not isinstance(profile, dict):
        raise ValueError(f"environments.{env}: must be a mapping")

    allowed_keys = REQUIRED_KEYS | OPTIONAL_KEYS - {"environments"}
    for key in profile:
        if key not in allowed_keys:
            raise ValueError(f"environments.{env}: unknown key '{key}'")

    # Section-level replace: profile sections replace base sections entirely
    merged = dict(config)
    for key, value in profile.items():
        merged[key] = value

    return merged


def _validate_resolved_config(config: dict) -> None:
    """Validate a fully resolved config (after overlay and env var substitution)."""
    if not config["llm"].get("api_key"):
        raise ValueError("llm.api_key is required (set the referenced env var)")

    provider = config["llm"].get("provider", "openai")
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"llm.provider: '{provider}' is not valid. Must be one of {VALID_PROVIDERS}")

    if provider == "local" and not config["llm"].get("base_url"):
        raise ValueError("llm.base_url is required when provider is 'local'")

    for slot_name in ("input", "output"):
        slot_cfg = config["guards"].get(slot_name, {})
        _validate_guard_slot(slot_cfg, slot_name)

    if "eval" in config:
        _validate_eval_section(config["eval"])


def _validate_guard_slot(slot_cfg: dict, slot_name: str) -> None:
    """Validate per-guard config within a guard slot."""
    slot_default = slot_cfg.get("on_guard_fail", "reject")
    if slot_default not in VALID_ON_GUARD_FAIL:
        raise ValueError(f"guards.{slot_name}.on_guard_fail must be one of {VALID_ON_GUARD_FAIL}")

    for i, guard_cfg in enumerate(slot_cfg.get("guards", [])):
        on_fail = guard_cfg.get("on_guard_fail", slot_default)
        if on_fail not in VALID_ON_GUARD_FAIL:
            raise ValueError(
                f"guards.{slot_name}.guards[{i}].on_guard_fail must be one of {VALID_ON_GUARD_FAIL}"
            )

        if on_fail == "reask":
            if slot_name == "input":
                raise ValueError(
                    f"guards.{slot_name}.guards[{i}]: on_guard_fail 'reask' is not allowed on input guards"
                )

            max_reask = guard_cfg.get("max_reask", 2)
            if not isinstance(max_reask, int) or max_reask < 1:
                raise ValueError(
                    f"guards.{slot_name}.guards[{i}].max_reask must be an integer >= 1"
                )

            reask_instruction = guard_cfg.get("reask_instruction")
            if reask_instruction and "{reason}" not in reask_instruction:
                raise ValueError(
                    f"guards.{slot_name}.guards[{i}].reask_instruction must contain '{{reason}}' placeholder"
                )

        if guard_cfg.get("type") == "nemo":
            _validate_nemo_guard(guard_cfg, slot_name, i)


def _validate_nemo_guard(guard_cfg: dict, slot_name: str, index: int) -> None:
    """Validate a NeMo guard config entry."""
    prefix = f"guards.{slot_name}.guards[{index}]"

    has_path = "config_path" in guard_cfg
    has_rails = "rails" in guard_cfg

    if has_path and has_rails:
        raise ValueError(f"{prefix}: config_path and rails are mutually exclusive")
    if not has_path and not has_rails:
        raise ValueError(f"{prefix}: NeMo guard requires config_path or rails")

    if has_rails:
        for j, rail in enumerate(guard_cfg["rails"]):
            try:
                name, params = _parse_rail_entry(rail)
            except ValueError:
                raise ValueError(f"{prefix}.rails[{j}]: must be a string or mapping")

            if name not in KNOWN_RAILS:
                raise ValueError(
                    f"{prefix}.rails[{j}]: unknown rail '{name}'. Known: {sorted(KNOWN_RAILS)}"
                )
            if name in PARAMETERIZED_RAILS:
                if not params or not isinstance(params, dict):
                    raise ValueError(f"{prefix}.rails[{j}]: rail '{name}' requires parameters")
                if name == "topic_control" and not (params.get("allowed") or params.get("denied")):
                    raise ValueError(
                        f"{prefix}.rails[{j}]: topic_control requires 'allowed' or 'denied'"
                    )


def _validate_eval_section(eval_cfg: dict) -> None:
    """Validate the eval: config section."""
    if not isinstance(eval_cfg, dict):
        raise ValueError("eval: must be a mapping")

    max_issues = eval_cfg.get("max_issues", 0)
    if not isinstance(max_issues, int) or max_issues < 0:
        raise ValueError("eval.max_issues must be a non-negative integer")

    ignore_severities = eval_cfg.get("ignore_severities", [])
    if not isinstance(ignore_severities, list):
        raise ValueError("eval.ignore_severities must be a list")
    for sev in ignore_severities:
        if sev not in VALID_SEVERITIES:
            raise ValueError(
                f"eval.ignore_severities: '{sev}' is not valid. Must be one of {VALID_SEVERITIES}"
            )

    garak_severities = eval_cfg.get("garak_severities", {})
    if not isinstance(garak_severities, dict):
        raise ValueError("eval.garak_severities must be a mapping")
    for probe, sev in garak_severities.items():
        if sev not in VALID_SEVERITIES:
            raise ValueError(
                f"eval.garak_severities.{probe}: '{sev}' is not valid. Must be one of {VALID_SEVERITIES}"
            )

    # Per-tool threshold overrides
    for tool_name in ("garak", "giskard"):
        tool_cfg = eval_cfg.get(tool_name)
        if tool_cfg is not None:
            _validate_eval_tool_section(tool_cfg, tool_name)


def _validate_eval_tool_section(tool_cfg: dict, tool_name: str) -> None:
    """Validate a per-tool override section under eval:."""
    if not isinstance(tool_cfg, dict):
        raise ValueError(f"eval.{tool_name}: must be a mapping")

    if "max_issues" in tool_cfg:
        max_issues = tool_cfg["max_issues"]
        if not isinstance(max_issues, int) or max_issues < 0:
            raise ValueError(f"eval.{tool_name}.max_issues must be a non-negative integer")

    if "ignore_severities" in tool_cfg:
        ignore_severities = tool_cfg["ignore_severities"]
        if not isinstance(ignore_severities, list):
            raise ValueError(f"eval.{tool_name}.ignore_severities must be a list")
        for sev in ignore_severities:
            if sev not in VALID_SEVERITIES:
                raise ValueError(
                    f"eval.{tool_name}.ignore_severities: '{sev}' is not valid. Must be one of {VALID_SEVERITIES}"
                )

    if "severities" in tool_cfg:
        severities = tool_cfg["severities"]
        if not isinstance(severities, dict):
            raise ValueError(f"eval.{tool_name}.severities must be a mapping")
        for probe, sev in severities.items():
            if sev not in VALID_SEVERITIES:
                raise ValueError(
                    f"eval.{tool_name}.severities.{probe}: '{sev}' is not valid. Must be one of {VALID_SEVERITIES}"
                )
