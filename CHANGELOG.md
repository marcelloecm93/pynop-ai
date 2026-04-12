# Changelog

All notable changes to pynop are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-11

First public release. The API surface is frozen and pynop is published to PyPI.

### Added
- Published to PyPI under the distribution name `pynop-ai`. The Python import name remains `pynop`.
- `CHANGELOG.md` summarizing every release from v0.1 through v1.0.
- `examples/` directory with a runnable `basic_openai.py` script and matching `basic_config.yaml`.
- README "Install" section covering `pip install pynop-ai` / `uv add pynop-ai` and the optional extras.
- README sections covering per-tool eval thresholds (v0.81), latency benchmarking (v0.9), and the integration test opt-in (`PYNOP_INTEGRATION=1`).
- API reference docstrings on every public symbol exported from `pynop` and `pynop.eval`.
- GitHub Actions workflows: `test.yml` (push/PR unit tests) and `integration.yml` (manual/scheduled integration tests).

### Changed
- `pyproject.toml` rewritten with full PyPI metadata: description, author, license field, project URLs, classifiers, and keywords.
- Dependencies split into a lean core (`pyyaml`, `langchain-core`, `langchain-openai`, `guardrails-ai`, `langfuse`) plus optional extras: `[anthropic]`, `[google]`, `[nemo]`, `[eval]`, and `[all]`. Users only install what they use; missing extras surface a clear `ModuleNotFoundError` at `from_yaml`/`run_*` time.
- Version bumped to `1.0.0`.

### Removed
- Pre-v1.0 scratch files from the repo root: `main.py`, `try_eval.py`, `try_guard_rejection.py`, `try_jailbreak.py`, `try_pipeline.py`, `try_reask.py`, `test_config.yaml`, `test_reask.yaml`, and `config_eval.yaml`.

### Known limitations
- `requires-python` is pinned to `>=3.12,<3.13`. Python 3.13 is not yet supported because `giskard` (and transitive `langchain-community` / `numpy`) lacks a compatible release for 3.13. The pin will widen as soon as the upstream stack catches up.

## [0.9.0] — Hardening & Integration Testing

### Added
- `LatencyBenchmark` utility that runs the same prompt set through two pipeline configurations and produces a side-by-side latency report (per-guard breakdown plus p50/p95/p99) sourced from Langfuse trace data.
- `tests/integration/` directory containing opt-in tests (`PYNOP_INTEGRATION=1`) that hit real OpenAI, Garak, Giskard, and Langfuse instances end-to-end.

### Changed
- Standardized `ValueError` and `GuardRejection` messages to the `"<component>: <what went wrong>"` format so failures are diagnosable without reading source.
- Hardened edge cases: empty prompt strings, guards returning non-`GuardResult` values, eval runs with empty probe/detector lists, severities referencing invalid levels, and concurrent pipeline runs sharing a tracer.
- Public API surface reviewed and frozen — every symbol without an underscore prefix is part of the v1.0 contract.

## [0.81] — Per-Tool Eval Thresholds

### Added
- `eval.garak` and `eval.giskard` config sections that override the top-level `eval` threshold per tool. Garak can run with zero-tolerance while Giskard runs with a lenient threshold inside the same pipeline.
- `SafetyPipeline.eval_threshold_for(tool)` to resolve the effective threshold for a given tool.

## [0.8.0] — Environment Profiles & Eval Thresholds

### Added
- Per-environment config profiles (`environments:` block). A single `config.yaml` defines `dev`, `staging`, and `prod` profiles that override entire top-level sections. Selected via `SafetyPipeline.from_yaml(..., env="prod")` or the `PYNOP_ENV` environment variable.
- `eval:` section with `max_issues`, `ignore_severities`, and `garak_severities` for configurable evaluation pass/fail thresholds.

## [0.7.0] — NeMo Config Builder

### Added
- Inline NeMo rails declared by name in pynop YAML (`jailbreak`, `topic_control`, `content_safety`, `pii`). pynop generates the NeMo `config.yml` and Colang files automatically — users no longer hand-write them.
- `NeMoConfigBuilder` exposed as a public helper for advanced users.

### Changed
- Custom NeMo configs are still supported via `config_path:`. `rails:` and `config_path:` are mutually exclusive on a single guard entry.

## [0.6.0] — Per-Guard Reask

### Added
- `reask` as a per-guard `on_guard_fail` strategy on output guards. On failure, pynop appends the rejection reason to the prompt, re-calls the LLM, and re-runs all output guards. Configurable `max_reask` (default 2) and `reask_instruction` template per guard.

### Changed
- `on_guard_fail` moved from the slot level to individual guard config so a single output slot can mix strategies (e.g. reask for toxicity, reject for PII).

## [0.5.0] — LangChain Multi-Backend & LLM Sharing

### Changed
- LLM layer migrated from the OpenAI SDK to LangChain `BaseChatModel`. Single internal interface unlocks multi-backend support (OpenAI, Anthropic, Google, local OpenAI-compatible servers) and lets NeMo Guardrails share the pipeline's LLM client instead of opening its own.
- `SafetyPipeline` now accepts an optional pre-built `llm: BaseChatModel` argument for users who want to inject their own LangChain model.

## [0.4.0] — Production Observability

### Added
- Configurable rejection strategies per guard slot: `reject`, `return_canned`, `include_reason`.
- Configurable error strategies: `on_guard_error: reject | pass`. Guard crashes either fail the pipeline or skip the guard with a logged warning.
- Public read-only `SafetyPipeline.tracer` accessor.
- Strict config validation at `from_yaml` time.

## [0.3.0] — Pre-Deployment Evaluation

### Added
- `EvalRunner` exposing `run_garak(probes=...)` and `run_giskard(detectors=...)` to probe the full `SafetyPipeline` end-to-end.
- `EvalResult` and `EvalIssue` types for programmatic access to scan output.
- Garak and Giskard runs are traced in Langfuse alongside normal pipeline runs.

## [0.2.0] — Policy Enforcement

### Added
- NeMo Guardrails as a second guard type. NeMo and Guardrails-AI guards are configured side-by-side in the same `guards:` slot via `type:` discriminator.
- Built-in NeMo rails support: jailbreak detection, topic control, content safety.

## [0.1.0] — Minimal Pipeline

### Added
- Async `SafetyPipeline` with input/output Guardrails-AI guards.
- OpenAI LLM call wrapped in Langfuse tracing.
- YAML config loader (`SafetyPipeline.from_yaml(path)`).
- `PipelineResult` and `GuardRejection` exception types.
- Initial test suite with mocked OpenAI, Langfuse, and Guardrails-AI fixtures.
