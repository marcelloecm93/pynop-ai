# PyNOP AI

[![tests](https://github.com/marcelloecm93/pynop/actions/workflows/test.yml/badge.svg)](https://github.com/marcelloecm93/pynop/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pynop-ai.svg)](https://pypi.org/project/pynop-ai/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Async LLM safety pipeline with input/output guardrails and observability tracing.

## Install

pynop is published on PyPI under the distribution name **`pynop-ai`**. The Python import name is **`pynop`**.

```bash
pip install pynop-ai
# or
uv add pynop-ai
```

```python
import pynop
from pynop import SafetyPipeline
```

The core install ships with OpenAI support, Guardrails-AI, and Langfuse tracing. Additional providers and tools are available as optional extras:

| Extra | Adds | Use when |
| --- | --- | --- |
| `pynop-ai[anthropic]` | `langchain-anthropic` | You configure `provider: anthropic` in YAML |
| `pynop-ai[google]` | `langchain-google-genai` | You configure `provider: google` in YAML |
| `pynop-ai[nemo]` | `nemoguardrails` | You add a `type: nemo` guard |
| `pynop-ai[eval]` | `garak`, `giskard` | You call `EvalRunner.run_garak` / `run_giskard` |
| `pynop-ai[all]` | All of the above | You want everything |

```bash
pip install "pynop-ai[anthropic,nemo]"
# or, install everything
uv add "pynop-ai[all]"
```

Pynop imports the optional dependencies lazily — picking a provider or tool you didn't install raises a clear `ModuleNotFoundError` at `from_yaml` / `run_*` time.

## Setup

### LLM provider

pynop requires an API key from your chosen LLM provider. Sign up and obtain a key from one of:
- [OpenAI](https://platform.openai.com)
- [Anthropic](https://console.anthropic.com)
- [Google AI Studio](https://aistudio.google.com)
- Or run a local server (Ollama, vLLM, LM Studio) — no key required

**pynop does not cap or monitor LLM spend.** Every pipeline run, and each reask retry, incurs token costs. Cost management is the user's responsibility.

### Guardrails-AI validators

Validators must be installed before use via the Guardrails Hub CLI:

```bash
guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/toxic_language
```

Browse available validators at [hub.guardrailsai.com](https://hub.guardrailsai.com). Any validator referenced in your config that is not installed will cause an `AttributeError` at pipeline startup.

Pin validators to a specific version to avoid silent behavioral changes when a validator package updates:

```bash
guardrails hub install "hub://guardrails/detect_pii~=1.4"
```

Validators do not update automatically. To update to the latest version, run:

```bash
guardrails hub install hub://guardrails/detect_pii --upgrade
```

### Langfuse (tracing)

Tracing requires a Langfuse instance. Sign up at [langfuse.com](https://langfuse.com) or self-host. Obtain a public key and secret key from your project settings.

### Environment variables

Set the required env vars before running pynop. Missing vars referenced in config raise a `ValueError` at startup:

```bash
export OPENAI_API_KEY=sk-...          # or your provider's key
export LANGFUSE_PUBLIC_KEY=pk-...     # if tracing is enabled
export LANGFUSE_SECRET_KEY=sk-...     # if tracing is enabled
```

---

## Usage

```python
import asyncio
from pynop import SafetyPipeline

async def main():
    pipeline = SafetyPipeline.from_yaml("config.yaml")
    result = await pipeline.run("Summarize this document for me.")
    print(result.output)

    # Select an environment profile
    pipeline = SafetyPipeline.from_yaml("config.yaml", env="prod")

asyncio.run(main())
```

## Config

See `config.yaml` for the default configuration. Supports:

- **LLM**: Multi-backend via LangChain — OpenAI, Anthropic, Google, and local (Ollama/vLLM/LM Studio)
- **Guards**: Guardrails-AI validators (PII, schema) and NeMo Guardrails (jailbreak, content safety) — configurable per input/output slot, run in config order
- **Tracing**: Langfuse observability (optional, auto-reads env vars)
- **Eval thresholds**: Configurable pass/fail criteria for evaluation runs
- **Environment profiles**: Per-environment config overrides (dev, staging, prod)

### LLM providers

```yaml
# OpenAI
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}

# Anthropic
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}

# Google Gemini
llm:
  provider: google
  model: gemini-2.0-flash
  api_key: ${GOOGLE_API_KEY}

# Local (OpenAI-compatible server — Ollama, vLLM, LM Studio)
llm:
  provider: local
  model: llama3
  base_url: http://localhost:11434/v1
  api_key: not-needed
```

You can also pass a pre-built LangChain `BaseChatModel` directly to the constructor (skipping `from_yaml`):

```python
from langchain_openai import ChatOpenAI
from pynop import SafetyPipeline
from pynop.tracing import Tracer
from pynop.types import GuardSlot

custom_llm = ChatOpenAI(model="gpt-4o", temperature=0)

pipeline = SafetyPipeline(
    llm_config={"provider": "openai", "model": "gpt-4o"},
    input_slot=GuardSlot(),     # add guards if you want input validation
    output_slot=GuardSlot(),    # add guards if you want output validation
    tracer=Tracer(enabled=False),
    llm=custom_llm,
)
```

### Guard slots

Each guard slot (input/output) supports configurable rejection and error strategies:

```yaml
guards:
  input:
    on_guard_fail: reject           # reject | return_canned | include_reason
    on_guard_error: reject          # reject | pass
    canned_response: "I can't process that request."  # required for return_canned
    guards:
      - type: guardrails_ai
        validators:
          - name: DetectPII
            on_fail: exception
      - type: nemo
        config_path: ./nemo_configs/input_rails
```

**`on_guard_fail`** — what happens when a guard rejects input/output. Set at slot level as a default; individual guards can override:
- `reject` (default): raise `GuardRejection` with generic message
- `return_canned`: return a `PipelineResult` with the `canned_response` string, skip LLM call
- `include_reason`: raise `GuardRejection` with the guard's rejection reason attached
- `reask` (output guards only): re-call the LLM with the rejection reason appended, then re-run all output guards. Falls back to `reject` after `max_reask` retries (default: 2)

```yaml
guards:
  output:
    on_guard_fail: reject                # slot default
    guards:
      - type: guardrails_ai
        on_guard_fail: reask             # per-guard override
        max_reask: 3
        reask_instruction: "Your response was flagged: {reason}. Rewrite it."
        validators:
          - name: ToxicLanguage
            on_fail: exception
      - type: guardrails_ai
        # inherits slot default: reject
        validators:
          - name: DetectPII
            on_fail: exception
```

Guard ordering matters when mixing strategies — guards run in config order and stop at the first failure.

**`on_guard_error`** — what happens when a guard crashes (unexpected exception):
- `reject` (default): treat the error as a guard failure (applies `on_guard_fail` strategy)
- `pass`: log the error, skip the failed guard, continue to next guard

### NeMo Guardrails

NeMo guards can be configured in two ways:

**Inline rails** (recommended) — declare rails by name directly in config. pynop generates the NeMo config automatically. Built-in NeMo rails (jailbreak, content safety, PII) are referenced directly; parameterized rails (topic control) accept custom parameters:

```yaml
guards:
  input:
    guards:
      - type: nemo
        rails:
          - jailbreak
          - topic_control:
              allowed: [coding, data science]
              denied: [politics, violence]
  output:
    guards:
      - type: nemo
        rails:
          - content_safety
          - pii
```

**Custom config directory** — for rails that require custom Colang flows, point to a directory containing a `config.yml` and `.co` files:

```yaml
      - type: nemo
        config_path: ./my_custom_rails
```

`rails` and `config_path` are mutually exclusive on a single guard entry. See `nemo_configs/` for custom config examples.

### Environment profiles

Define per-environment overrides in the `environments:` section. Each profile replaces entire top-level sections (no deep merge). Sections not defined in a profile fall through to the base config.

```yaml
eval:
  max_issues: 0

environments:
  dev:
    tracing:
      enabled: false
    eval:
      max_issues: 10
      ignore_severities: [minor]
  prod:
    eval:
      max_issues: 0
```

Select an environment via the `env` parameter or the `PYNOP_ENV` env var:

```python
pipeline = SafetyPipeline.from_yaml("config.yaml", env="dev")
# or: export PYNOP_ENV=dev
```

### Eval thresholds

The `eval:` section configures pass/fail criteria for evaluation runs:

```yaml
eval:
  max_issues: 0                # maximum issues before failing (default: 0)
  ignore_severities: [minor]   # exclude these severity levels from the count
  garak_severities:            # map Garak probe families to severity levels
    dan: major
    glitch: minor
    # unlisted probes default to major
```

Severity levels are `major`, `medium`, and `minor`. Without an `eval:` section, the default is zero-tolerance (any issue fails).

#### Per-tool thresholds

Garak and Giskard can have different thresholds within the same pipeline. Add a `garak:` or `giskard:` block under `eval:` — each block inherits from the top-level defaults and only overrides the keys you set:

```yaml
eval:
  max_issues: 0
  ignore_severities: [minor]   # default applied to both tools
  garak:
    max_issues: 0              # zero tolerance for vulnerability scans
    ignore_severities: []      # don't ignore minor either
  giskard:
    max_issues: 3              # lenient for quality checks
    ignore_severities: [minor]
```

Use `pipeline.eval_threshold_for("garak")` (or `"giskard"`) to inspect the resolved threshold from Python. `EvalRunner` uses the per-tool threshold automatically when computing `EvalResult.passed`.

## Evaluation

Run pre-deployment security evaluations using Garak and Giskard:

```python
from pynop import SafetyPipeline
from pynop.eval import EvalRunner

pipeline = SafetyPipeline.from_yaml("config.yaml")
runner = EvalRunner(pipeline)

# Garak vulnerability scan
garak_result = await runner.run_garak(probes=["dan", "promptinject"])
print(garak_result.summary)
print(garak_result.passed)    # uses the eval threshold from config

# Giskard quality scan
giskard_result = await runner.run_giskard(detectors=["prompt_injection"])
print(giskard_result.summary)
print(giskard_result.issues)
```

Both tools evaluate the full pipeline (guards + LLM). Results are traced in Langfuse when tracing is enabled.

Before running evaluations, review the available probe families and detectors to determine which are relevant to your use case:
- [Garak probe catalog](https://docs.garak.ai/garak/probes)
- [Giskard detector catalog](https://docs.giskard.ai/en/stable/open_source/scan/scan_llm/index.html)

**pynop does not provide CI/CD integration.** `EvalRunner` returns a Python result object — wiring evaluations into a CI pipeline (e.g. failing a build on low scores) is the user's responsibility.

## Latency benchmarking

`LatencyBenchmark` compares the per-guard latency of two pipeline configurations side-by-side. It runs a prompt set through both pipelines, fetches the resulting traces from Langfuse, and reports per-span p50/p95/p99.

```python
from pynop import LatencyBenchmark, SafetyPipeline

baseline = SafetyPipeline.from_yaml("config.baseline.yaml")
candidate = SafetyPipeline.from_yaml("config.candidate.yaml")

benchmark = LatencyBenchmark(baseline, candidate, label_a="baseline", label_b="candidate")
report = await benchmark.run([
    "Summarize this document.",
    "Explain quantum computing in one paragraph.",
    "Write a haiku about CI pipelines.",
])

for span in report.stats_a:
    print(f"{span.name:30s} p50={span.p50:.3f}s p95={span.p95:.3f}s")
print(f"baseline total p95: {report.total_a.p95:.3f}s")
print(f"candidate total p95: {report.total_b.p95:.3f}s")
```

`LatencyBenchmark` requires both pipelines to have Langfuse tracing enabled — it reads the per-span latency from Langfuse rather than instrumenting timers itself.

## Integration testing

The default `uv run pytest` command runs the unit suite with mocked OpenAI, Langfuse, and Guardrails-AI. End-to-end integration tests live in `tests/integration/` and are **opt-in** — they hit real OpenAI, Garak, Giskard, and Langfuse, so they require API keys and network access.

Enable them by setting `PYNOP_INTEGRATION=1`:

```bash
export PYNOP_INTEGRATION=1
export OPENAI_API_KEY=sk-...
export LANGFUSE_PUBLIC_KEY=pk-...
export LANGFUSE_SECRET_KEY=sk-...
uv run pytest tests/integration/
```

Without `PYNOP_INTEGRATION=1`, every test in `tests/integration/` is skipped — safe to run on a developer laptop or in PR-level CI.

## Development

```bash
uv sync
uv run pytest
```
