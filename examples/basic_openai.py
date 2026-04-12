"""Minimal pynop example: load a config, run a prompt, print the answer.

Prerequisites:
    pip install pynop-ai
    guardrails hub install hub://guardrails/detect_pii
    export OPENAI_API_KEY=sk-...

Run:
    python examples/basic_openai.py
"""

import asyncio
import os
import sys

from pynop import GuardRejection, SafetyPipeline


async def main() -> int:
    config_path = os.path.join(os.path.dirname(__file__), "basic_config.yaml")
    pipeline = SafetyPipeline.from_yaml(config_path)

    prompts = [
        "Explain what a Python decorator is in two sentences.",
        "My credit card 4111-1111-1111-1111 expired, what should I do?",
    ]

    for prompt in prompts:
        print(f"\n>>> {prompt}")
        try:
            result = await pipeline.run(prompt)
            print(result.output)
        except GuardRejection as exc:
            print(f"[guard rejected] {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
