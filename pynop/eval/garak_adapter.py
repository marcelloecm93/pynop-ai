"""Garak adapter — bridges SafetyPipeline to Garak's Generator interface."""

import asyncio

from garak.generators.base import Generator

from pynop.exceptions import GuardRejection


class PipelineGenerator(Generator):
    """Garak Generator that sends prompts through SafetyPipeline."""

    generator_family_name = "pynop"
    supports_multiple_generations = False

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self.name = "SafetyPipeline"
        self.fullname = f"{self.generator_family_name}:{self.name}"
        self.description = "pynop SafetyPipeline generator for Garak"

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> list[str | None]:
        """Call the pipeline, catching GuardRejection as a rejection string."""
        try:
            result = asyncio.run(self._pipeline.run(prompt))
            return [result.output]
        except GuardRejection as e:
            return [f"[REJECTED] {e}"]
        except Exception as e:
            return [f"[ERROR] {e}"]
