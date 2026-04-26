"""Integration test gating.

Skips every test under tests/integration unless the required env vars are set.
Centralized here so individual test modules don't need to repeat skipif markers
(and can't drift out of sync with the workflow's required secrets).
"""

import os

import pytest

REQUIRED_ENV = ("OPENAI_API_KEY",)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PYNOP_INTEGRATION") != "1":
        skip = pytest.mark.skip(reason="Integration tests require PYNOP_INTEGRATION=1")
        for item in items:
            item.add_marker(skip)
        return

    missing = [name for name in REQUIRED_ENV if not (os.environ.get(name) or "").strip()]
    if missing:
        skip = pytest.mark.skip(reason=f"Missing required env var(s): {', '.join(missing)}")
        for item in items:
            item.add_marker(skip)
