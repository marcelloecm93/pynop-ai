"""Tests for Langfuse tracing wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pynop.tracing import Tracer


class TestTracerEnabled:
    """Tests when Langfuse credentials are provided."""

    @patch("langfuse.Langfuse", create=True)
    def test_create_tracer_with_credentials(self, mock_langfuse):
        tracer = Tracer(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
        )
        assert tracer.enabled is True

    @pytest.mark.asyncio
    async def test_start_and_end_trace(self):
        with patch("langfuse.Langfuse", create=True) as mock_langfuse:
            mock_client = mock_langfuse.return_value
            tracer = Tracer(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test",
            )
            mock_trace = MagicMock()
            mock_trace.id = "trace-abc"
            mock_client.start_observation.return_value = mock_trace

            trace = tracer.start_trace(prompt="Hello")
            assert trace is not None

            tracer.end_trace(trace, output="World")

    @pytest.mark.asyncio
    async def test_start_and_end_span(self):
        with patch("langfuse.Langfuse", create=True) as mock_langfuse:
            mock_client = mock_langfuse.return_value
            tracer = Tracer(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test",
            )

            mock_trace = MagicMock()
            mock_span = MagicMock()
            mock_trace.start_observation.return_value = mock_span

            span = tracer.start_span(mock_trace, name="input_guard")
            assert span is not None

            tracer.end_span(span, metadata={"passed": True})

    @pytest.mark.asyncio
    async def test_trace_returns_id(self):
        with patch("langfuse.Langfuse", create=True) as mock_langfuse:
            mock_client = mock_langfuse.return_value
            tracer = Tracer(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test",
            )

            mock_trace = MagicMock()
            mock_trace.id = "trace-xyz"
            mock_client.start_observation.return_value = mock_trace

            trace = tracer.start_trace(prompt="Test")
            assert trace.id == "trace-xyz"


class TestTracerDisabled:
    """Tests when tracing is disabled (no credentials)."""

    def test_create_disabled_tracer(self):
        tracer = Tracer(enabled=False)
        assert tracer.enabled is False

    def test_start_trace_returns_none_when_disabled(self):
        tracer = Tracer(enabled=False)
        trace = tracer.start_trace(prompt="Hello")
        assert trace is None

    def test_start_span_returns_none_when_disabled(self):
        tracer = Tracer(enabled=False)
        span = tracer.start_span(None, name="guard")
        assert span is None

    def test_end_trace_noop_when_disabled(self):
        tracer = Tracer(enabled=False)
        tracer.end_trace(None, output="test")  # should not raise

    def test_end_span_noop_when_disabled(self):
        tracer = Tracer(enabled=False)
        tracer.end_span(None, metadata={})  # should not raise


class TestTracerAsType:
    """Tests for observation type tagging."""

    @pytest.mark.asyncio
    async def test_start_span_forwards_as_type(self):
        with patch("langfuse.Langfuse", create=True):
            tracer = Tracer(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test",
            )

            mock_trace = MagicMock()
            mock_span = MagicMock()
            mock_trace.start_observation.return_value = mock_span

            span = tracer.start_span(mock_trace, name="guard", as_type="guardrail")
            mock_trace.start_observation.assert_called_once_with(name="guard", as_type="guardrail")

    @pytest.mark.asyncio
    async def test_start_span_defaults_to_span_type(self):
        with patch("langfuse.Langfuse", create=True):
            tracer = Tracer(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test",
            )

            mock_trace = MagicMock()
            mock_trace.start_observation.return_value = MagicMock()

            tracer.start_span(mock_trace, name="test")
            mock_trace.start_observation.assert_called_once_with(name="test", as_type="span")
