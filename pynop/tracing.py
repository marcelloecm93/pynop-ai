"""Langfuse tracing wrapper."""


class Tracer:
    """Thin wrapper around Langfuse SDK."""

    def __init__(
        self,
        enabled: bool = False,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ):
        self.enabled = enabled
        self._client = None
        if enabled and public_key and secret_key:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url,
            )

    @property
    def client(self):
        """The underlying Langfuse client, or None if tracing is disabled."""
        return self._client

    def start_trace(self, prompt: str):
        """Open a new trace (root span). Returns None if disabled."""
        if not self.enabled or not self._client:
            return None
        return self._client.start_observation(name="pipeline", input=prompt)

    def end_trace(self, trace, output: str | None = None):
        """Close a trace with final output."""
        if trace is None:
            return
        trace.update(output=output)
        trace.end()

    def start_span(self, trace, name: str, as_type: str = "span"):
        """Open a child span within a trace. Returns None if disabled."""
        if trace is None:
            return None
        return trace.start_observation(name=name, as_type=as_type)

    def end_span(self, span, metadata: dict | None = None):
        """Close a span with metadata."""
        if span is None:
            return
        span.update(metadata=metadata)
        span.end()

    def flush(self):
        """Flush pending events to Langfuse. Call before process exit."""
        if self._client:
            self._client.flush()
