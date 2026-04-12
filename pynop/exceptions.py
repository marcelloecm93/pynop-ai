"""Pipeline exceptions."""


class GuardRejection(Exception):
    """Raised when a guard rejects input or output.

    Raised by ``SafetyPipeline.run`` when a guard fails under the ``reject`` or
    ``include_reason`` strategy, or after a ``reask`` guard exhausts its
    ``max_reask`` retries.

    Attributes:
        reason: The guard's rejection reason, when available. The ``include_reason``
            strategy also embeds it in the exception message; ``reject`` keeps the
            user-facing message generic and exposes the reason only on this attribute.
    """

    def __init__(self, message: str = "Invalid input detected", reason: str | None = None):
        super().__init__(message)
        self.reason = reason
