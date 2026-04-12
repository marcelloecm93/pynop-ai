"""Guard protocol definition."""

from typing import Protocol, runtime_checkable

from pynop.types import GuardResult


@runtime_checkable
class Guard(Protocol):
    """Structural protocol every guard must satisfy.

    Any class with an ``async validate(text: str) -> GuardResult`` method is
    treated as a ``Guard`` — no inheritance required. Implement this protocol
    to plug a custom validator into a ``GuardSlot``.
    """

    async def validate(self, text: str) -> GuardResult: ...
