"""Owned blocking provider calls for traffic synthesis, practice, and evaluation."""

import asyncio
from collections.abc import Callable


async def run_owned[T](operation: Callable[[], T]) -> T:
    """Join dispatched work before propagating cancellation, including repeated requests.

    Provider clients enforce finite transport deadlines. A cancelled caller still
    owns the dispatched call until it finishes, and its cancellation takes priority
    over an eventual ordinary provider error.
    """
    task = asyncio.create_task(asyncio.to_thread(operation))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = cancellation or error
        except Exception:
            if cancellation is not None:
                raise cancellation from None
            raise
        else:
            if cancellation is not None:
                raise cancellation
            return result
