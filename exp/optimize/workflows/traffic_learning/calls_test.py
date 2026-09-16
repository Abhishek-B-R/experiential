"""Blocking provider ownership survives cancellation and eventual provider failure."""

import asyncio
import threading

import pytest

from exp.optimize.workflows.traffic_learning.calls import run_owned


@pytest.mark.parametrize("provider_fails", [False, True])
@pytest.mark.parametrize("repeated_cancellation", [False, True])
def test_cancellation_joins_dispatched_provider_before_returning(
    provider_fails: bool, repeated_cancellation: bool
) -> None:
    """Neither cancellation nor a later provider error can abandon in-flight work."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def operation() -> str:
        """Hold the owned provider call until the test releases its explicit barrier."""
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("provider fixture was not released")
            if provider_fails:
                raise ValueError("provider failed after cancellation")
            return "complete"
        finally:
            finished.set()

    async def exercise() -> None:
        """Cancel the caller while its dispatched provider remains blocked."""
        task = asyncio.create_task(run_owned(operation))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            await asyncio.sleep(0)
            if repeated_cancellation:
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
            assert not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert finished.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_provider_result_or_error_propagates_without_cancellation() -> None:
    """Uncancelled calls preserve their exact provider outcome."""

    def success() -> str:
        """Return the provider result for the normal completion path."""
        return "complete"

    def failure() -> str:
        """Raise the provider failure for the uncancelled error path."""
        raise ValueError("provider failure")

    async def exercise() -> None:
        """Preserve successful values and ordinary provider exceptions."""
        assert await run_owned(success) == "complete"
        with pytest.raises(ValueError, match="provider failure"):
            await run_owned(failure)

    asyncio.run(exercise())
