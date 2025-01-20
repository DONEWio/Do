"""Utility functions for DoNew."""

import asyncio
from typing import Any, Optional


def run_sync(coro: Any, error_message: Optional[str] = None) -> Any:
    """Run an async operation synchronously.

    This provides sync interface to async methods.

    Args:
        coro: The coroutine to run
        error_message: Optional custom error message for async context detection

    Returns:
        The result of the coroutine

    Raises:
        RuntimeError: If called from an async context
    """
    if asyncio.events._get_running_loop() is not None:
        raise RuntimeError(
            error_message
            or "Cannot use sync API inside an async context. Use async methods instead."
        )

    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        should_close = True
    else:
        should_close = False

    try:
        return loop.run_until_complete(coro)
    finally:
        if should_close:
            loop.close()
            asyncio.set_event_loop(None)
