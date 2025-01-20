"""
DoNew
===========

Description of your package.
"""

__version__ = "0.1.5"  # Remember to update this when bumping version in pyproject.toml

from typing import Optional, Sequence, Union, cast, overload, Any
import asyncio
import greenlet
from donew.see.processors import BaseTarget, KeyValueSection, TableSection
from donew.see.processors.web import WebBrowser, WebProcessor
from donew.see import See
from donew.new.doers.super import SuperDoer
from donew.new.runtime import Runtime

__all__ = [
    "DO",
    "KeyValueSection",
    "TableSection",
    "BaseTarget",
    "WebBrowser",
    "WebProcessor",
    "See",
]


class MainGreenlet(greenlet.greenlet):
    def __str__(self) -> str:
        return "<MainGreenlet>"


class DO:
    _global_config = None

    @staticmethod
    def Config(
        headless: bool = True,
    ):
        """Set global configuration for DO class

        Args:
           headless:
        """
        DO._global_config = {
            "headless": headless,
        }

    @staticmethod
    def _sync(coro: Any) -> Any:
        if asyncio.events._get_running_loop() is not None:
            raise RuntimeError(
                """It looks like you are using DO's sync API inside an async context.
Please use the async methods (_browse_async, _new_async) instead."""
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    @staticmethod
    async def _browse_async(
        paths, config: Optional[dict] = None
    ) -> Union[WebBrowser, Sequence[WebBrowser]]:
        """Async version of Browse"""
        c = config if config else DO._global_config
        return cast(
            Union[WebBrowser, Sequence[WebBrowser]], await See(paths=paths, config=c)
        )

    @staticmethod
    def Browse(
        paths, config: Optional[dict] = None
    ) -> Union[WebBrowser, Sequence[WebBrowser]]:
        """Synchronous Browse operation.

        Args:
            paths: URL or list of URLs to browse
            config: Optional configuration dictionary

        Returns:
            WebBrowser instance or sequence of WebBrowser instances
        """
        return DO._sync(DO._browse_async(paths, config))

    @staticmethod
    async def _new_async(config: dict[str, Any]) -> SuperDoer:
        """Async version of New"""
        model = config["model"]
        runtime = Runtime(**config["runtime"]) if "runtime" in config else None
        return SuperDoer(_model=model, _runtime=runtime)

    @staticmethod
    def New(config: dict[str, Any]) -> SuperDoer:
        """Create a new SuperDoer instance for task execution.

        Args:
            config: Configuration dictionary containing:
                - model: Model instance (required)
                - runtime: Runtime configuration (optional)
                    - executor: Executor type (default: "smolagents.local")
                    - workspace: Workspace path (optional)
                - agent: Agent instance (optional)

        Returns:
            SuperDoer instance
        """
        return DO._sync(DO._new_async(config))
