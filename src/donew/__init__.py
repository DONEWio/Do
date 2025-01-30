"""
DoNew
===========

Description of your package.
"""

__version__ = "0.1.5"  # Remember to update this when bumping version in pyproject.toml

from typing import Literal, Optional, Sequence, Union, cast, Any
from donew.see.processors import BaseTarget, KeyValueSection, TableSection
from donew.see.processors.web import WebBrowser, WebProcessor
from donew.see import See
from donew.new.doers.super import SuperDoer
from donew.new.runtime import Runtime
from donew.utils import run_sync
from donew.new.types import BROWSE, SEE, NEW, Provision



__all__ = [
    "DO",
    "KeyValueSection",
    "TableSection",
    "BaseTarget",
    "WebBrowser",
    "WebProcessor",
    "See",
    "BROWSE",  # Add these to __all__ so they're available when importing
    "SEE",
    "NEW",
    "Provision",
]


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
        return run_sync(
            coro,
            """It looks like you are using DO's sync API inside an async context.
Please use the async methods (A_browse, A_new) instead.""",
        )

    @staticmethod
    async def A_documentation(target: Literal["browse"]):
        docs = []
        if target == "browse":
            browser = await See("https://documentation/request")
            docs.extend(browser.documentation())
        else:
            raise ValueError(f"Invalid target: {target}")
        docs = "\n".join(docs)
        return docs

    @staticmethod
    def Documentation(target: Literal["browse"]):
        return DO._sync(DO.A_documentation(target))

    @staticmethod
    async def A_browse(
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
        return DO._sync(DO.A_browse(paths, config))

    @staticmethod
    async def A_new(config: dict[str, Any]) -> SuperDoer:
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
        return DO._sync(DO.A_new(config))
