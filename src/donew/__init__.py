"""
DoNew
===========

Description of your package.
"""

__version__ = "0.1.5"  # Remember to update this when bumping version in pyproject.toml

from typing import Optional, Sequence, Union, cast, overload
from donew.see.processors import BaseTarget, KeyValueSection, TableSection
from donew.see.processors.web import WebBrowser, WebProcessor
from donew.see import See

__all__ = [
    "DO",
    "KeyValueSection",
    "TableSection",
    "BaseTarget",
    "WebBrowser",
    "WebProcessor",
    "See",
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

    @overload
    @staticmethod
    async def Browse(
        paths: str, config: Optional[dict] = None
    ) -> WebBrowser:  # single path = single result
        ...

    @overload
    @staticmethod
    async def Browse(
        paths: Sequence[str], config: Optional[dict] = None
    ) -> Sequence[WebBrowser]:  # multiple paths = sequence
        ...

    @staticmethod
    async def Browse(
        paths, config: Optional[dict] = None
    ) -> Union[WebBrowser, Sequence[WebBrowser]]:
        c = config if config else DO._global_config
        return cast(
            Union[WebBrowser, Sequence[WebBrowser]], await See(paths=paths, config=c)
        )
