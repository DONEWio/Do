"""
DoNew
===========

Description of your package.
"""

__version__ = "0.1.0"


from typing import Optional, Sequence, Union, cast, overload
from src.DoNew.see.processors import BaseTarget
from src.DoNew.see.processors.web import WebBrowser, WebProcessor
from src.DoNew.see import See


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
