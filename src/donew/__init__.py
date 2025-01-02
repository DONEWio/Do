"""
DoNew
===========

Description of your package.
"""

import tomli
from pathlib import Path


def _get_version():
    try:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            return tomli.load(f)["project"]["version"]
    except Exception:
        return "unknown"


__version__ = _get_version()


from typing import Optional, Sequence, Union, cast, overload
from src.donew.see.processors import BaseTarget
from src.donew.see.processors.web import WebBrowser, WebProcessor
from src.donew.see import See


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
