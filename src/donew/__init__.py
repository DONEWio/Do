"""
DoNew
===========

Description of your package.
"""

__version__ = "0.1.9"  # Remember to update this when bumping version in pyproject.toml

from typing import Literal, Optional, Sequence, Union, cast, Any
from donew.new import Model
from donew.see.processors import BaseTarget, KeyValueSection, TableSection
from donew.see.processors.web import WebBrowser, WebProcessor
from donew.new.doers.super import SuperDoer
from donew.utils import run_sync




__all__ = [
    "DO",
    "KeyValueSection",
    "TableSection",
    "BaseTarget",
    "WebBrowser",
    "WebProcessor",
]


class DO:
    

    @staticmethod
    def _sync(coro: Any) -> Any:
        return run_sync(
            coro,
            """It looks like you are using DO's sync API inside an async context.
Please use the async methods (A_browse, A_new) instead.""",
        )



    @staticmethod
    async def A_browse(**kwargs) -> Union[WebBrowser, Sequence[WebBrowser]]:
        """Async version of Browse"""

       
       
        web_processor = WebProcessor(**kwargs)
        result = await web_processor.initialize()
        return result
       

        

    @staticmethod
    def Browse(**kwargs) -> Union[WebBrowser, Sequence[WebBrowser]]:
        """Synchronous Browse operation.

        Args:
            **kwargs: Optional configuration dictionary {headless: bool, chrome_path: str}

        Returns:
            WebBrowser instance or sequence of WebBrowser instances
        """
        return DO._sync(DO.A_browse(**kwargs))

    @staticmethod
    async def A_new(model,**kwargs) -> SuperDoer:
        """Async version of New"""

        if "name" not in kwargs:
            raise ValueError("name is required")
        if "purpose" not in kwargs:
            raise ValueError("purpose is required")
       
       
        return SuperDoer(_model=model, _name=kwargs["name"], _purpose=kwargs["purpose"])

    @staticmethod
    def New(model: Model, **kwargs) -> SuperDoer:
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
        return DO._sync(DO.A_new(model, **kwargs))
