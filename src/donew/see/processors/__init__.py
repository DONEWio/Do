from abc import ABC, abstractmethod
from typing import (
    Any,
    Coroutine,
    Dict,
    List,
    Tuple,
    TypeVar,
    Generic,
    Optional,
    Callable,
    Protocol,
    cast,
    Literal,
    TypedDict,
    Union,
    Mapping,
)
from dataclasses import dataclass, field
import inspect
from functools import wraps
import uuid
from tabulate import tabulate
import asyncio

from donew.see.graph import KnowledgeGraph
from donew.utils import run_sync


# Type definitions for state dictionary
class TableSection(TypedDict):
    name: str
    type: Literal["table"]
    headers: List[str]
    rows: List[List[str]]


class KeyValueSection(TypedDict):
    name: str
    type: Literal["keyvalue"]
    data: Mapping[str, Union[str, Mapping[str, str]]]


class StateDict(TypedDict):
    sections: List[Union[TableSection, KeyValueSection]]


T = TypeVar("T")  # Input type
F = TypeVar("F", bound=Callable[..., Any])


class PublicMethod(Protocol):
    _public: bool
    _order: int


def public(order: int = 100):
    """Decorator to mark methods as public API with optional ordering"""

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            async_wrapper._public = True  # type: ignore
            async_wrapper._order = order  # type: ignore
            return cast(F, async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            sync_wrapper._public = True  # type: ignore
            sync_wrapper._order = order  # type: ignore
            return cast(F, sync_wrapper)

    return decorator


def documentation(extends: Callable, template: Optional[str] = None):
    """Decorator to mark methods as manual documentation source with templating."""

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            wrapper = cast(F, async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            wrapper = cast(F, sync_wrapper)

        # Copy public/order metadata if exists
        if hasattr(extends, "_public"):
            wrapper._public = extends._public  # type: ignore
        if hasattr(extends, "_order"):
            wrapper._order = extends._order  # type: ignore

        # Get docstring from extended method
        extended_doc = extends.__doc__ or ""

        # Apply template
        if template:
            wrapper.__doc__ = template.format(extendee=extended_doc.strip())
        else:
            wrapper.__doc__ = extended_doc.strip()

        return wrapper

    return decorator


@dataclass
class BaseTarget:
    """Base class for all targets.

    Targets represent processed data that can be analyzed and queried.
    Each target should implement both sync and async interfaces.
    """

    _annotated_image: str = ""  # base64 encoded
    _raw_image: str = ""  # base64 encoded
    _text_content: List[str] = field(default_factory=list)
    _debug_info: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _kg_analyzer: Optional[KnowledgeGraph] = None

    def _sync(self, coro: Any) -> Any:
        """Run an async operation synchronously.

        This provides sync interface to async methods.
        """
        return run_sync(
            coro,
            """It looks like you are using the sync API inside an async context.
Please use the async methods (a_*) instead.""",
        )

    async def a_analyze(self, **kwargs: Any) -> Dict[str, Any]:
        """Async version of analyze."""
        if self._kg_analyzer is None:
            self._kg_analyzer = KnowledgeGraph()
        text = await self.a_text()
        id = uuid.uuid4()
        return self._kg_analyzer.analyze(id, text, **kwargs)

    def analyze(self, **kwargs: Any) -> Dict[str, Any]:
        """Synchronous analyze operation."""
        return self._sync(self.a_analyze(**kwargs))

    async def a_query(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """Async version of query."""
        return self._kg_analyzer.query(text, params=kwargs.get("params", None))

    def query(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """Synchronous query operation."""
        return self._sync(self.a_query(text, **kwargs))

    def documentation(self) -> List[str]:
        """Returns a list of documentation strings for all public methods in order.

        The documentation includes:
        1. Class docstring (if exists)
        2. All methods marked with @public decorator in specified order
        3. Method docstrings and type hints, including templated documentation from @manual
        """
        docs = []

        # Add class documentation if it exists
        if self.__class__.__doc__:
            docs.append(
                f"# {self.__class__.__name__}\n{self.__class__.__doc__.strip()}\n"
            )

        # Get all public methods
        methods = []
        for name, method in inspect.getmembers(self.__class__):
            if hasattr(method, "_public"):
                methods.append((method._order, name, method))

        # Sort by order
        methods.sort(key=lambda x: x[0])

        # Add method documentation
        for _, name, method in methods:
            signature = inspect.signature(method)

            # Get the docstring, handling both direct and templated docs
            if hasattr(method, "__wrapped__"):
                # For decorated methods, get the processed docstring
                doc = method.__doc__ or "No documentation available"
            else:
                # For regular methods, use the direct docstring
                doc = method.__doc__ or "No documentation available"

            docs.append(f"\n## {name}{signature}\n{doc.strip()}")

        return docs



    @abstractmethod
    def debug(self) -> Dict[str, Any]:
        """Return debug information about the target's processing.

        Returns:
            Dict containing debug information such as processing times,
            intermediate results, and any error messages.
        """
        pass

    def _format_state(self, state_dict: StateDict) -> str:
        """Convert a state dictionary to a formatted string using tabulate.

        This formats nested structures using markdown headers and separate tables
        for better readability and easier parsing.
        """
        output = []

        def format_section(
            name: str, data: Mapping[str, Union[str, Mapping[str, str]]], level: int = 2
        ) -> List[str]:
            """Helper to format a section with proper header level and table."""
            section_output = []
            # Add section header with proper level
            section_output.append(f"{'#' * level} {name}\n")

            # Convert dict to rows and create table
            rows = [[k, str(v)] for k, v in data.items()]
            table = tabulate(rows, headers=["Property", "Value"], tablefmt="pipe")
            section_output.append(table + "\n")
            return section_output

        for section in state_dict["sections"]:
            # Add main section header
            output.append(f"## {section['name']}\n")

            if section["type"] == "table":
                # Format as full table
                table = tabulate(
                    section["rows"], headers=section["headers"], tablefmt="pipe"
                )
                output.append(table + "\n")

            elif section["type"] == "keyvalue":
                # Handle each subsection
                for key, value in section["data"].items():
                    if isinstance(value, dict):
                        # Create a subsection for nested dict
                        output.extend(format_section(key, value, level=3))
                    else:
                        # Single key-value pair
                        table = tabulate(
                            [[key, value]],
                            headers=["Property", "Value"],
                            tablefmt="pipe",
                        )
                        output.append(table + "\n")

        return "\n".join(output)

    async def a_state(self) -> str:
        """Async version of state."""
        return self._format_state(await self.a_get_state_dict())

    def state(self) -> str:
        """Synchronous state operation."""
        return self._sync(self.a_state())

    @abstractmethod
    async def a_get_state_dict(self) -> StateDict:
        """Async version of get_state_dict."""
        pass

    def get_state_dict(self) -> StateDict:
        """Synchronous get_state_dict operation."""
        return self._sync(self.a_get_state_dict())


class BaseProcessor(ABC, Generic[T]):
    """Base class for all processors.

    Processors handle converting input sources into Targets.
    Each processor should implement both sync and async interfaces.
    """

    def _sync(self, coro: Any) -> Any:
        """Run an async operation synchronously.

        This provides sync interface to async methods.
        """
        return run_sync(
            coro,
            """It looks like you are using the sync API inside an async context.
Please use the async methods (a_*) instead.""",
        )

    @abstractmethod
    async def a_process(self, source: T) -> List[BaseTarget]:
        """Async version of process.

        Args:
            source: The input source to process

        Returns:
            List of processed targets
        """
        pass

    def process(self, source: T) -> List[BaseTarget]:
        """Synchronous process operation.

        This provides a sync interface to the async a_process method.
        Do not override this method - implement a_process instead.

        Args:
            source: The input source to process

        Returns:
            List of processed targets
        """
        return self._sync(self.a_process(source))
