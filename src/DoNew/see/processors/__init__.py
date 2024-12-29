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
)
from dataclasses import dataclass, field
import inspect
from functools import wraps
from tabulate import tabulate

T = TypeVar("T")  # Input type


def public(order: int = 100):
    """Decorator to mark methods as public API with optional ordering"""

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            # If it's already async, wrap it preserving async
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            async_wrapper._public = True
            async_wrapper._order = order
            return async_wrapper
        else:
            # If it's not async, don't make it async
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            sync_wrapper._public = True
            sync_wrapper._order = order
            return sync_wrapper

    return decorator


def manual(template: str, extends: Callable):
    """Decorator to mark methods as manual documentation source with templating."""

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            # If it's already async, wrap it preserving async
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            wrapper = async_wrapper
        else:
            # If it's not async, don't make it async
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper = sync_wrapper

        # Copy public/order metadata if exists
        if hasattr(extends, "_public"):
            wrapper._public = extends._public
        if hasattr(extends, "_order"):
            wrapper._order = extends._order

        # Get docstring from extended method
        extended_doc = extends.__doc__ or ""

        # Apply template
        wrapper.__doc__ = template.format(extendee=extended_doc.strip())

        return wrapper

    return decorator


@dataclass
class BaseTarget:
    """Base class for all targets"""

    _annotated_image: str = ""  # base64 encoded
    _raw_image: str = ""  # base64 encoded
    _text_content: List[str] = field(default_factory=list)
    _debug_info: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)

    def manuals(self) -> List[str]:
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

    @public(order=1)
    def get_metadata(self) -> Dict[str, Any]:
        """Get target metadata.

        Returns:
            Dict containing metadata about the target including any custom fields
            set during processing.
        """
        return self._metadata

    @abstractmethod
    @public(order=2)
    def debug(self) -> Dict[str, Any]:
        """Return debug information about the target's processing.

        Returns:
            Dict containing debug information such as processing times,
            intermediate results, and any error messages.
        """
        pass

    def _format_state(self, state_dict: Dict[str, Any]) -> str:
        """Convert a state dictionary to a formatted string using tabulate.

        This formats nested structures using markdown headers and separate tables
        for better readability and easier parsing.
        """
        output = []

        def format_section(
            name: str, data: Dict[str, Any], level: int = 2
        ) -> List[str]:
            """Helper to format a section with proper header level and table."""
            section_output = []
            # Add section header with proper level
            section_output.append(f"{'#' * level} {name}\n")

            # Convert dict to rows and create table
            rows = [[k, v] for k, v in data.items()]
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

    async def state(self) -> str:
        """Get a formatted string describing the current state.
        Each processor must implement _get_state_dict()."""
        return self._format_state(await self._get_state_dict())

    @abstractmethod
    async def _get_state_dict(self) -> Coroutine[Dict[str, Any], Any, None]:
        """Get a dictionary containing the current state.

        Must be implemented by each processor to provide their specific
        state information in the format expected by _format_state().
        """
        pass


class BaseProcessor(ABC, Generic[T]):
    """Base class for all processors"""

    @abstractmethod
    async def process(self, source: T) -> List[BaseTarget]:
        """Process the input and return list of targets"""
        pass
