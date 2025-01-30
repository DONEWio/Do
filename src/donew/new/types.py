from typing import Protocol, Final, Literal, Union


class Model(Protocol):
    """Protocol for model interfaces"""

    async def generate(self, prompt: str, **kwargs) -> str: ...


class Provision:
    """Represents a target type in the DO system.
    Each instance is a unique symbol that can't be confused with strings or integers."""
    def __init__(self, name: str) -> None:
        self._name: str = name
    
    def __str__(self) -> str:
        return self._name.lower()
    
    def __repr__(self) -> str:
        return f"Provision({self._name})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Provision):
            return False
        return self._name == other._name

# Create singleton instances as Final types
BROWSE: Final[Provision] = Provision("BROWSE")
SEE: Final[Provision] = Provision("SEE")
NEW: Final[Provision] = Provision("NEW")

# Type alias for type hints - using Union since we can't use Literal with custom classes
ProvisionType = Union[
    Literal["browse", "see", "new"],  # string literals for runtime type checking
    Provision  # actual class type for static type checking
]

__all__ = [
    "Model",
    "Provision",
    "BROWSE",
    "SEE",
    "NEW",
    "ProvisionType",
]
