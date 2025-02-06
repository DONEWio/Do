from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union
from dataclasses import dataclass, field

from donew.new.types import Provision, Model, BROWSE, SEE, NEW
from donew.new.runtime import Runtime
from smolagents import TransformersModel, HfApiModel, LiteLLMModel


T = TypeVar("T", bound="BaseDoer")

@dataclass(frozen=True)
class BaseDoer(ABC):
    """Base class for all doers"""
    _name: str
    _purpose: str
    _model: Optional[Model] = None
    _runtime: Optional[Runtime] = None
    _constraints: Optional[dict] = None
    _provisions: List[Provision] = field(default_factory=list)
    _verify: Optional[Callable[[Any], Any]] = None

    @property
    def model(self) -> Model:
        """Get model from either direct model or agent"""
        if self._model:
            return self._model
        
        raise ValueError("No model available - provide either model or agent")

    @abstractmethod
    def envision(self: T, constraints: dict[str, Any], verify: Optional[Callable[[Any], None]] = None) -> T:
        """Set constraints and return new instance"""
        pass

    @abstractmethod
    def realm(self: T, provisions: List[type[Provision]]) -> T:
        """Set provisions and return new instance"""
        pass

    @abstractmethod
    async def enact(self, task: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Execute a task"""
        pass
