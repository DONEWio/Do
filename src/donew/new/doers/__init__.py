from abc import ABC, abstractmethod
from typing import Any, List, Optional, TypeVar
from dataclasses import dataclass, field

from donew.new.types import Provision, Model
from donew.new.runtime import Runtime
from smolagents import TransformersModel, HfApiModel, LiteLLMModel


T = TypeVar("T", bound="BaseDoer")


@dataclass(frozen=True)
class BaseDoer(ABC):
    """Base class for all doers"""

    _model: Optional[Model] = None
    _agent: Optional[TransformersModel | HfApiModel | LiteLLMModel] = None
    _runtime: Optional[Runtime] = None
    _constraints: Optional[dict] = None
    _provisions: List[Provision] = field(default_factory=list)

    @property
    def model(self) -> Model:
        """Get model from either direct model or agent"""
        if self._model:
            return self._model
        if self._agent and self._agent.model:
            return self._agent.model
        raise ValueError("No model available - provide either model or agent")

    @abstractmethod
    def envision(self: T, constraints: dict[str, Any]) -> T:
        """Set constraints and return new instance"""
        pass

    @abstractmethod
    def realm(self: T, provisions: List[Provision]) -> T:
        """Set provisions and return new instance"""
        pass

    @abstractmethod
    async def enact(self, task: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Execute a task"""
        pass
