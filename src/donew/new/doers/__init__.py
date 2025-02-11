from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, TypeVar
from dataclasses import dataclass, field

from donew.new.realm.provisions import Provision
from donew.new import Model




T = TypeVar("T", bound="BaseDoer")

@dataclass(frozen=True)
class BaseDoer(ABC):
    """Base class for all doers"""
    _name: str
    _purpose: str
    _model: Model
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
    async def enact(self, task: str, **kwargs) -> Any:
        """Execute a task"""
        pass
