from typing import Any, List, Optional
from dataclasses import dataclass, replace

from donew.new.doers import BaseDoer
from donew.new.types import Provision
from smolagents import CodeAgent


@dataclass(frozen=True)
class SuperDoer(BaseDoer):
    """Advanced task execution with constraint validation and context management"""

    def envision(self, constraints: dict[str, Any]) -> "SuperDoer":
        """Return new instance with constraints"""
        return replace(self, _constraints=constraints)

    def realm(self, provisions: List[Provision]) -> "SuperDoer":
        """Return new instance with provisions"""
        return replace(self, _provisions=provisions)

    def enact(self, task: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Execute a task with validation and context management"""

        try:
            # Setup contexts in sequence
            for ctx in self._provisions:
                ctx.setup()

            # Format task with params if provided
            formatted_task = task.format(**params) if params else task

            agent = CodeAgent(tools=[], model=self.model, add_base_tools=False)
            result = agent.run(formatted_task)

            # Validate result if constraints exist
            if self._constraints and "verify" in self._constraints:
                validation = self._constraints["verify"](result)
                if isinstance(validation, str):
                    raise ValueError(validation)
                if not validation:
                    raise ValueError("Task validation failed")

            return result

        finally:
            # Cleanup contexts in reverse order
            for ctx in reversed(self._provisions):
                ctx.cleanup()
