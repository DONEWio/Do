import json
from typing import Any, Callable, List, Optional
from dataclasses import dataclass, replace
from pydantic import BaseModel
from donew.new.doers import BaseDoer
from donew.new.types import Provision
from smolagents import CodeAgent
from donew.utils import is_pydantic_model, parse_to_pydantic, pydantic_model_to_simple_schema


@dataclass(frozen=True)
class SuperDoer(BaseDoer):
    """Advanced task execution with constraint validation and context management"""

    def envision(self, constraints: dict[str, Any], verify: Optional[Callable[[Any], Any]] = None) -> "SuperDoer":
        """Return new instance with constraints"""
        return replace(self, _constraints=constraints, _verify=verify)


    def realm(self, provisions: List[Provision]) -> "SuperDoer":
        """Return new instance with provisions"""
        return replace(self, _provisions=provisions)


    def enact(self, task: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Execute a task with validation and context management"""
        try:
            for ctx in self._provisions:
                ctx.setup()

            if self._constraints:
                if is_pydantic_model(self._constraints):
                    constraints_schema = pydantic_model_to_simple_schema(self._constraints)
                    task = task + f"\nYou must answer in the following JSON format:\n---\n{json.dumps(constraints_schema)}"
                else:
                    task = task + f"\nYou must answer in the following format:\n---\n{self._constraints}"

            formatted_task = task.format(**params) if params else task
            
            agent = CodeAgent(tools=[], model=self.model, add_base_tools=False)
            result = agent.run(formatted_task)

            if self._verify:
                result = self._verify(result)
            elif self._constraints and is_pydantic_model(self._constraints):
                result = parse_to_pydantic(result, self._constraints)
                
            return result

        finally:
            for ctx in reversed(self._provisions):
                ctx.cleanup()