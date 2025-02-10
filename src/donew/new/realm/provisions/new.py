import json
from typing import Optional
from smolagents.tools import Tool
from donew.new.doers import BaseDoer
from opentelemetry import trace
from donew.new.realm.provisions import Provision

from donew.utils import is_pydantic_model, parse_to_pydantic

STATE = {}



class New(Provision):
    name = "new"
    description = """"""
    inputs = {
        "task": {
            "type": "string",
            "description": "Natural language description of the task that adheres to the pupose of the tool",
        }
    }
    output_type = "string"
    superdoer: Optional[BaseDoer] = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.superdoer = kwargs.get("superdoer", None)
        self.name = self.superdoer._name
        self.description = f"This tool has the purpose of {self.superdoer._purpose}"

    def forward(self, task: str):
        """Execute a task with validation and context management"""
        try:
            # Try to get tracer, but don't fail if tracing is not enabled
            try:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(self.name) as span:
                    span.set_attribute("task", task)
                    result = self._execute_task(task)
                    span.set_attribute("result", str(result))
                    return result
            except Exception:  # Tracing not available or failed
                return self._execute_task(task)
                
        except Exception as e:
            return str(e)

    def _execute_task(self, task: str):
        """Internal method to execute the task"""
        # Create and configure the agent
        result = self.superdoer.enact(task)
        if is_pydantic_model(result):
            result = parse_to_pydantic(result, self.superdoer.constraints).model_dump_json()
        elif isinstance(result, dict) or isinstance(result, list):
            result = json.dumps(result)
        elif isinstance(result, str):
            result = result
        else:
            raise ValueError("Result is not a valid pydantic model or dict")
        return f"""
        {self.superdoer._name} has completed the task.
        ---
        {result}
        ---
        """

#