import requests
from opentelemetry import trace
from donew.new.assistants import Provision
from pydantic import BaseModel
from donew.utils import parse_to_pydantic, pydantic_model_to_simple_schema
from logging import getLogger

logger = getLogger(__name__)

class RestackWorflowAssistant(Provision):
    name = "restack"
    description = """"""
    base_url = ""
    workflow_id = ""
    input_model = BaseModel
    inputs = {
        "task": {
            "type": "string",
            "description": "Natural language description of the task that adheres to the pupose of the tool",
        }
    }
    output_type = "string"
    def __init__(self, *args, **kwargs):
       
       
        self.name = kwargs.get("name", "restack")
        self.description = kwargs.get("description", "This tool has the purpose of running a workflow hosted on a remote server")
        self.base_url = kwargs.get("base_url", "")
        self.workflow_id = kwargs.get("workflow_id", "")
        self.input_model = kwargs.get("input_model", BaseModel)
        self.inputs = {
            "input": {
                "type": "object",
                "description": "The input that must match the required schema",
                **pydantic_model_to_simple_schema(self.input_model),
            }
        }
        self.timeout = kwargs.get("timeout", 30)
        super().__init__(*args, **kwargs)
    def run(self, input: dict):
        url = f"{self.base_url}/api/workflows/{self.workflow_id}"
        payload = {
            "input": input,
            "schedule": None
        }
        post_response = requests.post(url, json=payload)
        post_data = post_response.json()
        run_id = post_data.get("runId")
        workflow_id = post_data.get("workflowId")
        if not run_id:
            return post_data
        response_url = f"{self.base_url}/api/workflows/{self.workflow_id}/{workflow_id}/{run_id}"
        try:
            response = requests.get(response_url, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Workflow polling timed out after {self.timeout} seconds")
        response_data = response.json()
        return response_data
    def forward(self, input):
        """Execute a task with validation and context management"""
        try:
            task_params = parse_to_pydantic(input, self.input_model)
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")

        try:
            # Try to get tracer, but don't fail if tracing is not enabled
            try:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(self.name) as span:
                    span.set_attribute("task", input)
                    
                    result = self.run(input)
                    span.set_attribute("result", str(result))
                    return result
            except Exception:  # Tracing not available or failed
                logger.warning("Tracing not available or failed")
                return self.run(input)
                
        except Exception as e:
            return str(e)

    

