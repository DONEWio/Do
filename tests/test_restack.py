from pydantic import BaseModel, Field
from donew.new.assistants.restack import RestackWorflowAssistant

def test_restack_task():
 
    class Input(BaseModel):
        name: str = Field(description="The name of the person")
    # Pre-create a cookiejar.json file in the temporary directory with a dummy "sessionId"
    assistant = RestackWorflowAssistant(
        base_url="https://reff9k1p.clj5khk.gcp.restack.it",
        workflow_id="MultistepWorkflow",
        input_model=Input,
        name="weather_assistant",
        description="Retrieve the weather and returns a greeting for a person",
        timeout=90
    )
    result = assistant.forward({"name": "Kenan Deniz"})
    print(result)