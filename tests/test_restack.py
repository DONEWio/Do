import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from donew.new.realm.provisions.restack import RestackWorkflow

def test_restack_task():
    #ensure you are running restack server locally
    # ```bash
    # docker run -d --pull always --name restack -p 5233:5233 -p 6233:6233 -p 7233:7233 ghcr.io/restackio/restack:main
    # ```
    # go to the folder of the workflow:
    # ```bash
    # cd <path-to-workflow>
    # ```
    # run uv 
    # ```bash
    # uv sync
    # uv run dev
    # ```
    
    # Input model for the workflow. 
    class Input(BaseModel):
        load_dotenv()
        name: str = Field(description="The name of the person")
        location: str = Field(description="The location to get the weather for")
    
    assistant = RestackWorkflow(
        # the base url of the restack server typically: http://localhost:5233
        base_url=os.getenv("RESTACK_WEATHER_BASE_URL"), 
        # the id of the workflow. grab it from the restack server ui (aka the url of the restack base url)
        # e.g. for api/workflows/MultistepWorkflow workflow_id is MultistepWorkflow
        workflow_id=os.getenv("RESTACK_WEATHER_WORKFLOW_ID"),
        input_model=Input,
        name="weather_assistant",
        description="Retrieve the weather and returns a greeting for a person",
        timeout=90
    )
    result = assistant.forward({"name": "Kenan Deniz", "location": "Istanbul"})
    print(result)