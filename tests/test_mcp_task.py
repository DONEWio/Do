from donew.new.realm.provisions.mcprun import MCPRun
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

def test_mcp_task_init():
    profile = os.getenv("MCP_PROFILE")
    task = os.getenv("MCP_TASK")
    class InputSchema(BaseModel):
        url: str = Field(description="The url to fetch")
    mcp = MCPRun(profile=profile, task=task, input_model=InputSchema)
    assert mcp.description is not None
    # assert mcp.inputs is not None # HELP?!
    assert mcp.name is not None

def test_mcp_task_run():

    profile = os.getenv("MCP_PROFILE")
    task = os.getenv("MCP_TASK")
    
    class InputSchema(BaseModel):
        url: str = Field(description="The url to fetch")
        
    mcp = MCPRun(profile=profile, task=task, input_model=InputSchema)
    result = mcp.forward({"url": "https://mcp.run"})
    assert result is not None
    