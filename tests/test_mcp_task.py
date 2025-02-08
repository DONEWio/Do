from donew.new.assistants.mcprun import MCPRun
from pydantic import BaseModel, Field

def test_mcp_task_init():
    mcp = MCPRun(model="gpt-4o-mini", profile="pachacamac/default", task="FetchWeb").init()
    assert mcp.description is not None
    # assert mcp.inputs is not None # HELP?!
    assert mcp.name is not None

def test_mcp_task_run():
    class InputSchema(BaseModel):
        url: str = Field(description="The url to fetch")
        
    mcp = MCPRun(model="gpt-4o-mini", profile="pachacamac/default", task="FetchWeb", input_model=InputSchema).init()
    result = mcp.forward({"url": "https://example.com"})
    assert result is not None
    