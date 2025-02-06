import os
import pytest
from donew.new.assistants.mcp_task import NewMCPTask

presigned_url = "https://www.mcp.run/api/runs/florianlehmann-ops/default/URL%20Fetch?nonce=Bv4_XO3DaezmDnPr7tq6aA&sig=Sp2znOSoBNPgrlDNuXMFDTK3LyzxOjxFn10zlK6LBN8"

def test_mcp_task():
    """
    Test the MCP task by fetching a URL and checking the result.
    """
    
    task_obj = NewMCPTask(name="URL Fetch", description="Fetch a URL", inputs={"url": "https://unrealists.com"}, output_type="string", presigned_url=presigned_url)
    task_obj.login()
    result = task_obj.forward("https://unrealists.com")

    assert result is not None
    assert "Unrealists" in result

