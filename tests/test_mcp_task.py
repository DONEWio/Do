import os
import pytest
from donew.new.assistants.mcp_task import NewMCPRunTask
import json
from requests.cookies import RequestsCookieJar
import requests

presigned_url = "https://www.mcp.run/api/runs/florianlehmann-ops/default/URL%20Fetch?nonce=Bv4_XO3DaezmDnPr7tq6aA&sig=Sp2znOSoBNPgrlDNuXMFDTK3LyzxOjxFn10zlK6LBN8"

def test_mcp_task():
    """
    Test the MCP task by fetching a URL and checking the result.
    """
    
    task_obj = NewMCPRunTask(name="URL Fetch", description="Fetch a URL", inputs={"url": "https://unrealists.com"}, output_type="string", presigned_url=presigned_url)
    task_obj.login()
    task_obj.run({"url": "https://unrealists.com"})
    result = task_obj.result()

    assert result is not None
    assert "Unrealists" in result

class FakeResponse:
    def __init__(self, json_data, status_code, cookies=None):
        self._json_data = json_data
        self.status_code = status_code
        self.cookies = cookies if cookies is not None else RequestsCookieJar()

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception("HTTP error")

def fake_post(url, **kwargs):
    # For login start endpoint
    if "login/start" in url:
        return FakeResponse({"code": "fakecode", "approveUrl": "http://fake_approve_url"}, 200)
    # For run_task endpoint (presigned URL)
    if "runs" in url:
        return FakeResponse({"url": "http://fake_result_url"}, 200)
    return FakeResponse({}, 404)

def fake_get(url, **kwargs):
    # For login poll endpoint
    if "login/poll" in url:
        fake_cookiejar = RequestsCookieJar()
        fake_cookiejar.set("sessionId", "fake_session", domain="www.mcp.run", path="/")
        return FakeResponse({"status": "ok"}, 200, cookies=fake_cookiejar)
    # For retrieving results
    if url == "http://fake_result_url":
        # Simulate a task response with final message
        return FakeResponse({
            "status": "ready",
            "results": [
                {
                    "msg": "final message",
                    "lastMessage": {"content": "Fake final message"}
                }
            ]
        }, 200)
    return FakeResponse({}, 404)

def test_login_without_cookiejar(monkeypatch):
    # Monkeypatch requests methods used in login
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    # Create instance without providing a cookie jar (store_cookie defaults to False)
    task_obj = NewMCPRunTask(presigned_url=presigned_url)
    poll_response = task_obj.login()

    assert poll_response.get("status") == "ok"
    # Verify that cookie jar now contains a "sessionId" cookie
    session_cookie_exists = any(cookie.name == "sessionId" for cookie in task_obj._cookiejar)
    assert session_cookie_exists

def test_login_with_cookiejar_store(monkeypatch, tmp_path):
    # Monkeypatch requests methods used in login
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    # Change the working directory to a temporary directory for isolation.
    monkeypatch.chdir(tmp_path)

    # Create instance with store_cookie enabled.
    task_obj = NewMCPRunTask(presigned_url=presigned_url, store_cookie=True)
    poll_response = task_obj.login()

    assert poll_response.get("status") == "ok"

    # Verify that the cookiejar file was created.
    cookie_path = tmp_path / "cookiejar.json"
    assert cookie_path.exists()

    # Load the stored cookies and verify they include the sessionId.
    with open(cookie_path, "r") as f:
        cookies_dict = json.load(f)
    assert "sessionId" in cookies_dict

def test_retrieve_results_without_login():
    # Create instance without calling login so that cookie jar remains empty.
    task_obj = NewMCPRunTask(presigned_url=presigned_url)
    result_message, run_data, status_code = task_obj.result("http://fake_result_url")

    assert result_message == "Not logged in. Please login first."
    assert run_data is None
    assert status_code is None

def test_run_task_and_retrieve_results_with_login(monkeypatch):
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    # Create instance and simulate prior login by manually setting the session cookie.
    task_obj = NewMCPRunTask(presigned_url=presigned_url)
    task_obj._cookiejar.set("sessionId", "fake_session", domain="www.mcp.run", path="/")

    # Test run_task: it should return a response containing a result URL.
    initial_response = task_obj.run({"url": "https://example.com"})
    assert "url" in initial_response

    # Test retrieve_results using the fake result URL.
    final_message, run_data, status_code = task_obj.result("http://fake_result_url")
    assert final_message == "Fake final message"
    assert run_data is not None
    assert status_code == 200

# Integration tests (should only run when explicitly marked)

@pytest.mark.integration
def test_real_login_without_existing_cookiejar():
    """
    Real integration test for login without an existing cookiejar.
    Skips unless RUN_INTEGRATION env var is set to 'true'.
    """
    if os.environ.get("RUN_INTEGRATION", "false").lower() != "true":
        pytest.skip("Skipping real integration tests. Set RUN_INTEGRATION=true to run.")

    task_obj = NewMCPRunTask(presigned_url=presigned_url)
    poll_response = task_obj.login()
    assert poll_response.get("status") == "ok"
    session_cookie_exists = any(cookie.name == "sessionId" for cookie in task_obj._cookiejar)
    assert session_cookie_exists

@pytest.mark.integration
def test_real_login_with_existing_cookiejar(monkeypatch, tmp_path):
    """
    Real integration test for login with an existing cookiejar.
    Pre-create a cookiejar.json file with a dummy cookie and verify it gets updated.
    Skips unless RUN_INTEGRATION env var is set to 'true'.
    """
    if os.environ.get("RUN_INTEGRATION", "false").lower() != "true":
        pytest.skip("Skipping real integration tests. Set RUN_INTEGRATION=true to run.")

    import json
    # Pre-create a cookiejar.json file in the temporary directory with a dummy "sessionId"
    cookie_data = {"sessionId": "old_session"}
    cookie_file = tmp_path / "cookiejar.json"
    with open(cookie_file, "w") as f:
        json.dump(cookie_data, f)

    # Change working directory so that NewMCPTask picks up the cookie file.
    monkeypatch.chdir(tmp_path)

    # Instantiate the task; the constructor should load the existing cookiejar.json.
    task_obj = NewMCPRunTask(presigned_url=presigned_url)

    # Before login, verify that the cookie jar has the dummy "old_session"
    init_cookies = requests.utils.dict_from_cookiejar(task_obj._cookiejar)
    assert init_cookies.get("sessionId") == "old_session"

    # Call login. A real login will open the browser for approval and update the cookie.
    poll_response = task_obj.login()
    assert poll_response.get("status") == "ok"

    # After login, verify that the cookie jar now has an updated session cookie.
    updated_cookies = requests.utils.dict_from_cookiejar(task_obj._cookiejar)
    assert updated_cookies.get("sessionId") != "old_session"

