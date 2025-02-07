import json
import time
import os
import requests
from requests.cookies import RequestsCookieJar
from opentelemetry import trace
from donew.new.assistants import Provision

class MCPRun(Provision):
    name = ""
    description = """"""
    inputs = {
        "task": {
            "type": "string",
            "description": ""
        }
    }
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = kwargs.get("model", None)
        self.mcprun_profile = kwargs.get("profile", "default")
        self.mcprun_task = kwargs.get("task", None)
        self.inputs = kwargs.get("inputs", {})
        self.name = self.mcprun_task
        self.mcprun_presigned_url = None
        self.mcprun_login_timeout = kwargs.get("mcprun_login_timeout", 60)
        self.mcprun_run_timeout = kwargs.get("mcprun_run_timeout", 60)
        self.cookies = RequestsCookieJar()
        self.load_cookie_jar()
    
    def init(self):
        self.login()
        task = self.find_task()
        if task is None:
            raise ValueError("Task not found")
        self.description = task.get("openai", task.get("anthropic", {}).get("prompt", ""))
        self.inputs = {} # TODO: KEINE AHNUNG ALTER
        # self.create_signed_url() # gets done in run_task automatically
        return self
    
    
    def persist_cookie_jar(self):
        # TODO: use lib to save cookies in the right folder
        with open("cookiejar.json", "w") as f:
            json.dump(requests.utils.dict_from_cookiejar(self.cookies), f)
    
    def load_cookie_jar(self):
        # TODO: use lib to load cookies from the right folder
        if os.path.exists("cookiejar.json"):
            with open("cookiejar.json", "r") as f:
                self.cookies = requests.utils.cookiejar_from_dict(json.load(f))

    def api_request(self, path, method, data=None, headers={}):
        url = f"https://www.mcp.run/api{path}"
        default_headers = {"Accept": "*/*"}
        response = requests.request(
            method,
            url,
            headers={**default_headers, **headers},
            cookies=self.cookies,
            data=data
        )
        response.raise_for_status()
        if response.cookies:
            self.cookies.update(response.cookies)
            self.persist_cookie_jar()
        return response.json()
    

    def whoami(self):
        return self.api_request("/auth/whoami", "GET")


    def login(self):
        try:
            # if whoami returns an error or empty, we need to login
            if self.whoami() is not None:
                print('SKIPPING LOGIN')
                return # already logged in
        except Exception as e:
            pass
        print('PERFORMING LOGIN')
        resp = self.api_request("/login/start", "POST", headers={"Content-Type": "text/plain;charset=UTF-8"})
        code, approve_url = resp.get('code'), resp.get('approveUrl')
        if not code or not approve_url:
            raise ValueError("Invalid login response")
        print(f"Received login initiation response. Code: {code}, Approve URL: {approve_url}")
        try:
            import webbrowser
            print(f"Opening browser for approval URL: {approve_url}")
            webbrowser.open(approve_url)
        except Exception as e:
            print(f"Error opening browser: {e}")
            print(f"Please open the following URL in your browser and approve the login within {self.mcprun_login_timeout} seconds: {approve_url}")
        print("Waiting for login approval...")
        for _ in range(self.mcprun_login_timeout//2):
            resp = self.api_request(f"/login/poll?code={code}", "GET")
            if resp.get('status') == 'ok':
                print("Login approved")
                break
            time.sleep(2)
        else:
            raise ValueError("Login approval timed out")
        

    def extract_final_message(self, run_data):
        runs = run_data if isinstance(run_data, list) else [run_data]
        is_valid_message = lambda msg: (msg.get('content') and "Please provide the URL" not in msg.get('content'))
        for run in runs:
            for result in run.get('results', []):
                if (result.get('msg') == 'final message' and is_valid_message(result.get('lastMessage', {}))):
                    return result['lastMessage']['content']
        return "<no output>"


    def run_task(self, payload: dict):
        if not self.mcprun_presigned_url:
            self.create_signed_url()
        resp = requests.post(self.mcprun_presigned_url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        if "url" not in data:
            raise ValueError("No URL found in initial response")
        result_url = data["url"]
        for _ in range(self.mcprun_run_timeout//2):
            resp = requests.get(result_url, cookies=self.cookies)
            resp.raise_for_status()
            data = resp.json()
            print(data)
            if data.get("status") == "ready":
                return self.extract_final_message(data)
            time.sleep(2)
        raise ValueError("Task result timed out")


    def get_tasks(self):
        return self.api_request("/users/~/tasks", "GET")
    
    def find_task(self, task_name = None):
        task_name = task_name or self.mcprun_task
        tasks = self.get_tasks()
        for task in tasks:
            if task.get("name") == task_name:
                return task
        return None

    # def get_servelets(self):
    #     return self.api_request(f"/profiles/{self.mcprun_profile}/installations", "GET")

    # def get_task(self, task_name = None):
    #     task_name = task_name or self.mcprun_task
    #     servelets = self.get_servelets()
    #     for task in servelets.get('installs',[]):
    #         if task.get("name") == task_name:
    #             return task
    #     return None

    def create_signed_url(self):
        resp = self.api_request(f"/tasks/{self.mcprun_profile}/{self.mcprun_task}/signed", "POST")
        self.mcprun_presigned_url = resp.get("url")


    def _execute_task(self, params: dict):
        result = "<no result>"
        try:
            result = self.run_task(params)
        except Exception as e:
            result = str(e)
        return f"""
        {self.__class__.__name__} - {self.name} has completed the task.
        ---
        {result}
        ---
        """


    def forward(self, task: str):
        """Execute a task with validation and context management"""
        try:
            # Try to get tracer, but don't fail if tracing is not enabled
            try:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(self.name) as span:
                    span.set_attribute("task", task)
                    result = self._execute_task(json.loads(task))
                    span.set_attribute("result", str(result))
                    return result
            except Exception:  # Tracing not available or failed
                return self._execute_task(task)
        except Exception as e:
            return str(e)