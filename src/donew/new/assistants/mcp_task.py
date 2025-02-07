import json
import time
import os                   # added for checking local cookie jar file
import requests
from requests.cookies import RequestsCookieJar  # added for cookie jar support

class NewMCPTask():
    def __init__(self, *args, **kwargs):
        self._headers = kwargs.get("headers", {"Content-Type": "application/json"})
        # Get cookies as a dict (if provided)
        self._cookies = kwargs.get("cookies")
        # Check if a cookie jar is provided or if a local cookie jar file exists.
        if "cookiejar" in kwargs:
            self._cookiejar = kwargs.get("cookiejar")
        elif os.path.exists("cookiejar.json"):
            with open("cookiejar.json", "r") as f:
                cookies_dict = json.load(f)
            self._cookiejar = requests.utils.cookiejar_from_dict(cookies_dict)
        else:
            self._cookiejar = RequestsCookieJar()
        
        # If _cookies is provided as a dict, add them to the cookie jar.
        if self._cookies:
            for key, value in self._cookies.items():
                self._cookiejar.set(key, value)
        
        self._presigned_url = kwargs.get("presigned_url")
        self._payload = kwargs.get("inputs")
        
        self._response_url = None
        self._response_status_code = None
        self._response_final_message = None
        self._code = None
        self._store_cookie = kwargs.get("store_cookie", False)

    def _get_payload(self):
        return self._payload
    
    def login(self):
        """
        Initiates the login process by contacting the login API, opening the approval URL,
        and then polling until the login is approved. Uses a cookie jar to store cookies.
        """
        url = "https://www.mcp.run/api/login/start"
        print(f"Initiating login with URL: {url}")

        headers = {
            "Accept": "*/*",
            "Content-Type": "text/plain;charset=UTF-8"
        }

        try:
            response = requests.post(url, headers=headers)
            print(f"Login initiation response status: {response.status_code}")
            response_json = response.json()
        except Exception as e:
            print(f"Login start request failed: {e}")
            raise

        code = response_json.get('code')
        approve_url = response_json.get('approveUrl')

        if not code or not approve_url:
            print(f"Invalid login response, missing code or approveUrl: {response_json}")
            raise ValueError("Invalid login response")

        print(f"Received login initiation response. Code: {code}, Approve URL: {approve_url}")

        # Open the approval URL in the browser.
        import webbrowser
        print(f"Opening browser for approval URL: {approve_url}", flush=True)
        webbrowser.open(approve_url)

        print("Waiting for login approval...", flush=True)

        # Build the poll URL using the received code.
        poll_url = f"https://www.mcp.run/api/login/poll?code={code}"
        print(f"Polling login confirmation URL: {poll_url}", flush=True)

        start_time = time.time()
        timeout = 10  # seconds; adjust this timeout according to your use-case

        while True:
            print(f"Sending poll request to: {poll_url}")
            poll_response = requests.get(
                poll_url, headers={
                    "Accept": "*/*",
                    "Content-Type": "application/json"
                }
            )
            print(f"Poll response status: {poll_response.status_code}")
            try:
                poll_response_json = poll_response.json()
            except ValueError:
                print("Poll response did not contain valid JSON.")
                poll_response_json = {}

            print(f"Poll response JSON: {poll_response_json}")

            if poll_response_json.get("status") == "ok":
                print(f"Login approved with poll response: {poll_response_json}")
                # Update the cookie jar with cookies from the poll response.
                self._cookiejar.update(poll_response.cookies)
                break

            if time.time() - start_time > timeout:
                print(f"Timeout waiting for login approval after {timeout} seconds")
                raise TimeoutError("Timed out waiting for login approval.")

            sleep_time = 2
            if poll_response.status_code == 202:
                backoff = poll_response_json.get("backoff")
                if isinstance(backoff, (int, float)):
                    sleep_time = backoff
            print(f"Login not approved yet. Sleeping for {sleep_time} seconds before next poll.")
            time.sleep(sleep_time)

        print("Login process completed successfully.")
        if self._store_cookie:
            with open("cookiejar.json", "w") as f:
                json.dump(requests.utils.dict_from_cookiejar(self._cookiejar), f)
            print("Cookie jar stored to cookiejar.json")
        return poll_response_json

    def run(self, task=None):
        """
        Sends a POST request to the presigned URL with the payload (or overriding task)
        and returns the initial JSON response.
        """
        if task is not None:
            self._payload = task
        print(f"Sending task to presigned URL: {self._presigned_url}")
        post_response = requests.post(self._presigned_url, json=self._payload, headers=self._headers)
        post_response.raise_for_status()
        initial_response = post_response.json()
        print(f"Initial response: {initial_response}")
        if "url" in initial_response:
            self._result_url = initial_response["url"]
        else:
            raise ValueError("No URL found in initial response")
        return initial_response

    def result(self, result_url=None):
        """
         Polls the given result URL until the task status becomes 'ready' or a timeout occurs.
         Then extracts and returns the final message, the run data, and the response status code.
         Before polling, checks if the user is logged in by verifying the cookie jar contains a session cookie.
         """
        if result_url is None:
            result_url = self._result_url

        # Check if logged in by verifying if a session cookie (assumed to be "sessionId") exists.
        logged_in = any(cookie.name == "sessionId" for cookie in self._cookiejar)
        if not logged_in:
            return "Not logged in. Please login first.", None, None

        max_retries = 30  # 1 minute total with 2 second sleep intervals
        retry_count = 0
        run_data = None

        while retry_count < max_retries:
            print(f"Polling result URL ({retry_count + 1}/{max_retries}): {result_url}")
            get_response = requests.get(result_url, cookies=self._cookiejar)
            run_data = get_response.json()
            print(f"Response status: {get_response.status_code}")
            print(f"Run data: {json.dumps(run_data, indent=2)}")
            
            if isinstance(run_data, dict):
                status = run_data.get("status")
                print(f"Current status: {status}")
                if status == "ready":
                    print("Status is ready, proceeding to extract message.")
                    break

            retry_count += 1
            print("Waiting 2 seconds before next poll...")
            time.sleep(2)

        if retry_count >= max_retries:
            raise TimeoutError("Maximum retries exceeded while waiting for ready status")

        final_message = "No output found"
        if isinstance(run_data, list):
            for run in run_data:
                results = run.get("results", [])
                for result in results:
                    if result.get("msg") == "final message":
                        last_message = result.get("lastMessage", {})
                        candidate = last_message.get("content") if last_message else None
                        if candidate and "Please provide the URL" not in candidate:
                            final_message = candidate
                            break
                if final_message != "No output found":
                    break
        elif isinstance(run_data, dict):
            results = run_data.get("results", [])
            for result in results:
                if result.get("msg") == "final message":
                    last_message = result.get("lastMessage", {})
                    candidate = last_message.get("content") if last_message else None
                    if candidate and "Please provide the URL" not in candidate:
                        final_message = candidate
                        break

        return final_message

    # def forward(self, task=None):
    #     """
    #     Orchestrates running the task and retrieving its results. It first sends the task using run_task,
    #     then polls for the results using retrieve_results. If no result URL is provided in the response,
    #     a fallback message is returned.
    #     """
    #     initial_response = self.run_task(task)
    #     if "url" in initial_response:
    #         result_url = initial_response["url"]
    #         final_message, run_data, status_code = self.retrieve_results(result_url)
    #         self.final_message = final_message
    #         self._response = run_data
    #         self._status_code = status_code
    #         return final_message
    #     else:
    #         self.final_message = initial_response.get("message", "No message returned.")
    #         self._response = initial_response
    #         self._status_code = post_response.status_code if 'post_response' in locals() else None
    #         return self.final_message
