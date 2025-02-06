import json
import time
import requests

class NewMCPTask():
    def __init__(self, *args, **kwargs):
        self._headers = kwargs.get("headers", {"Content-Type": "application/json"})
        self._cookies = kwargs.get("cookies", {})
        self._presigned_url = kwargs.get("presigned_url")

        self._payload = kwargs.get("inputs")
        
        self._response_url = None
        self._response_status_code = None
        self._response_final_message = None
        self._code = None

    def _get_payload(self):
        return self._payload
    
    def login(self):
        """
        Initiates the login process by contacting the login API, opening the approval URL,
        and then polling until the login is approved.
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

        # Poll the poll URL until approved.
        import time
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

            # Check for an approval flag.
            if poll_response_json.get("status") == "ok":
                print(f"Login approved with poll response: {poll_response_json}")
                # Store the headers including cookies from the successful poll response
                self._headers = poll_response.headers
                break

            if time.time() - start_time > timeout:
                print(f"Timeout waiting for login approval after {timeout} seconds")
                raise TimeoutError("Timed out waiting for login approval.")

            # If a 202 response is received, use the provided "backoff" or default to 2 seconds.
            sleep_time = 2
            if poll_response.status_code == 202:
                backoff = poll_response_json.get("backoff")
                if isinstance(backoff, (int, float)):
                    sleep_time = backoff
            print(f"Login not approved yet. Sleeping for {sleep_time} seconds before next poll.")
            time.sleep(sleep_time)

        print("Login process completed successfully.")

        # Extract session cookie from response headers
        cookies = poll_response.headers.get('Set-Cookie')
        if cookies:
            # Parse the Set-Cookie header to find sessionId
            for cookie in cookies.split(';'):
                if cookie.strip().startswith('sessionId='):
                    session_id = cookie.split('=')[1].strip()
                    # Update headers with session cookie
                    self._cookies = {'sessionId': session_id}
                    break
        return poll_response_json

    def forward(self, task):
        """
        Executes a POST request to a presigned URL and then fetches the final message from
        the URL provided in the response.

        If task is a string, it is assumed to be a URL and is wrapped into 
        {"url": <string>}. Alternatively, you can supply a dictionary with any keys 
        you need.
        """
        # Determine the payload based on the type of task provided.
        # if isinstance(self._payload, str):
        #     self._payload = json.dumps(self._payload)
        # elif isinstance(self._payload, dict):
        #     self._payload = json.dumps(self._payload)
        # else:
        #     raise ValueError("Task must be either a URL string or a dictionary of parameters.")

        # POST the JSON payload to the presigned URL.
        post_response = requests.post(self._presigned_url, json=self._payload, headers=self._headers)
        post_response.raise_for_status()
        initial_response = post_response.json()
        
        # If the initial response contains a URL for the runs, fetch the detailed results.
        if "url" in initial_response:
            result_url = initial_response["url"]
            
            # Poll until status is ready or max retries exceeded
            max_retries = 30  # 1 minute total with 2 second sleep
            retry_count = 0
            
            while retry_count < max_retries:
                get_response = requests.get(result_url, cookies=self._cookies)
                run_data = get_response.json()
                print(f"Poll attempt {retry_count + 1}/{max_retries}")
                print(f"Response status: {get_response.status_code}")
                print(f"Run data: {json.dumps(run_data, indent=2)}")
                
                if isinstance(run_data, dict):
                    status = run_data.get("status")
                    print(f"Current status: {status}")
                    if status == "ready":
                        print("Status is ready, proceeding to extract message")
                        break
                
                retry_count += 1
                print(f"Waiting 2 seconds before next poll...")
                time.sleep(2)
            
            if retry_count >= max_retries:
                raise TimeoutError("Maximum retries exceeded while waiting for ready status")
            
            # Extract the final message content from the run data
            final_message = "No output found"
            if isinstance(run_data, list):
                for run in run_data:
                    results = run.get("results", [])
                    for result in results:
                        if result.get("msg") == "final message":
                            last_message = result.get("lastMessage", {})
                            candidate = last_message.get("content") if last_message else None
                            # Skip if the message is a default/fallback prompt
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
                        # Skip if the message is a default/fallback prompt
                        if candidate and "Please provide the URL" not in candidate:
                            final_message = candidate
                            break
            
            self.final_message = final_message
            self._response = run_data
            self._status_code = get_response.status_code
            return self.final_message
        else:
            # Fallback: if no URL is provided, extract the message directly.
            self.final_message = initial_response.get("message", "No message returned.")
            self._response = initial_response
            self._status_code = post_response.status_code
            return self.final_message
