from typing import Optional
from smolagents import CodeAgent
from smolagents.tools import Tool
from donew.new.assistants import Provision
from donew.new.runtimes.local import LocalPythonInterpreter
from donew.new.types import Model
from donew.see.processors.web import WebBrowser, WebProcessor
from opentelemetry import trace

STATE = {}

CODE_SYSTEM_PROMPT = """You are an expert assistant who can solve any web browsing task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
Additionally, you have been given a web browser, which you can use to browse the web.



To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.



--- BROWSER DOCUMENTATION ---
a local variable "browser" which is an instance of "WebBrowser" is already defined. you can carefully read the documentation of the browser to use it.
It looks similar to playwright browser using chrome.
It uses local web browser and user profile is loaded safely.
It is safe to call this for a potentially auth required website.
browser tool is isolated and do not share cookies with other tools

{documentation}
---
Here are a few examples using WebBrowser api:
---
Task: "Give details about the website https://example.com and the people behind it."

Thought: I will use the browser to visit the website and get the details about people behind it.
Code:
```py

browser.goto("https://example.com")
text = browser.text()
print(text)
```<end_code>
Observation: "I am going to visit the https://example.com/about" which is tagged with element id 9

Thought: I will now visit the website and get the details about people behind it.
Code:
```py
browser.click(9)
text = browser.text()
print(text)
```<end_code>
Observation: "I have visited the website and got the details about people behind it. John Doe and Jane Doe are the founders of the company."
Thought: I will now generate a python code that submits the details to the `final_answer` toolqq.
Code:
```py
final_answer(f"The founders of the company are John Doe and Jane Doe.")
```<end_code>

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:

{tool_descriptions}

{managed_agents_descriptions}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {authorized_imports}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

!!! IMPORTANT:
at the begiging browser bapge is idle at about:blank. nabigate to the target website first.
```py
browser.goto("https://unrealists.com")
...
browser.text()
```
when returning a response get it as a text by calling `browser.text()`
dont return browser object, just the text.

text return links with their element_id. so dont shy away from browsing it like a human, meaning try to click on the links and see what happens if it is necessary.

DONT TRUNCATE OR CUT OFF ANYTHING. return what you see. that is RELEVANT to the task request.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""
class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {"type": "any", "description": "The final answer to the problem"}
    }
    output_type = "any"

    def forward(self, answer):
        return answer


class BrowseAssistant(Provision):
    name = "browse"
    description = """
    This browser tool is used to browse the web. It uses local web browser and user profile is loaded safely.
    It is safe to call this for a potentially auth required website.
    brwoser tool is siolated and do not share cookies with other tools
    """
    inputs = {
        "task": {
            "type": "string",
            "description": "Natural language description of the task only within the confines of web browsing",
        }
    }
    output_type = "string"
    browser:WebBrowser
    model:Optional[Model] = None
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = kwargs.get("model", None)
        self.browser = kwargs.get("browser", None)
    def forward(self, task: str):
        """Execute a task with validation and context management"""
        try:
            # Try to get tracer, but don't fail if tracing is not enabled
            try:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span("browse_task") as span:
                    span.set_attribute("task", task)
                    result = self._execute_task(task)
                    span.set_attribute("result", str(result))
                    return result
            except Exception:  # Tracing not available or failed
                return self._execute_task(task)
                
        except Exception as e:
            return str(e)

    def _execute_task(self, task: str):
        """Internal method to execute the task"""
        # Create and configure the agent

        documentation = "\n".join(self.browser.documentation())

        system_prompt = CODE_SYSTEM_PROMPT.format(
            documentation=documentation,
            tool_descriptions="{{tool_descriptions}}",
            managed_agents_descriptions="{{managed_agents_descriptions}}",
            authorized_imports="{{authorized_imports}}"
        )

        
        agent = CodeAgent(
            tools=[],
            model=self.model,
            add_base_tools=False,
            system_prompt=system_prompt,
        )
        tools = {tool.name: tool for tool in [FinalAnswerTool()]}
        runtime = LocalPythonInterpreter(additional_authorized_imports=["requests"], tools=tools)
        runtime.state["browser"] = self.browser
        agent.python_executor = runtime
        return agent.run(task)

#