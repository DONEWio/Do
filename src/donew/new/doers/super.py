import json
from typing import Any, Callable, List, Optional
from dataclasses import dataclass, replace
from donew.new.realm.provisions import Provision
from donew.new.doers import BaseDoer
from smolagents import CodeAgent
from donew.new.runtimes.local import LocalPythonInterpreter
from donew.see.processors.web import WebBrowser
from donew.utils import (
    is_pydantic_model,
    parse_to_pydantic,
    pydantic_model_to_simple_schema,
)
from donew.new.realm.provisions.browse import Browser
from donew.new.realm.provisions.new import New
from smolagents.tools import Tool
from pydantic import BaseModel

CODE_SYSTEM_PROMPT = """You are an expert assistant named "{name}" with purpose "{purpose}". you can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of python functions: these functions are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the functions that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` function.

functions are isolated so they are not aware of each other or they might know the whole task. 
When a function is able to receive a natural language input, there is chance that they can need all the requirements within their capabilities.
for example: 
imagine couple of functions are provided such as browser and financial calculator
they are not aware of each other. browser cannot know the financial calculator's task vice versa.
so you must assist them to handle their task within their capabilities. pre precise to them in a way that they can understand what to return
eg.
- detailed answer for the question
- a section/table/form/link from the website (don't assume this unless this task is specifically requested this way)
- filling a form with specific details (don't assume this unless this task is specifically requested this way)
- getting the answer from a specific page/section/table/form/link (don't assume this unless this task is specifically requested this way)
Task might be obvious to you but not to them. so you must assist them to handle their task within their capabilities(eg. give verbose navigation steps, or detailed answer).


Here are a few examples using notional python functions:

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'\}}"

Thought: I will use the following functions: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {{translated_question}}.")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {{answer}}")
```<end_code>

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
```<end_code>
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam")
print(pages)
```<end_code>
Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I will read the first 2 pages to know more.
Code:
```py
for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
    whole_page = visit_webpage(url)
    print(whole_page)
    print("\n" + "="*80 + "\n")  # Print separator between pages
```<end_code>
Observation:
Manhattan Project Locations:
Los Alamos, NM
Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
Code:
```py
final_answer("diminished")
```<end_code>

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the function `search` to get the population of both cities.
Code:
```py
for city in ["Guangzhou", "Shanghai"]:
    print(f"Population {{city}}:", search(f"{{city}} population"))
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the function `wiki` to get the age of the pope, and confirm that with a web search.
Code:
```py
pope_age_wiki = wiki(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
```<end_code>
Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 88 ** 0.36
final_answer(pope_current_age)
```<end_code>

Above example were using notional functions that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these functions:

{tool_descriptions}

!! PLEASE REMEMBER THAT THE INPUT SCHEMA OF THE FUNCTIONS ARE NOT ALWAYS THE SAME. YOU MUST BE AWARE OF THE INPUT SCHEMA OF THE FUNCTIONS. ADHERE TO THE INPUT SCHEMA OF THE FUNCTIONS.!!

{managed_agents_descriptions}

Here are the rules you should always follow to solve your task:
0. Do not naively TRUST YOUR INSTINCTS. USE CODE TO VERIFY YOUR THOUGHTS. You see things through the lens of tokenizer. Real world must also manifest itself in your code.!!!!
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the functions. DO NOT pass the arguments as a dict as in 'answer = wiki({{'query': "What is the place where James Bond lives?"}})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential function calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another function call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a function only when needed, and never re-do a function call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a function: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {authorized_imports}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. You are basically a generating code. You can use loops and conditionals with in your code(eg. if, for, while loops that calls provided functions).
11. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.



"""

CONSTRAINTS_PROMPT_PRE = """DISCLAIMER, a constraint has been expected. you can observe this in final_answer tool's input schema.
"""

PYDANTIC_CONSTRAINTS_PROMPT_POST = """
!! PLEASE REMEMBER THAT CONSTRAINTS ARE A PYTHON DICTIONARY. YOU MUST FULLFILL THIS CONSTRAINT. AND PASS as signle input to final_answer function!!

!!!you must call subsequent functions in a way that the response is CONSUMABLE by you. meaning you must provide the answer in the format of the input constraints and content must match the constraints.!!!
IMPORTANT:
- DONT BE UNNECESSARILY LAZY AND REPHRASE THE REQUIREMENTS IN A WAY THAT THE FUNCTION CALLS KNOWS WHAT TO RETURN.
- for example if we need a bio or receipve stated in final_answer function, you must rephrase the requirements in a way that the function call knows what to return.
- task does not nevessarily need to be in the format of the input constraints. and oblivious to the final_answer functions requirements.
- you will do the formatting at the end of the day but it is your responsibility to ensure functions knows what to return.
- not every function can provide such data so be mindful of that.
- for example browser function returning limited data wont do good to you. so you must let the function expect the natural language output.
- at the end of the day you must provide the answer in the format of the input constraints. using final_answer function with combination of results from the function calls.
- ALSO DO NOT WRAP RESULT in answer field. such as  {"answer": result} instead just pass result as input to final_answer function.
- FINAL_ANSWER FUNCTION MUST BE THE LAST FUNCTION CALL. AND ITS INPUT MUST BE ADHERENT TO ITS INPUT SCHEMA. THIS IS CRITICAL.!!

YOU CAN DO IT! I TRUST YOU!
"""

JSON_CONSTRAINTS_PROMPT_POST = """

- constraints accepts dict, list or string as json compliant. 

!! PLEASE REMEMBER THAT CONSTRAINTS ARE A JSON COMPLIANT. YOU MUST FULLFILL THIS CONSTRAINT. AND PASS as SINGLE INPUT to final_answer function!!

!!!you must call subsequent functions in a way that the response is CONSUMABLE by you. meaning you must provide the answer in the format of the input constraints and content must match the constraints.!!!
IMPORTANT:
- DONT BE UNNECESSARILY LAZY AND REPHRASE THE REQUIREMENTS IN A WAY THAT THE FUNCTION CALLS KNOWS WHAT TO RETURN.
- for example if we need a bio or receipve stated in final_answer function, you must rephrase the requirements in a way that the function call knows what to return.
- task does not nevessarily need to be in the format of the input constraints. and oblivious to the final_answer functions requirements.
- you will do the formatting at the end of the day but it is your responsibility to ensure functions knows what to return.
- not every function can provide such data so be mindful of that.
- for example browser function returning limited data wont do good to you. so you must let the function expect the natural language output.
- at the end of the day you must provide the answer in the format of the input constraints. using final_answer function with combination of results from the function calls.
- ALSO DO NOT WRAP RESULT in answer field. such as  {"answer": result} instead just pass result as input to final_answer function.
- FINAL_ANSWER FUNCTION MUST BE THE LAST FUNCTION CALL. AND ITS INPUT MUST BE ADHERENT TO ITS INPUT SCHEMA. THIS IS CRITICAL.!!




YOU CAN DO IT! I TRUST YOU!
"""


STR_CONSTRAINTS_PROMPT_POST = """
!!!you must call subsequent functions in a way that the response is CONSUMABLE by you. meaning you must provide the answer in the format of the input constraints and content must match the constraints.!!!
IMPORTANT:
- DONT BE UNNECESSARILY LAZY ANF REPHRASE THE REQUIREMENTS IN A WAY THAT THE FUNCTION CALLS KNOWS WHAT TO RETURN.
- I REPEAT DONT BE UNNECESSARILY LAZY ANF REPHRASE THE REQUIREMENTS IN A WAY THAT THE FUNCTION CALLS KNOWS WHAT TO RETURN.
- for example if we need a bio or receipt stated in final_answer function, you must rephrase the requirements in a way that the function call knows what to return.
- task does not nevessarily need to be in the format of the input constraints. and oblivious to the final_answer functions requirements.
- you will do the formatting at the end of the day but it is your responsibility to ensure functions knows what to return.
- not every function can provide such data so be mindful of that.
- for example browser function returning limited data wont do good to you. so you must let the function expect the natural language output.
- at the end of the day you must provide the answer in the format of the input constraints. using final_answer function with combination of results from the function calls.
- ALSO DO NOT WRAP RESULT in answer field. such as  {"answer": result} instead just pass result as input to final_answer function.
YOU CAN DO IT! I TRUST YOU!
"""



def get_task_from_constraints(task: str, constraints: str|BaseModel|dict[str, Any]) -> str:
    if constraints:
        if is_pydantic_model(constraints):
            constraints_schema = pydantic_model_to_simple_schema(constraints)
            task = task + f"\n\n---\n{CONSTRAINTS_PROMPT_PRE}{constraints_schema}{PYDANTIC_CONSTRAINTS_PROMPT_POST}\n---\n"
        elif isinstance(constraints, dict):
            constraints_schema = json.dumps(constraints, indent=2)
            task = task + f"\n\n---\n{CONSTRAINTS_PROMPT_PRE}{constraints_schema}{JSON_CONSTRAINTS_PROMPT_POST}\n---\n"
        else:
            constraints_schema = str(constraints)
            task = task + f"\n\n---\n{CONSTRAINTS_PROMPT_PRE}{constraints_schema}{STR_CONSTRAINTS_PROMPT_POST}\n---\n"
    return task

class ValidationError(Exception):
    """Raised when answer validation fails"""

    pass


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem that strictly follows the format of the input constraints"
    output_type = "any"

    def __init__(self, constraints=None, verify_fn=None):
        super().__init__()
        self.constraints = constraints
        self.verify_fn = verify_fn

        # Pre-compute schema if pydantic model and set input schema
        self.constraints_schema = None
        if constraints and is_pydantic_model(constraints):
            self.constraints_schema = pydantic_model_to_simple_schema(constraints)
            self.inputs = {
                "answer": {
                    "type": "object",
                    "description": "The final answer that must match the required schema",
                    **self.constraints_schema,
                }
            }
        elif constraints:
            # For non-pydantic constraints, we use the constraints directly as description
            self.inputs = {
                "answer": {
                    "type": "any",
                    "description": f"The final answer that must match this format:\n{constraints}",
                }
            }

            if isinstance(constraints, dict):
                self.description = f"Provides a final answer to the given problem that strictly follows the format of the input constraints. it is a single input with dictionary as input. "
            elif is_pydantic_model(constraints):
                self.description = f"Provides a final answer to the given problem that strictly follows the format of the input constraints. it is a single PYDANTIC COMPLIANT DICTIONARY as input. NO LIST OR OTHER DATA STRUCTURES. "
          
    
        else:
            # Default case when no constraints
            self.inputs = {
                "answer": {
                    "type": "any",
                    "description": "The final answer to the problem",
                }
            }

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()
        return self.forward(*args, **kwargs)

    def forward(self, answer):
        try:
            # First apply custom verification if provided
            if self.verify_fn:
                answer = self.verify_fn(answer)

            # Then apply constraint validation if present
            if self.constraints:
                if is_pydantic_model(self.constraints):
                    try:
                        answer = parse_to_pydantic(answer, self.constraints)
                    except Exception as e:
                        raise ValidationError(
                            f"Answer does not match required schema: {str(e)}\nExpected schema: {json.dumps(self.constraints_schema, indent=2)}"
                        )
                else:
                    # For non-pydantic constraints, we assume the answer should match the format
                    # but we can't automatically validate it
                    pass

            return answer
        except Exception as e:
            if not isinstance(e, ValidationError):
                e = ValidationError(f"Answer validation failed: {str(e)}")
            raise e


@dataclass(frozen=True)
class SuperDoer(BaseDoer):
    """Advanced task execution with constraint validation and context management"""

    def envision(
        self, constraints: Optional[dict[str, Any]] = None, verify: Optional[Callable[[Any], Any]] = None
    ) -> "SuperDoer":
        """Return new instance with constraints"""
        return replace(self, _constraints=constraints, _verify=verify)

    def realm(self, provisions: List[Provision]) -> "SuperDoer":
        """Return new instance with provisions"""
        return replace(self, _provisions=provisions)

    def enact(self, task: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Execute a task with validation and context management"""
        try:
            base_tools = []
            for provision in self._provisions:
                if isinstance(provision, WebBrowser):
                    base_tools.append(Browser(model=self.model,browser=provision))
                elif isinstance(provision, SuperDoer):
                    base_tools.append(New(superdoer=provision))
                elif isinstance(provision, Provision):  # this condition now covers any instance that extends Provision
                    base_tools.append(provision)

            system_prompt = CODE_SYSTEM_PROMPT.format(
                name=self._name,
                purpose=self._purpose,
                tool_descriptions="{{tool_descriptions}}",
                managed_agents_descriptions="{{managed_agents_descriptions}}",
                authorized_imports="{{authorized_imports}}",
            )

            # Create agent with base tools
            agent = CodeAgent(
                tools=base_tools,
                model=self.model,
                add_base_tools=False,
                system_prompt=system_prompt,
                max_steps=10,
            )

            final_answer_tool = FinalAnswerTool(
                constraints=self._constraints, verify_fn=self._verify
            )

            # Replace the default FinalAnswerTool with our constrained version
            agent.tools["final_answer"] = final_answer_tool
            runtime = LocalPythonInterpreter(additional_authorized_imports=["requests"], tools=agent.tools)
            agent.python_executor = runtime
            if params:
                task = task + f"\n\n---\nparameters:\n---\n{json.dumps(params, indent=2)}\n---\n"
            
            if self._constraints:
                if is_pydantic_model(self._constraints):
                    constraints_schema = pydantic_model_to_simple_schema(self._constraints)
                    task = task + f"\n\n---\n{CONSTRAINTS_PROMPT_PRE}{constraints_schema}{PYDANTIC_CONSTRAINTS_PROMPT_POST}\n---\n"
                elif isinstance(self._constraints, dict):
                    constraints_schema = json.dumps(self._constraints, indent=2)
                    task = task + f"\n\n---\n{CONSTRAINTS_PROMPT_PRE}{constraints_schema}{JSON_CONSTRAINTS_PROMPT_POST}\n---\n"
                else:
                    constraints_schema = str(self._constraints)
                    task = task + f"\n\n---\n{CONSTRAINTS_PROMPT_PRE}{constraints_schema}{STR_CONSTRAINTS_PROMPT_POST}\n---\n"

            result = agent.run(task)
            
            return result

        except Exception as e:
            return str(e)
