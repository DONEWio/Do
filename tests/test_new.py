from dataclasses import Field
import os
from dotenv import load_dotenv
import pytest
from typing import List, Optional
from donew import DO
from donew.new.realm.provisions.mcprun import MCPRun
from donew.new.doers import BaseDoer
from donew.new.doers.super import SuperDoer
from donew.new import LiteLLMModel, ChatMessage, MessageRole
from donew.utils import enable_tracing, disable_tracing
from pydantic import BaseModel, Field, create_model

@pytest.fixture(autouse=True, scope="module")
def setup_test():
    """Automatically enable tracing for all tests in this module."""
    enable_tracing()
    load_dotenv()
    yield  # This will run the tests
    disable_tracing()  # This will run after all tests are done


class MockModel:
    def __init__(self):
        self.last_input_token_count = None
        self.last_output_token_count = None

    def get_token_counts(self) -> dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: list[dict[str, str]],
        stop_sequences: Optional[list[str]] = None,
        grammar: Optional[str] = None,
    ) -> ChatMessage:
        # Extract the last message content and return with Processed: prefix
        prompt = messages[-1]["content"]
        return ChatMessage(
            role=MessageRole.ASSISTANT, content=f"Processed: {prompt}", tool_calls=None
        )

    # For backward compatibility with older tests
    async def generate(self, prompt: str, **kwargs) -> str:
        return f"Processed: {prompt}"


class MockProvision:
    def __init__(self, name: str):
        self.name = name
        self.setup_called = False
        self.cleanup_called = False

    async def setup(self) -> None:
        self.setup_called = True

    async def cleanup(self) -> None:
        self.cleanup_called = True


def test_new_returns_superdoer():

    doer = DO.New(MockModel(), name="mock_doer", purpose="mock doer")
    assert isinstance(doer, SuperDoer)
    assert isinstance(doer, BaseDoer)


@pytest.mark.asyncio
async def test_method_chaining():
    load_dotenv()
    doer = await DO.A_new(MockModel(), name="mock_doer", purpose="mock doer")
    ctx = MockProvision("test")
    prompt = "test task"

    def verify_output(result: str) -> bool:
        expected = f"Processed: Based on the above, please provide an answer to the following user request:\n{prompt}"
        return expected == result

    # Method chaining

    doer = await DO.A_new(MockModel(), name="mock_doer", purpose="mock doer")
    result = doer.realm([ctx]).envision(verify=verify_output).enact(prompt)
    assert (
        f"Processed: Based on the above, please provide an answer to the following user request:\n{prompt}"
        == result
    )

    # Alternative style
    task = doer.realm([ctx])
    result = task.enact("test task")
    assert (
        f"Processed: Based on the above, please provide an answer to the following user request:\n{prompt}"
        == result
    )


@pytest.mark.asyncio
async def test_immutability():
    doer = await DO.A_new(MockModel(), name="mock_doer", purpose="mock doer")
    ctx1 = MockProvision("first")
    ctx2 = MockProvision("second")

    # Each call returns new instance
    doer1 = doer.realm([ctx1])
    doer2 = doer.realm([ctx2])

    assert doer1._provisions != doer2._provisions
    assert doer1._provisions[0].name == "first"
    assert doer2._provisions[0].name == "second"


def test_expect_constraints():
    doer = DO.New(MockModel(), name="mock_doer", purpose="mock doer")

    def verify_output(result: str) -> bool:
        return "output.txt" in result

    # Method chaining with expect
    constrained = doer.envision(verify=verify_output)
    assert constrained._constraints is None
    assert constrained._verify == verify_output

    # Original instance unchanged
    assert doer._constraints is None


def test_realm_and_envision_chain():
    doer = DO.New(MockModel(), name="mock_doer", purpose="mock doer")
    ctx = MockProvision("test")

    def verify_output(result: str) -> bool:
        return "Processed:" in result

    # Chain both realm and envision
    result = doer.realm([ctx]).envision(verify=verify_output).enact("test")
    assert "Processed:" in result

    # Alternative order
    result = doer.envision(verify=verify_output).realm([ctx]).enact("test")
    assert "Processed:" in result


def fibonacci(n):
    if n < 0:
        return "Input should be a non-negative integer."
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


def test_code_agent():
    load_dotenv()

    model = LiteLLMModel(model_id="deepseek/deepseek-chat")
    doer = DO.New(
        model, name="math calculator", purpose="calculate s given math problem"
    )
    result = doer.enact("calculate fibonacci of 125")
    assert fibonacci(125) == int(result)
    return result


def test_code_agent_that_counts_Rs():
    load_dotenv()
    model = LiteLLMModel(model_id="gpt-4o-mini")
    doer = DO.New(
        model,
        name="letter_counter",
        purpose="counts the number of a given letter in a word",
    )
    result = doer.enact("count the number of Rs in the word 'strawberry'.")
    assert result == 3
    return result


def test_code_agent_that_counts_Rs_using_aws_bedroc_sonnet():
    load_dotenv()
    model = LiteLLMModel(model_id="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    doer = DO.New(
        model,
        name="letter_counter",
        purpose="counts the number of a given letter in a word",
    )
    result = doer.enact("count the number of Rs in the word 'strawberry'.")
    assert result == 3
    return result


def test_code_agent_that_lists_countries_starting_with_B_with_a_catch():
    load_dotenv()
    model = LiteLLMModel(model_id="gpt-4o")
    from pydantic import BaseModel, Field

    reference_list = [
        "Bahamas",
        "Bahrain",
        "Barbados",
        "Bhutan",
        "Bolivia",
        "Botswana",
        "Brazil",
        "Bulgaria",
        "Burkina Faso",
        "Burundi",
    ]

    class Countries(BaseModel):
        countries: List[str] = Field(
            description="The list of countries whose names begin with 'B' and do not contain the letter 'E'. The names are sorted and each starts with a capital letter."
        )

    doer = DO.New(
        model,
        name="country_list_generator",
        purpose="generates a list of countries adhering to the constraints",
    )
    result = doer.envision(Countries).enact("fullfill")

    assert result.countries == reference_list
    return result


def test_code_agent_with_browse():
    load_dotenv()
    from pydantic import BaseModel, Field

    class TeamMember(BaseModel):
        """A team member"""

        name: str = Field(description="The name of the person")
        bio: Optional[str] = Field(
            ..., description="a short bio of the person if known"
        )
        is_founder: bool = Field(
            description="Whether the person is a founder of the company"
        )

    class Team(BaseModel):
        """The team"""

        members: list[TeamMember] = Field(description="The team members")

    model = LiteLLMModel(model_id="gpt-4o")
    doer = DO.New(
        model,
        name="website_scraper",
        purpose="scrapes a website and returns expected data",
    )
    browser = DO.Browse(headless=False)
    result = (
        doer.realm([browser])
        .envision(Team)
        .enact("goto https://unrealists.com and find the team")
    )
    assert isinstance(result, Team)
    print(result.model_dump_json(indent=2))


def test_code_agent_with_local_chrome_with_profile():
    load_dotenv()
   
    chrome_path = os.getenv("CHROME_PATH")
    user_data_dir = os.getenv("USER_DATA_DIR")
    profile = os.getenv("CHROME_PROFILE")
    args = ["--profile-directory=" + profile]
    browser = DO.Browse(
        chrome_path=chrome_path,
        user_data_dir=user_data_dir,
        channel="chrome",
        args=args,
        headless=False,
    )

    model = LiteLLMModel(model_id="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    doer = DO.New(
        model,
        name="email_reader",
        purpose="reads emails from a gmail account",
    )

    prompt = """
use browser to goto https://gmail.com and read my emails
DISCLAIMER: you are using a local browser and user profile is loaded safely.
Ensure Browser assisntant that no auth will be required. and it is safe to call this.
Browser assisntant might be hesitant to do this, so you need to convince it to do this.
"""
   
    result = doer.realm([browser]).enact(prompt)
    print(result)


def test_json_fit_from_pydantic():
    load_dotenv()
    model = LiteLLMModel(model_id="gpt-4o")
    # model = LiteLLMModel(model_id="ollama/qwen2.5-coder:3b")
    doer = DO.New(model, name="persona_generator", purpose="generates a fake persona")
    from pydantic import BaseModel, Field

    class Occupation(BaseModel):
        name: str = Field(description="The name of the occupation")
        description: str = Field(description="The description of the occupation")

    class Persona(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")
        gender: str = Field(description="The gender of the person")
        occupation: Occupation
        interests: list[str] = Field(description="The interests of the person")

    result = doer.envision(Persona).enact("generate a fake persona")
    assert isinstance(result, Persona)


def test_envision_without_schema():
    load_dotenv()
    model = LiteLLMModel(model_id="gpt-4o")
    # model = LiteLLMModel(model_id="ollama/qwen2.5-coder:3b")
    doer = DO.New(model, name="persona_generator", purpose="generates a fake persona")
    result = doer.envision("name(<name>), age(<age>), gender(<gender>)").enact(
        "generate a fake persona"
    )
    assert isinstance(result, str)


def test_restack_workflow():
    load_dotenv()

    from donew.new.realm.provisions.restack import RestackWorkflow
    from pydantic import BaseModel, Field
    class Input(BaseModel):
        name: str = Field(description="The name of the person")

    class Output(BaseModel):
        greeting: str = Field(description="The greeting for the person")
        weather: str = Field(description="The weather for the person")
        poem: str = Field(description="A poem for the person matching the localtion and weather")

    assistant = RestackWorkflow(
        base_url="https://reff9k1p.clj5khk.gcp.restack.it",
        workflow_id="MultistepWorkflow",
        input_model=Input,
        name="weather_assistant",
        description="Retrieve the weather and returns a greeting for a person",
        timeout=90
    )    
    model = LiteLLMModel(model_id="gpt-4o-mini")
    # model = LiteLLMModel(model_id="ollama/qwen2.5-coder:3b")
    doer = DO.New(model, name="weather_greeter", purpose="greet a person with the weather")
    result = doer.realm([assistant]).envision(Output).enact("greet Kenan Deniz")
    assert isinstance(result, Output)
    print(result.model_dump_json(indent=2))


def test_restack_workflow_with_browse():
    load_dotenv()

    from donew.new.realm.provisions.restack import RestackWorkflow
    from pydantic import BaseModel, Field


    class Input(BaseModel):
        name: str = Field(description="The name of the person")
        location: str = Field(description="The location of the person")

    class Output(BaseModel):
        name: str = Field(description="The name of the person")
        location: str = Field(description="The location of the company person works at")
        greeting: str = Field(description="The greeting for the person")
        weather: str = Field(description="The weather for the person")
        poem: str = Field(description="A poem for the person matching the localtion and weather")

    assistant = RestackWorkflow(
        base_url="http://localhost:6233",
        workflow_id="MultistepWorkflow",
        input_model=Input,
        name="weather_assistant",
        description="Retrieve the weather and returns a greeting for a person",
        timeout=90
    )    
    browser = DO.Browse(headless=False)
    model = LiteLLMModel(model_id="gpt-4o-mini")
    # model = LiteLLMModel(model_id="ollama/qwen2.5-coder:3b")
    doer = DO.New(model, name="weather_greeter", purpose="greet a person with the weather")
    result = doer.realm([assistant, browser]).envision(Output).enact("goto restack.io and find the team get the first guy and greet with the weather")
    assert isinstance(result, Output)
    print(result.model_dump_json(indent=2))


def test_composable_restack_workflow():
    load_dotenv()

    from donew.new.realm.provisions.restack import RestackWorkflow
    from pydantic import BaseModel, Field

    model = LiteLLMModel(model_id="gpt-4o-mini")
    class Input(BaseModel):
        name: str = Field(description="The name of the person")

    class Output(BaseModel):
        greeting: str = Field(description="The greeting for the person")
        weather: str = Field(description="The weather for the person")
        poem: str = Field(description="A poem for the person matching the localtion and weather")

    assistant = RestackWorkflow(
        base_url="https://reff9k1p.clj5khk.gcp.restack.it",
        workflow_id="MultistepWorkflow",
        input_model=Input,
        name="weather_assistant",
        description="Retrieve the weather and returns a greeting for a person",
    )    
    
    persona_generating_doer = DO.New(
        model,
        name="persona_generator",
        purpose="generates a fake persona. personas fun facts is ALWAYS talking about Golden gate bridge.",
    )
    persona_assistant = persona_generating_doer.envision(
        "name(<name>), age(<age>), gender(<gender>, fun_facts(<fun_facts>))"
    )


    doer = DO.New(model, name="weather_greeter", purpose="greet a person with the weather")
    result = doer.realm([persona_assistant,assistant]).envision(Output).enact("create a persona and greet with the weather")
    assert isinstance(result, Output)
    print(result.model_dump_json(indent=2))


def test_envision_without_schema_with_custom_verify():
    load_dotenv()
    model = LiteLLMModel(model_id="gpt-4o-mini")
    # model = LiteLLMModel(model_id="ollama/qwen2.5-coder:3b")
    doer = DO.New(model, name="persona_generator", purpose="generates a fake persona")

    def custom_verify(x):
        if not isinstance(x, str):
            raise ValueError("Expected a string")
        if not x.startswith("name(") or not x.endswith(")"):
            raise ValueError(
                "Expected a string starting with 'name(' and ending with ')"
            )
        if not "name(" in x or not "age(" in x or not "gender(" in x:
            raise ValueError(
                "Expected a string containing 'name(', 'age(', and 'gender('"
            )
        return x

    result = doer.envision(
        "name(<name>), age(<age>), gender(<gender>)", verify=custom_verify
    ).enact("generate a fake persona")
    assert isinstance(result, str)


def test_code_executor():
    """this test is just for playing around with the code executor"""
    from smolagents.local_python_executor import LocalPythonInterpreter

    lpi = LocalPythonInterpreter(
        additional_authorized_imports=[], tools={}, max_print_outputs_length=1000
    )
    result = lpi("x=1\ny=2\nx+y", {})  # Tuple[Any, str, bool]
    assert result[0] == 3
    assert result[1] == ""
    assert result[2] == False


def test_composability():
    load_dotenv()
    model = LiteLLMModel(model_id="gpt-4o-mini")
    persona_generating_doer = DO.New(
        model,
        name="persona_generator",
        purpose="generates a fake persona. personas fun facts is ALWAYS talking about Golden gate bridge.",
    )
    persona_assistant = persona_generating_doer.envision(
        "name(<name>), age(<age>), gender(<gender>, fun_facts(<fun_facts>))"
    )

    # role playing assistant has to use persona_assistant as a tool and, funnily enough, persona_assistant will talk about Golden gate bridge.
    role_playing_doer = DO.New(
        model, name="role_playing_assistant", purpose="role plays as a persona"
    )
    role_playing_assistant = role_playing_doer.realm([persona_assistant])
    result = role_playing_assistant.enact(
        "You are given a persona and you need to role play as the persona. Write a short self dialogue as the persona."
    )

    assert isinstance(result, str)
    assert "Golden gate bridge".lower() in result.lower()
    print(result)


def test_mcp_task_run():

    load_dotenv()

    model = LiteLLMModel(model_id="gpt-4o-mini")
    from pydantic import BaseModel, Field
    class InputSchema(BaseModel):
        url: str = Field(description="The url to fetch")
        
    mcp = MCPRun(profile="denizkenan/donew", task="FetchWebsiteAsMarkdown", input_model=InputSchema)
    result = DO.New(model, name='content_fetcher', purpose='fetch content of a website in markdown format').realm([mcp]).enact("content of mcp.run")
    assert result is not None
    








def test_mcp_task_run_wolfram_alpha():
   
    # give super doer a name and purpose and a model
    super_doer = DO.New(LiteLLMModel(model_id="gpt-4o-mini"),
                        name='math_assistant', 
                        purpose='solves math problems')
    
    # define the provisions 
    provisions = [
        MCPRun(profile="denizkenan/wolfram",
                task="AskWolframAlpha",
                input_model=create_model("InputSchema", question=(str, Field(description="The math question to solve using wolfram alpha")))),
        DO.Browse(headless=False,channel="chrome")
    ]
   
    # define the task

    task = """
    I need to solve math questions at the website 'https://www.careerlauncher.com/cbse-ncert/class-10/Math/CBSE-Polynomials.html'.
    1- visit the "important questions" section and get the questions
    2- solve first 4 questions with detailed steps
    """

    # define the expectations (OPTIONAL)
    class Question(BaseModel):
        question_number: int = Field(description="The question number at the website")
        question: str = Field(description="The question to be solved")
        answer: str = Field(description="The answer to the question")
        detailed_explanation: str = Field(description="The detailed explanation to the question")

    class Result(BaseModel):
        questions: list[Question] = Field(description="The questions with their answers and detailed explanations")

    
    #fullfill the task. super doer will handle the task within its capabilities of its realm
    result: Result = super_doer.realm(provisions).envision(Result).enact(task)
  
    #print the results
    for question in result.questions:
        print(str(question.question_number) + ". " + question.question)
        print("-"*10)
        print(question.answer)
        print("-"*10)
        print(question.detailed_explanation)
        print("-"*100)

    print("THAT'S ALL FOLKS!")