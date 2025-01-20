import os
from dotenv import load_dotenv
import pytest
from typing import Any, Optional
from donew import DO
from donew.new.doers import BaseDoer
from donew.new.doers.super import SuperDoer
from smolagents import LiteLLMModel
from smolagents.models import ChatMessage, MessageRole


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
    config = {"model": MockModel()}
    doer = DO.New(config)
    assert isinstance(doer, SuperDoer)
    assert isinstance(doer, BaseDoer)


@pytest.mark.asyncio
async def test_method_chaining():
    load_dotenv()
    doer = await DO.A_new({"model": MockModel()})
    ctx = MockProvision("test")
    prompt = "test task"

    def verify_output(result: str) -> bool:
        expected = f"Processed: Based on the above, please provide an answer to the following user request:\n{prompt}"
        return expected == result

    # Method chaining

    doer = await DO.A_new({"model": MockModel()})
    result = await doer.realm([ctx]).envision({"verify": verify_output}).enact(prompt)
    assert (
        f"Processed: Based on the above, please provide an answer to the following user request:\n{prompt}"
        == result
    )

    # Alternative style
    task = doer.realm([ctx])
    result = await task.enact("test task")
    assert (
        f"Processed: Based on the above, please provide an answer to the following user request:\n{prompt}"
        == result
    )


@pytest.mark.asyncio
async def test_immutability():
    doer = await DO.A_new({"model": MockModel()})
    ctx1 = MockProvision("first")
    ctx2 = MockProvision("second")

    # Each call returns new instance
    doer1 = doer.realm([ctx1])
    doer2 = doer.realm([ctx2])

    assert doer1._provisions != doer2._provisions
    assert doer1._provisions[0].name == "first"
    assert doer2._provisions[0].name == "second"


def test_expect_constraints():
    doer = DO.New({"model": MockModel()})

    def verify_output(result: str) -> bool:
        return "output.txt" in result

    # Method chaining with expect
    constrained = doer.envision({"verify": verify_output})
    assert constrained._constraints is not None
    assert constrained._constraints["verify"] == verify_output

    # Original instance unchanged
    assert doer._constraints is None


def test_realm_and_envision_chain():
    doer = DO.New({"model": MockModel()})
    ctx = MockProvision("test")

    def verify_output(result: str) -> bool:
        return "Processed:" in result

    # Chain both realm and envision
    result = doer.realm([ctx]).envision({"verify": verify_output}).enact("test")
    assert "Processed:" in result

    # Alternative order
    result = doer.envision({"verify": verify_output}).realm([ctx]).enact("test")
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
    doer = DO.New({"model": model})
    result = doer.enact("calculate fibonacci of 125")
    assert fibonacci(125) == int(result)
    return result

def test_json_fit():
    load_dotenv()
    model = LiteLLMModel(model_id="deepseek/deepseek-chat")
    doer = DO.New({"model": model})
    format_json = dict(
        name = "<string>",
        age = "<int>",
        gender = "<string>",
        occupation = "<string>",
        interests = "<list[string]>",
    )
    import json
    result = doer.enact(f"generate a fake persona with the following json format: {json.dumps(format_json)}")
    assert isinstance(result, dict)
    assert len(result) == len(format_json)
    assert all(key in result for key in format_json)
    return result
