import os
from dotenv import load_dotenv
import pytest
from typing import Any
from donew import DO
from donew.new.doers import BaseDoer
from donew.new.doers.super import SuperDoer
from smolagents import LiteLLMModel


class MockModel:
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
        return f"Processed: {prompt}" == result

    # Method chaining

    doer = await DO.A_new({"model": MockModel()})
    result = await doer.realm([ctx]).envision({"verify": verify_output}).enact(prompt)
    assert "Processed: test task" in result

    # Alternative style
    task = doer.realm([ctx])
    result = await task.enact("test task")
    assert "Processed: test task" in result


@pytest.mark.asyncio
async def test_immutability():
    doer = await DO.New({"model": MockModel()})
    ctx1 = MockProvision("first")
    ctx2 = MockProvision("second")

    # Each call returns new instance
    doer1 = doer.realm([ctx1])
    doer2 = doer.realm([ctx2])

    assert doer1._provisions != doer2._provisions
    assert doer1._provisions[0].name == "first"
    assert doer2._provisions[0].name == "second"


@pytest.mark.asyncio
async def test_expect_constraints():
    doer = await DO.New({"model": MockModel()})

    def verify_output(result: str) -> bool:
        return "output.txt" in result

    # Method chaining with expect
    constrained = doer.envision({"verify": verify_output})
    assert constrained._constraints is not None
    assert constrained._constraints["verify"] == verify_output

    # Original instance unchanged
    assert doer._constraints is None


@pytest.mark.asyncio
async def test_realm_and_envision_chain():
    doer = await DO.New({"model": MockModel()})
    ctx = MockProvision("test")

    def verify_output(result: str) -> bool:
        return "Processed:" in result

    # Chain both realm and envision
    result = await doer.realm([ctx]).envision({"verify": verify_output}).enact("test")
    assert "Processed:" in result

    # Alternative order
    result = await doer.envision({"verify": verify_output}).realm([ctx]).enact("test")
    assert "Processed:" in result


def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@pytest.mark.asyncio
async def test_code_agent():
    load_dotenv()

    model = LiteLLMModel(model_id="deepseek/deepseek-chat")
    doer = await DO.A_new({"model": model})
    browser = await DO.A_browse(["https://www.unrealists.com"])
    result = await doer.enact("calculate fibonacci of 10")
    assert fibonacci(10) == int(result)
