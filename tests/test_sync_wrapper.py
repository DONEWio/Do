import pytest
from donew import DO
from donew.new.doers.super import SuperDoer
import asyncio


class MockModel:
    async def generate(self, prompt: str, **kwargs) -> str:
        return f"Processed: {prompt}"


def test_sync_browse(httpbin_url, httpbin_available):
    """Test synchronous Browse functionality"""
    browser = DO.Browse()

    assert browser is not None
    browser.close()  # Close browser to clean up


def test_sync_new():
    """Test synchronous New functionality"""
    config = {"model": MockModel(),"name":"test", "purpose":"test"}
    doer = DO.New(**config)
    assert isinstance(doer, SuperDoer)


@pytest.mark.asyncio
async def test_async_methods():
    """Test that async methods work in async context"""
    config = {"model": MockModel(),"name":"test", "purpose":"test"}

    # Use the async methods directly
    browser = await DO.A_browse()
    assert browser is not None
    await browser.a_close()

    doer = await DO.A_new(**config)
    assert isinstance(doer, SuperDoer)

