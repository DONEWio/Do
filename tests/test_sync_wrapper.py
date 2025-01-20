import pytest
from donew import DO
from donew.new.doers.super import SuperDoer
import asyncio


class MockModel:
    async def generate(self, prompt: str, **kwargs) -> str:
        return f"Processed: {prompt}"


def test_sync_browse(httpbin_url, httpbin_available):
    """Test synchronous Browse functionality"""
    browser = DO.Browse(httpbin_url)
    assert browser is not None
    browser.close()  # Close browser to clean up


def test_sync_new():
    """Test synchronous New functionality"""
    config = {"model": MockModel()}
    doer = DO.New(config)
    assert isinstance(doer, SuperDoer)


@pytest.mark.asyncio
async def test_async_methods():
    """Test that async methods work in async context"""
    config = {"model": MockModel()}

    # Use the async methods directly
    browsers = await DO._browse_async(["http://example.com"])
    assert browsers is not None
    for browser in browsers:
        await browser.close()

    doer = await DO._new_async(config)
    assert isinstance(doer, SuperDoer)


def test_sync_in_async_error():
    """Test that using sync in async raises proper error"""
    with pytest.raises(RuntimeError, match=".*using DO's sync API inside.*"):

        async def run():
            return DO.Browse("http://example.com")

        asyncio.run(run())


def test_config():
    """Test that Config works"""
    DO.Config(headless=True)  # Should not raise any errors
