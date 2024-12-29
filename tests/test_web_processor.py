import pytest
from DoNew import DO
from DoNew.see.processors.web import WebBrowser
import asyncio


@pytest.mark.asyncio
async def test_web_processing():
    """Test web processing through DO.See interface"""

    result = await DO.Browse("https://httpbin.org/")
    assert result is not None
    assert result._current_page().is_live()


@pytest.mark.asyncio
async def test_cookie_management():
    """Test cookie management using httpbin.org's cookie endpoints"""
    DO.Config(
        headless=False,
    )
    browser = await DO.Browse("https://httpbin.org/cookies/set/test_cookie/test_value")

    try:
        # Verify cookie was set
        cookies = await browser.cookies()
        assert any(
            c["name"] == "test_cookie" and c["value"] == "test_value" for c in cookies
        )

        # Navigate to cookies page to verify
        await browser.navigate("https://httpbin.org/cookies")

        # Get page content to verify cookies
        content = await browser.text()
        assert "test_cookie" in content
        assert "test_value" in content
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_storage_management():
    """Test storage management using a real page"""

    browser = await DO.Browse("https://httpbin.org/html")

    try:
        # Set storage values
        await browser.storage(
            {
                "localStorage": {"test_key": "test_value"},
                "sessionStorage": {"session_key": "session_value"},
            }
        )

        # Verify storage values
        storage = await browser.storage()
        assert storage["localStorage"]["test_key"] == "test_value"
        assert storage["sessionStorage"]["session_key"] == "session_value"

        # Navigate to another page and verify storage persists
        await browser.navigate("https://httpbin.org/")
        new_storage = await browser.storage()
        assert (
            new_storage["localStorage"]["test_key"] == "test_value"
        )  # Storage should persist
        assert (
            new_storage["sessionStorage"]["session_key"] == "session_value"
        )  # Session storage should also persist
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_element_annotation():
    """Test web element annotation functionality"""
    # Configure with headless=False to see the visual annotations
    DO.Config(
        headless=False,
    )

    # Use GitHub's login page which has consistent elements
    browser = await DO.Browse("https://github.com/login")

    try:
        # Wait for page to load completely
        await asyncio.sleep(2)

        # Enable annotations
        await browser.toggle_annotation(True)

        await asyncio.sleep(1)  # Wait to see the change
        print(await browser.text())
        print(await browser.image())

        # Verify annotations are added
        script = "document.querySelectorAll('.DoSee-highlight').length"
        if browser._current_page()._page is not None:
            highlight_count = await browser._current_page().evaluate(script)
            assert highlight_count > 0, "Annotations were not properly added"

        # Disable annotations
        await browser.toggle_annotation(False)
        await asyncio.sleep(1)  # Wait to see the change

        # Verify annotations are removed
        highlight_count = await browser._current_page().evaluate(script)
        assert highlight_count == 0, "Annotations were not properly removed"

        # Re-enable annotations
        await browser.toggle_annotation(True)
        await asyncio.sleep(1)  # Wait to see the change

        # Verify annotations are added back
        highlight_count = await browser._current_page().evaluate(script)
        assert highlight_count > 0, "Annotations were not properly added back"

        # Keep browser open for a moment to see the annotations
        await asyncio.sleep(3)

    finally:
        # Clean up
        await browser.close()


@pytest.mark.asyncio
async def test_browser_state():
    """Test browser state reporting functionality"""
    DO.Config(
        headless=False,
    )
    browser = await DO.Browse("https://httpbin.org/forms/post")

    try:
        # Wait for page to load and print available elements for debugging
        await asyncio.sleep(1)

        # Debug: Print ALL elements first
        form_elements = browser.elements()
        print("\nAll elements:")
        for id, elem in form_elements.items():
            print(
                f"ID: {id}, Type: {elem.element_type}, HTML: {elem.element_html[:100]}"
            )

        # Try to find any input element that we can type into
        input_id = next(
            id
            for id, elem in form_elements.items()
            if elem.element_type == "input"
            and elem.attributes.get("type") in ["text", "email", "tel", "password"]
        )

        await browser.type(input_id, "John Doe")
        await asyncio.sleep(0.5)  # Let the interaction register

        # Get state
        state = await browser.state()
        print("\nCurrent State:")
        print(state)

        # Verify main sections exist
        assert "## Timeline" in state
        assert "## Current State" in state

        # Verify Timeline section format
        assert "| Time" in state
        assert "| Action" in state
        assert "John Doe" in state

        # Verify Current State subsections exist
        assert "### Page" in state
        assert "### Elements" in state
        assert "### Browser" in state

        # Verify Page section content
        assert "URL" in state
        assert "https://httpbin.org/forms/post" in state
        assert "Element Count" in state

        # Verify Elements section content
        assert "Interactive" in state
        assert "buttons" in state
        assert "inputs" in state
        assert "Text Elements" in state

        # Verify Browser section content
        assert "Active" in state
        assert "Pages in History" in state
        assert "Cookies" in state

        # Navigate to another page and verify state updates
        await browser.navigate("https://httpbin.org/")
        new_state = await browser.state()

        # Verify navigation updated the state
        assert "https://httpbin.org/" in new_state
        assert "### Page" in new_state  # Should still maintain structure
        assert "### Browser" in new_state
        assert "Pages in History" in new_state

    finally:
        await browser.close()
