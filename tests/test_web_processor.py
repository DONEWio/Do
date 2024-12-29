import pytest
from src.donew import DO
from src.donew.see.processors.web import WebBrowser
import asyncio
import json


@pytest.mark.asyncio
async def test_web_processing(httpbin_url, httpbin_available):
    """Test web processing through DO.See interface"""
    result = await DO.Browse(f"{httpbin_url}/")
    assert result is not None
    assert result._current_page().is_live()


@pytest.mark.asyncio
async def test_cookie_management(httpbin_url, httpbin_available):
    """Test cookie management using httpbin's cookie endpoints"""
    browser = await DO.Browse(f"{httpbin_url}/cookies/set/test_cookie/test_value")

    try:
        # Verify cookie was set
        cookies = await browser.cookies()
        assert any(
            c["name"] == "test_cookie" and c["value"] == "test_value" for c in cookies
        )

        # Navigate to cookies page to verify
        await browser.navigate(f"{httpbin_url}/cookies")

        # Get page content to verify cookies
        content = await browser.text()
        assert "test_cookie" in content
        assert "test_value" in content
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_storage_management(httpbin_url, httpbin_available):
    """Test storage management using httpbin's HTML page"""
    browser = await DO.Browse(f"{httpbin_url}/html")

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
        await browser.navigate(f"{httpbin_url}/")
        new_storage = await browser.storage()
        assert new_storage["localStorage"]["test_key"] == "test_value"
        assert new_storage["sessionStorage"]["session_key"] == "session_value"
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_http_methods(httpbin_url, httpbin_available):
    """Test different HTTP methods using httpbin endpoints"""
    browser = await DO.Browse(f"{httpbin_url}/forms/post")

    try:
        # Find form elements
        elements = browser.elements()

        # Find input fields and submit button
        input_fields = {
            elem.element_label or elem.attributes.get("name", ""): id
            for id, elem in elements.items()
            if elem.element_type == "input"
            and elem.attributes.get("type") in ["text", "email"]
        }

        # Fill out the form
        for label, element_id in input_fields.items():
            await browser.type(element_id, f"test_{label}")

        # Find and click submit button
        submit_button = next(
            (
                id
                for id, elem in elements.items()
                if elem.element_type == "button"
                and elem.attributes.get("type") == "submit"
            ),
            None,
        )

        if submit_button:
            await browser.click(submit_button)
            await asyncio.sleep(1)  # Wait for form submission

            # Verify we got redirected to the result page
            current_url = browser._current_page().pw_page().url
            assert "/post" in current_url

            # Get the response content
            content = await browser.text()
            assert "test_" in content  # Verify our test data is in the response
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_response_headers(httpbin_url, httpbin_available):
    """Test response headers using httpbin's headers endpoint"""
    browser = await DO.Browse(f"{httpbin_url}/headers")

    try:
        # Get page content
        content = await browser.text()

        # Parse the JSON response
        headers = json.loads(content)

        # Verify basic headers are present
        assert "headers" in headers
        assert "User-Agent" in headers["headers"]
        assert "Host" in headers["headers"]
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_status_codes(httpbin_url, httpbin_available):
    """Test different HTTP status codes using httpbin's status endpoints"""
    browser = await DO.Browse(f"{httpbin_url}/status/200")

    try:
        # Test successful response
        assert browser._current_page().is_live()

        # Navigate to a 404 page
        await browser.navigate(f"{httpbin_url}/status/404")
        # The page should still be live even with 404
        assert browser._current_page().is_live()

        # Get the status code using JavaScript
        status_code = await browser.evaluate(
            "window.performance.getEntries()[0].responseStatus"
        )
        assert status_code == 404
    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_image_processing(httpbin_url, httpbin_available):
    """Test image processing using httpbin's image endpoints"""
    browser = await DO.Browse(f"{httpbin_url}/image/png")

    try:
        # Get elements and find the image
        elements = browser.elements()
        img_elements = [
            (id, elem)
            for id, elem in elements.items()
            if elem.element_name == "img" or elem.element_name == "svg"
        ]
        assert len(img_elements) == 1, "Expected exactly one image element"
        img_id, img_elem = img_elements[0]

        # Verify image source for PNG
        assert img_elem.attributes.get("src", "").endswith("/image/png")

        # Test JPEG format
        await browser.navigate(f"{httpbin_url}/image/jpeg")
        elements = browser.elements()
        img_elements = [
            (id, elem)
            for id, elem in elements.items()
            if elem.element_name == "img" or elem.element_name == "svg"
        ]
        assert len(img_elements) == 1, "Expected exactly one image element"
        img_id, img_elem = img_elements[0]

        # Verify JPEG image source
        assert img_elem.attributes.get("src", "").endswith("/image/jpeg")

        # Test SVG format
        await browser.navigate(f"{httpbin_url}/image/svg")
        elements = browser.elements()
        img_elements = [
            (id, elem)
            for id, elem in elements.items()
            if elem.element_name == "img" or elem.element_name == "svg"
        ]
        assert len(img_elements) == 1, "Expected exactly one image element"
        img_id, img_elem = img_elements[0]

        # For SVG, either it's an <img> with src or an inline <svg>
        if img_elem.element_name == "img":
            assert img_elem.attributes.get("src", "").endswith("/image/svg")
        else:
            assert img_elem.element_name == "svg"

    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_element_annotation(httpbin_url, httpbin_available):
    """Test web element annotation functionality"""
    browser = await DO.Browse(f"{httpbin_url}/forms/post")

    try:
        await asyncio.sleep(1)  # Wait for page load

        # Enable annotations
        await browser.toggle_annotation(True)
        await asyncio.sleep(1)

        # Verify annotations are added
        script = "document.querySelectorAll('.DoSee-highlight').length"
        highlight_count = await browser._current_page().evaluate(script)
        assert highlight_count > 0, "Annotations were not properly added"

        # Disable annotations
        await browser.toggle_annotation(False)
        await asyncio.sleep(1)

        # Verify annotations are removed
        highlight_count = await browser._current_page().evaluate(script)
        assert highlight_count == 0, "Annotations were not properly removed"

    finally:
        await browser.close()


@pytest.mark.asyncio
async def test_browser_state(httpbin_url, httpbin_available):
    """Test browser state reporting functionality"""
    browser = await DO.Browse(f"{httpbin_url}/forms/post")

    try:
        await asyncio.sleep(1)

        # Get initial state
        state = await browser._get_state_dict()

        # Verify state structure
        assert "sections" in state
        assert len(state["sections"]) == 2
        assert state["sections"][0]["name"] == "Timeline"
        assert state["sections"][1]["name"] == "Current State"

        # Verify page info
        page_data = state["sections"][1]["data"]["Page"]
        assert "forms/post" in page_data["URL"]
        assert int(page_data["Element Count"]) > 0

        # Perform some interactions
        elements = browser.elements()
        input_id = next(
            (
                id
                for id, elem in elements.items()
                if elem.element_type == "input"
                and elem.attributes.get("type") in ["text", "email"]
            ),
            None,
        )

        if input_id:
            await browser.type(input_id, "test_input")
            await asyncio.sleep(0.5)

            # Get updated state
            new_state = await browser._get_state_dict()

            # Verify interaction is recorded
            timeline = new_state["sections"][0]["rows"]
            assert any("test_input" in str(row) for row in timeline)
    finally:
        await browser.close()
