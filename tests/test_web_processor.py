import time
from dotenv import load_dotenv
import pytest
from donew import DO, KeyValueSection, TableSection
import asyncio
import json
from typing import cast, TypedDict, Dict, Any

from donew.utils import run_sync


def test_web_processing_docs():
    """Test web processing through DO.See interface"""
    docs = DO.Documentation("browse")
    assert docs is not None
    assert len(docs) > 0
    print(docs)


def test_web_processing(httpbin_url, httpbin_available):
    """Test web processing through DO.See interface"""
    result = DO.Browse(f"{httpbin_url}/")
    assert result is not None
    assert result._current_page().is_live()
    result.close()


def test_cookie_management(httpbin_url, httpbin_available):
    """Test cookie management using httpbin's cookie endpoints"""
    browser = DO.Browse(f"{httpbin_url}/cookies/set/test_cookie/test_value")

    try:
        # Verify cookie was set
        cookies = browser.cookies()
        assert any(
            c["name"] == "test_cookie" and c["value"] == "test_value" for c in cookies
        )

        # Navigate to cookies page to verify
        browser.navigate(f"{httpbin_url}/cookies")

        # Get page content to verify cookies
        content = browser.text()
        assert "test_cookie" in content
        assert "test_value" in content
    finally:
        browser.close()


def test_storage_management(httpbin_url, httpbin_available):
    """Test storage management using httpbin's HTML page"""
    browser = DO.Browse(f"{httpbin_url}/html")

    try:
        # Set storage values
        browser.storage(
            {
                "localStorage": {"test_key": "test_value"},
                "sessionStorage": {"session_key": "session_value"},
            }
        )

        # Verify storage values
        storage = browser.storage()
        assert storage["localStorage"]["test_key"] == "test_value"
        assert storage["sessionStorage"]["session_key"] == "session_value"

        # Navigate to another page and verify storage persists
        browser.navigate(f"{httpbin_url}/")
        new_storage = browser.storage()
        assert new_storage["localStorage"]["test_key"] == "test_value"
        assert new_storage["sessionStorage"]["session_key"] == "session_value"
    finally:
        browser.close()


def test_http_methods(httpbin_url, httpbin_available):
    """Test different HTTP methods using httpbin endpoints"""
    browser = DO.Browse(f"{httpbin_url}/forms/post")

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
            browser.type(element_id, f"test_{label}")

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
            browser.click(submit_button)
            time.sleep(1)  # Wait for form submission

            # Verify we got redirected to the result page
            current_url = browser._current_page().pw_page().url
            assert "/post" in current_url

            # Get the response content
            content = browser.text()
            assert "test_" in content  # Verify our test data is in the response
    finally:
        browser.close()


def test_response_headers(httpbin_url, httpbin_available):
    """Test response headers using httpbin's headers endpoint"""
    browser = DO.Browse(f"{httpbin_url}/headers")

    try:
        # Get page content
        content = browser.text()

        # Parse the JSON response
        headers = json.loads(content)

        # Verify basic headers are present
        assert "headers" in headers
        assert "User-Agent" in headers["headers"]
        assert "Host" in headers["headers"]
    finally:
        browser.close()


def test_status_codes(httpbin_url, httpbin_available):
    """Test different HTTP status codes using httpbin's status endpoints"""
    browser = DO.Browse(f"{httpbin_url}/status/200", {"headless": True})

    try:
        # Test successful response
        assert browser._current_page().is_live()

        # Navigate to a 404 page
        browser.navigate(f"{httpbin_url}/status/404")
        # The page should still be live even with 404
        assert browser._current_page().is_live()

        # Get the status code using JavaScript
        status_code = browser.evaluate(
            "window.performance.getEntries()[0].responseStatus"
        )
        assert status_code == 404
    finally:
        browser.close()


def test_image_processing(httpbin_url, httpbin_available):
    """Test image processing using httpbin's image endpoints"""
    browser = DO.Browse(f"{httpbin_url}/image/png")

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
        browser.navigate(f"{httpbin_url}/image/jpeg")
        time.sleep(1)
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
        browser.navigate(f"{httpbin_url}/image/svg")
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
        browser.close()


def test_element_annotation(httpbin_url, httpbin_available):
    """Test web element annotation functionality"""
    browser = DO.Browse(f"{httpbin_url}/forms/post")

    try:
        time.sleep(1)  # Wait for page load

        # Enable annotations
        browser.toggle_annotation(True)
        time.sleep(1)

        # Verify annotations are added
        script = "document.querySelectorAll('.DoSee-highlight').length"
        highlight_count = browser.evaluate(script)
        assert highlight_count > 0, "Annotations were not properly added"

        # Disable annotations
        browser.toggle_annotation(False)
        time.sleep(1)

        # Verify annotations are removed
        highlight_count = browser.evaluate(script)
        assert highlight_count == 0, "Annotations were not properly removed"

    finally:
        browser.close()


def test_browser_state(httpbin_url, httpbin_available):
    """Test browser state reporting functionality"""
    browser = DO.Browse(f"{httpbin_url}/forms/post")

    try:
        time.sleep(1)

        # Get initial state
        state = run_sync(browser._get_state_dict())

        # Verify state structure
        assert "sections" in state
        assert len(state["sections"]) == 2
        assert state["sections"][0]["name"] == "Timeline"
        assert state["sections"][1]["name"] == "Current State"

        # Verify page info
        section = state["sections"][1]
        assert section["type"] == "keyvalue"
        kv_section = cast(KeyValueSection, section)
        page_data = cast(Dict[str, str], kv_section["data"]["Page"])
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
            browser.type(input_id, "test_input")
            time.sleep(0.5)

            # Get updated state
            new_state = run_sync(browser._get_state_dict())

            # Verify interaction is recorded
            timeline_section = cast(TableSection, new_state["sections"][0])
            assert any("test_input" in str(row) for row in timeline_section["rows"])
    finally:
        browser.close()


def test_browser_config(httpbin_url, httpbin_available):
    load_dotenv()
    import os
    chrome_path = os.getenv("CHROME_PATH")
    user_data_dir = os.getenv("USER_DATA_DIR")
    profile=os.getenv("CHROME_PROFILE")
    args = ["--profile-directory=" + profile]
    DO.Config(chrome_path=chrome_path, user_data_dir=user_data_dir, channel="chrome",args=args, headless=False)
    browser = DO.Browse("https://docs.google.com")
    time.sleep(30)  # Sleep for 30 seconds
    browser.close()


def test_knowledge_graph_extraction(httpbin_url, httpbin_available):
    """Test Knowledge Graph extraction from web content"""
    # First browse to a page with structured content
    browser = DO.Browse(
        f"{httpbin_url}/html"
    )  # local version of https://httpbin.org/html

    try:
        # Get the page content
        result = browser.analyze()

        entities = result["entities"]

        # Verify basic HTML structure entities
        assert any(
            e["text"] == "Herman Melville" and e["label"] == "Person"
            for e in entities
            if e["text"] == "Herman Melville"
        )

        # Test querying the knowledge graph
        query_result = browser.query(
            "MATCH (e:Entity) WHERE e.label = $label RETURN e.text as name",
            params={"label": "Person"},
        )

        # Verify query returns expected results
        assert "Herman Melville" in [record.get("name") for record in query_result]

    finally:
        browser.close()
