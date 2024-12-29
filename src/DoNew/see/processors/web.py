import time
from typing import List, Dict, Any, Sequence, Union, Optional, Tuple
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Browser, Page


from . import BaseProcessor, BaseTarget, manual, public


@dataclass
class ElementMetadata:
    """Rich element metadata incorporating parsing patterns."""

    element_id: int
    element_name: str  # HTML tag name
    element_label: Optional[str]  # HTML label attribute
    element_html: str  # Opening tag HTML
    xpath: str  # Unique XPath
    bounding_box: Optional[Dict[str, float]]
    is_interactive: bool
    element_type: str  # button, link, input, icon, text
    attributes: Dict[str, str]  # All HTML attributes
    computed_styles: Optional[Dict[str, str]]  # Key CSS properties
    listeners: List[str]  # Event listeners
    parent_id: Optional[int]  # Parent element ID
    children_ids: List[int]  # Child element IDs
    state: Dict[str, Any]  # Element state


@dataclass
class Interaction:
    """Record of an interaction with a page element."""

    element_id: int
    interaction_type: str  # e.g., click, type
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebPage(BaseTarget):
    """Manages individual page state and elements."""

    _elements: Dict[int, ElementMetadata] = field(default_factory=dict)
    _interaction_history: List[Interaction] = field(default_factory=list)
    _page: Optional[Page] = None
    _annotation_enabled: bool = False

    async def process(self, url: str) -> "WebPage":
        """Process a webpage and extract its elements.

        Args:
            url: The URL to process.

        Returns:
            WebPage: The processed WebPage instance.
        """
        if not self.is_live():
            raise ValueError("Page is not live")
        if not self._page:
            raise ValueError("No page object available")

        await self._page.goto(url)

        # Log navigation as an interaction
        self._interaction_history.append(
            Interaction(
                element_id=-1,  # No element for navigation
                interaction_type="navigate",
                timestamp=time.time(),
                data={"url": url},
            )
        )

        # Inject and execute element detection script
        with open(
            "src/DoNew/see/processors/web/scripts/element_detection.js", "r"
        ) as f:
            script = f.read()
        elements = await self._page.evaluate(script)

        # Store detected elements
        self._elements = {
            id: ElementMetadata(**metadata) for id, metadata in elements.items()
        }

        # Enable annotations by default
        if self._annotation_enabled:
            await self.toggle_annotation(True)

        return self

    def elements(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[int, ElementMetadata]:
        """Get all elements, optionally filtered by bounding box.

        Args:
            bbox (Tuple[float, float, float, float], optional): Bounding box filter (x1, y1, x2, y2).

        Returns:
            Dict[int, ElementMetadata]: A dictionary of element IDs to metadata.
        """
        if bbox:
            return {
                id: elem
                for id, elem in self._elements.items()
                if elem.bounding_box
                and all(
                    elem.bounding_box[key] >= bbox[i]
                    for i, key in enumerate(["x", "y", "width", "height"])
                )
            }
        else:
            return self._elements

    def interactions(self) -> List[Interaction]:
        """Get all interactions"""
        return self._interaction_history

    def pw_page(self) -> Page:
        if not self._page:
            raise ValueError("No page object available")
        elif self._page.is_closed():
            raise ValueError("Page is closed")
        else:
            return self._page

    async def click(self, element_id: int):
        """Click an element.

        Args:
            element_id (int): The ID of the element to click.
        """
        if not self._page:
            raise ValueError("No live page connection")

        element = self._elements.get(element_id)
        if not element:
            raise ValueError(f"No element found with ID {element_id}")

        await self._page.click(element.xpath)
        self._interaction_history.append(Interaction(element_id, "click", time.time()))

    async def type(self, element_id: int, text: str):
        """Type text into an element.

        Args:
            element_id (int): The ID of the input element.
            text (str): The text to type.
        """
        if not self._page:
            raise ValueError("No live page connection")

        element = self._elements.get(element_id)
        if not element:
            raise ValueError(f"No element found with ID {element_id}")

        await self._page.fill(element.xpath, text)
        self._interaction_history.append(
            Interaction(element_id, "type", time.time(), {"text": text})
        )

    def is_live(self) -> bool:
        try:
            self.pw_page()
            return True
        except ValueError:
            return False

    def disconnect(self):
        """Disconnect from the page"""
        if self._page:
            self._page = None

    async def image(
        self,
        element_id: Optional[int] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        viewport: Optional[bool] = None,
    ) -> bytes:
        """Get element's image content

        Args:
            element_id: The ID of the element to get the image from. If None, gets the entire page.
            bbox: A tuple of (x1, y1, x2, y2) to crop the image to.
            viewport: Whether to get the image of the viewport or the entire page. Only applies if element_id is None.
        """
        if element_id:
            element = self._elements.get(element_id)
            if not element:
                raise ValueError(f"No element found with ID {element_id}")
            result = await self.pw_page().locator(element.xpath).screenshot()
        elif element_id is None and bbox is None:
            result = await self.pw_page().screenshot(full_page=viewport)
        elif bbox is not None:
            # Calculate clip dimensions from bbox coordinates
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # Use Playwright's clip option for precise region capture
            result = await self.pw_page().screenshot(
                clip={"x": x1, "y": y1, "width": width, "height": height}
            )
        else:
            raise ValueError("Either element_id or bbox must be provided")

        return result

    async def text(
        self,
        element_id: Optional[int] = None,
    ) -> str:
        """Get element's text content with interactive elements marked.
        Args:
            element_id: The ID of the element to get the text from. If None, gets the page content.
        Returns:
            str: The text content with interactive elements marked with [id@type#subtype].
        """
        if not self._page:
            raise ValueError("No live page connection")

        # First, temporarily modify interactive elements to show their IDs and types
        with open("src/DoNew/see/processors/web/scripts/text_markers.js", "r") as f:
            script = f.read()
        num_modified = await self._page.evaluate(script)

        try:
            # Get text content
            if element_id:
                id = str(element_id)
                element = self._elements.get(id)  # type: ignore
                if not element:
                    raise ValueError(f"No element found with ID {element_id}")
                result = await self.pw_page().locator(element.xpath).inner_text()
            else:
                result = await self.pw_page().locator("body").inner_text()

            return result

        finally:
            # Restore original state
            with open(
                "src/DoNew/see/processors/web/scripts/restore_text_markers.js", "r"
            ) as f:
                restore_script = f.read()
            await self._page.evaluate(restore_script)

    def _from_bbox_to_text(
        self, bboxes: Tuple[str, List[Tuple[int, int, int, int]]]
    ) -> str:
        text, _bboxes = bboxes
        # Ensure 'text' is a list of strings
        if isinstance(text, str):
            text_list = text.strip().split()
        else:
            text_list = text

        # Check if lengths match
        if len(text_list) != len(_bboxes):
            print("Warning: number of text elements and bounding boxes do not match.")
            return " ".join(text_list) if isinstance(text_list, list) else text_list

        # Create list of (text, bbox) tuples
        text_bbox_pairs = list(zip(text_list, _bboxes))

        # Define functions to compute the y-center and height of the bbox
        def y_center(bbox):
            x1, y1, x2, y2 = bbox
            return (y1 + y2) / 2

        def bbox_height(bbox):
            x1, y1, x2, y2 = bbox
            return abs(y2 - y1)

        # Sort the text_bbox_pairs by y-center (top to bottom)
        text_bbox_pairs.sort(key=lambda item: y_center(item[1]))

        # Group the text_bbox_pairs into lines based on y-coordinate proximity
        lines = []
        current_line = []
        current_y = None
        line_threshold = 0.5  # Proportion of bbox height to consider same line

        for text_elem, bbox in text_bbox_pairs:
            yc = y_center(bbox)
            h = bbox_height(bbox)
            if current_y is None:
                current_y = yc
                current_line.append((text_elem, bbox))
            else:
                if abs(yc - current_y) <= h * line_threshold:
                    current_line.append((text_elem, bbox))
                else:
                    # Sort the current line by x-coordinate
                    current_line.sort(key=lambda item: (item[1][0] + item[1][2]) / 2)
                    lines.append(current_line)
                    current_line = [(text_elem, bbox)]
                    current_y = yc

        # Add the last line
        if current_line:
            current_line.sort(key=lambda item: (item[1][0] + item[1][2]) / 2)
            lines.append(current_line)

        # Concatenate text within lines and join lines
        line_texts = [" ".join([text for text, bbox in line]) for line in lines]
        concatenated_text = "\n".join(line_texts)

        return concatenated_text

    async def scroll(self, element_id: int):
        """Scroll element into view"""
        if not self._page:
            raise ValueError("No live page connection")

        element = self._elements.get(element_id)
        if not element:
            raise ValueError(f"No element found with ID {element_id}")

        await self._page.evaluate(
            f"document.querySelector(\"[data-dosee-element-id='{element_id}']\").scrollIntoView()"
        )
        self._interaction_history.append(Interaction(element_id, "scroll", time.time()))

    async def cookies(
        self, cookies: Optional[Dict[str, str]] = None
    ) -> Sequence[Dict[str, str]]:
        """Gets or sets cookies for the current browser context using Playwright's native cookie handling.
        Direct passthrough to Browser's context.cookies() and context.add_cookies() methods.

        **Parameters**
            cookies : Dict[str, str], optional
                Cookie dictionary to set using Playwright's add_cookies.
                If None, returns current cookies via Playwright's cookies() method.

        **Returns**
            Dict[str, str]
                Dictionary of current cookies from Playwright's context.cookies()

        **Raises**
            ValueError
                If no live page connection exists

        **Usage**
        ```python
        # Get current  cookies
        current_cookies = await processor.cookies()

        # Set Playwright cookies and get updated state
        new_cookies = await processor.cookies({
            "session": "abc123",
            "user_id": "12345"
        })
        ```
        """
        if not self._page:
            raise ValueError("No live page connection")

        if cookies is not None:
            await self._page.context.add_cookies(cookies)  # type: ignore

        return await self._page.context.cookies()  # type: ignore

    async def storage(
        self, storage_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Gets or sets storage state (localStorage and sessionStorage) for the current page.

        **Parameters**
            storage_state : Dict[str, Any], optional
                Storage state to set. Should be in format:
                {
                    "localStorage": {"key": "value", ...},
                    "sessionStorage": {"key": "value", ...}
                }
                If None, returns current storage state.

        **Returns**
            Dict[str, Any]
                Dictionary containing current localStorage and sessionStorage state:
                {
                    "localStorage": {...},
                    "sessionStorage": {...}
                }

        **Raises**
            ValueError
                If no live page connection exists

        **Usage**
        ```python
        # Get current storage state
        state = await page.storage()

        # Set new storage state
        await page.storage({
            "localStorage": {"user": "jane"},
            "sessionStorage": {"token": "xyz789"}
        })
        ```
        """
        if not self._page:
            raise ValueError("No live page connection")

        if storage_state is not None:
            # Set localStorage
            if "localStorage" in storage_state:
                for key, value in storage_state["localStorage"].items():
                    await self._page.evaluate(
                        f"localStorage.setItem('{key}', '{value}')"
                    )

            # Set sessionStorage
            if "sessionStorage" in storage_state:
                for key, value in storage_state["sessionStorage"].items():
                    await self._page.evaluate(
                        f"sessionStorage.setItem('{key}', '{value}')"
                    )

        # Get current storage state
        return await self._page.evaluate(
            """() => {
            return {
                localStorage: Object.fromEntries(Object.entries(localStorage)),
                sessionStorage: Object.fromEntries(Object.entries(sessionStorage))
            };
        }"""
        )

    async def toggle_annotation(self, enabled: bool = True) -> None:
        """Toggle visual annotation of elements on the page.

        Args:
            enabled: Whether to enable or disable annotation
        """
        self._annotation_enabled = enabled
        if self._page:
            if enabled:
                await self._inject_annotation_styles()
                await self._highlight_elements()
            else:
                await self._remove_annotations()

    async def _inject_annotation_styles(self) -> None:
        """Inject CSS styles for element annotation"""
        with open(
            "src/DoNew/see/processors/web/scripts/highlight_styles.css", "r"
        ) as f:
            styles = f.read()
        await self._page.add_style_tag(content=styles)  # type: ignore

    async def _highlight_elements(self) -> None:
        """Add highlight overlays to all detected elements"""
        with open(
            "src/DoNew/see/processors/web/scripts/highlight_elements.js", "r"
        ) as f:
            script = f.read()
        await self._page.evaluate(script)  # type: ignore

    async def _remove_annotations(self) -> None:
        """Remove all element annotations from the page"""
        await self._page.evaluate(  # type: ignore
            "document.querySelectorAll('.DoSee-highlight').forEach(el => el.remove())"
        )

    async def close(self):
        """Close the page"""
        if self._page:
            await self._page.close()
            self._page = None

    def interaction_history(self) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Get page's interaction history as [(timestamp, action_type, metadata)]
        Metadata includes element info, data, and page URL
        """
        history = []

        # Process all interactions including navigation
        for interaction in self._interaction_history:
            if interaction.interaction_type == "navigate":
                metadata = {"url": interaction.data["url"]}
            else:
                element = self._elements.get(interaction.element_id)
                metadata = {
                    "element_type": element.element_type if element else None,
                    "element_label": element.element_label if element else None,
                    "xpath": element.xpath if element else None,
                    "data": interaction.data,
                }

            history.append(
                (interaction.timestamp, interaction.interaction_type, metadata)
            )

        return history

    async def evaluate(self, script: str) -> Any:
        return await self._page.evaluate(script)  # type: ignore


@dataclass
class WebBrowser(BaseTarget):
    """Manages browser session and page history."""

    _browser: Optional[Browser] = None
    _pages: List[WebPage] = field(default_factory=list)
    _headless: bool = True
    _interaction_history: List[Tuple[int, int]] = field(
        default_factory=list
    )  # (page_index, interaction_index)

    def _current_page(self) -> WebPage:
        """Internal method to get current page."""
        if not self._pages:
            raise ValueError(
                "No pages available Initiate a browser and add a page first"
            )
        return self._pages[-1]

    @public(order=1)
    async def navigate(self, url: str):
        """Navigate to a URL in a new page.

        Args:
            url (str): The URL to navigate to.
        """
        if not self._browser:
            raise ValueError("No browser session")
        current_page = self._current_page()
        pw_page = current_page.pw_page()
        current_page._page = None
        new_page = WebPage(
            _page=pw_page, _annotation_enabled=current_page._annotation_enabled
        )
        await new_page.process(url)
        self._pages.append(new_page)

    @public(order=2)
    @manual(
        template="Extends page annotation to browser level: {extendee}",
        extends=WebPage.toggle_annotation,
    )
    async def toggle_annotation(self, enabled: bool = True) -> None:
        """Toggle visual annotation of elements on the current page.

        Args:
            enabled: Whether to enable or disable annotation
        """
        if self._pages:
            await self._current_page().toggle_annotation(enabled)

    @public(order=2)
    @manual(
        template="Get/Set cookies from current page: {extendee}",
        extends=WebPage.cookies,
    )
    async def cookies(
        self, cookies: Optional[Dict[str, str]] = None
    ) -> Sequence[Dict[str, str]]:
        return await self._current_page().cookies(cookies)

    @public(order=3)
    @manual(
        template="Get/Set storage state from current page: {extendee}",
        extends=WebPage.storage,
    )
    async def storage(
        self, storage_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, str]]:
        return await self._current_page().storage(storage_state)

    @public(order=4)
    @manual(
        template="Click an element: {extendee}",
        extends=WebPage.click,
    )
    async def click(self, element_id: int):
        return await self._current_page().click(element_id)

    @public(order=5)
    @manual(
        template="Type text into an element: {extendee}",
        extends=WebPage.type,
    )
    async def type(self, element_id: int, text: str):
        return await self._current_page().type(element_id, text)

    @public(order=5)
    @manual(
        template="Get image content from an element: {extendee}",
        extends=WebPage.image,
    )
    async def image(
        self,
        element_id: Optional[int] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        viewport: Optional[bool] = None,
    ) -> bytes:
        return await self._current_page().image(element_id, bbox, viewport)

    @public(order=5)
    @manual(
        template="Get text content from an element: {extendee}",
        extends=WebPage.text,
    )
    async def text(
        self,
        element_id: Optional[int] = None,
    ) -> str:
        return await self._current_page().text(element_id)

    @public(order=6)
    async def close(self):
        """Close the browser and clean up resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._pages.clear()

    @public(order=7)
    @manual(
        template="Get all elements on the current page: {extendee}",
        extends=WebPage.elements,
    )
    def elements(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[int, ElementMetadata]:
        """Get all elements on the current page."""
        return self._current_page().elements(bbox)

    async def _get_state_dict(self) -> Dict[str, Any]:
        """Get browser state including page history and interactions."""
        current_page = self._current_page() if self._pages else None

        # Get element type counts if we have a current page
        element_counts = {"buttons": 0, "inputs": 0, "links": 0, "text": 0, "images": 0}
        if current_page:
            for elem in current_page._elements.values():
                if elem.element_type == "button":
                    element_counts["buttons"] += 1
                elif elem.element_type == "input":
                    element_counts["inputs"] += 1
                elif elem.element_type == "link":
                    element_counts["links"] += 1
                elif elem.element_type == "icon":
                    element_counts["images"] += 1
                else:
                    element_counts["text"] += 1

        # Build timeline from all pages' histories
        timeline_rows = []
        for page in self._pages:
            for timestamp, action_type, metadata in page.interaction_history():
                time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

                # Format action based on type
                if action_type == "navigate":
                    action = f"Navigated to {metadata['url']}"
                elif action_type == "type":
                    text = metadata["data"].get("text", "")
                    xpath = metadata.get("xpath", "unknown")
                    label = metadata.get("element_label", "")
                    element_desc = f'"{label}" ({xpath})' if label else xpath
                    action = f'Typed value: "{text}" to {element_desc}'
                elif action_type == "click":
                    xpath = metadata.get("xpath", "unknown")
                    label = metadata.get("element_label", "")
                    element_desc = f'"{label}" ({xpath})' if label else xpath
                    action = f"Clicked {element_desc}"
                else:
                    xpath = metadata.get("xpath", "unknown")
                    label = metadata.get("element_label", "")
                    element_desc = f'"{label}" ({xpath})' if label else xpath
                    action = f"Interacted with {element_desc}"

                timeline_rows.append([time_str, action])

        return {
            "sections": [
                {
                    "name": "Timeline",
                    "type": "table",
                    "headers": ["Time", "Action"],
                    "rows": timeline_rows,
                },
                {
                    "name": "Current State",
                    "type": "keyvalue",
                    "data": {
                        "Page": {
                            "URL": (
                                current_page.pw_page().url
                                if current_page and current_page.is_live()
                                else "disconnected"
                            ),
                            "Title": (
                                await current_page.pw_page().title()
                                if current_page and current_page.is_live()
                                else "N/A"
                            ),
                            "Element Count": (
                                str(len(current_page._elements))
                                if current_page
                                else "0"
                            ),
                        },
                        "Elements": {
                            "Interactive": f"{element_counts['buttons']} buttons, {element_counts['inputs']} inputs, {element_counts['links']} links",
                            "Text Elements": f"{element_counts['text']}",
                            "Images": f"{element_counts['images']}",
                        },
                        "Browser": {
                            "Active": str(bool(self._browser)),
                            "Pages in History": str(len(self._pages)),
                            "Cookies": f"{len(await current_page.cookies()) if current_page and current_page.is_live() else 0} active",
                        },
                    },
                },
            ]
        }

    @public(order=8)
    @manual(
        template="{extendee}",
        extends=WebPage.evaluate,
    )
    async def evaluate(self, script: str):
        return await self._current_page().evaluate(script)


class WebProcessor(BaseProcessor[Union[str, Page]]):
    """Main processor for web page analysis and interaction."""

    _headless: bool = True

    def __init__(self, headless: bool = True):
        self._headless = headless

    async def process(self, source: str) -> List[WebBrowser]:
        """Process a URL or Page object and return a WebBrowser target.

        Args:
            source (Union[str, Page]): The URL to process or an existing Page object.

        Returns:
            List[WebBrowser]: A list containing the WebBrowser instance.
        """
        # Initialize Playwright and browser
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self._headless)
        pw_page = await browser.new_page()
        web_page = WebPage(_page=pw_page)
        await web_page.process(source)

        web_browser = WebBrowser(
            _browser=browser, _pages=[web_page], _headless=self._headless
        )

        return [web_browser]
