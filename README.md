# DoNew

A powerful Python package designed for AI agents to perform web processing, document navigation, and autonomous task execution. DoNew provides a high-level, agentic interface that makes it easy for AI systems to interact with web content and documents.

## Why DoNew?

DoNew is built with AI agents in mind, providing intuitive interfaces for:
- Autonomous web navigation and interaction
- Document understanding and processing
- Task execution and decision making
- State management and context awareness

## Features

- Browser automation using Playwright
- Web page processing and interaction
- Vision-related tasks and image processing
- Easy-to-use API for web automation
- Async support for better performance
- AI-friendly interfaces for autonomous operation

## Roadmap

### Current Features
- **DO.Browse**: Agentic web navigation
  - Autonomous webpage interaction
  - Element detection and manipulation
  - State awareness and context management
  - Cookie and storage handling
  - Visual debugging tools

### Coming Soon
- **DO.Read**: Agentic document navigation
  - PDF processing and understanding
  - Document structure analysis
  - Content extraction and processing
  - Cross-document reference handling

- **DO(...).New**: Agentic behavior execution
  - Task planning and execution
  - Decision making based on content
  - Multi-step operation handling
  - Context-aware actions

## Installation

1. Install the package:
```bash
pip install donew
```

2. Install required browsers (needed for web automation):
```bash
donew-install-browsers
```

## Quick Start

```python
import asyncio
from DoNew import DO

async def main():
    # Configure browser settings (optional)
    DO.Config(headless=True)  # Run in headless mode
    
    # Start agentic web navigation
    browser = await DO.Browse("https://example.com")
    
    try:
        # Analyze page content
        content = await browser.text()
        print("Page content:", content)
        
        # Get all interactive elements with their context
        elements = browser.elements()
        
        # Smart element detection (finds relevant input fields by context)
        input_fields = {
            elem.element_label or elem.attributes.get("name", ""): id
            for id, elem in elements.items()
            if elem.element_type == "input"
            and elem.attributes.get("type") in ["text", "email"]
        }
        
        # Autonomous form interaction
        for label, element_id in input_fields.items():
            await browser.type(element_id, f"test_{label}")
        
        # State management
        cookies = await browser.cookies()
        print("Current browser state (cookies):", cookies)
        
        # Context persistence
        await browser.storage({
            "localStorage": {"agent_context": "form_filling"},
            "sessionStorage": {"task_state": "in_progress"}
        })
        
        # Visual debugging (helps AI understand page state)
        await browser.toggle_annotation(True)
        
        # Get current state for decision making
        state = await browser._get_state_dict()
        print("Current agent state:", state)
        
    finally:
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: AI Agent Task Execution

```python
from DoNew import DO

async def search_and_extract(query: str):
    browser = await DO.Browse("https://example.com/search")
    try:
        # Find and interact with search form
        elements = browser.elements()
        search_input = next(
            (id for id, elem in elements.items() 
             if elem.element_type == "input" and 
             ("search" in elem.element_label.lower() if elem.element_label else False)),
            None
        )
        
        if search_input:
            # Execute search
            await browser.type(search_input, query)
            await browser.press("Enter")
            
            # Wait for and analyze results
            content = await browser.text()
            
            # Extract structured data
            return {
                "query": query,
                "results": content,
                "page_state": await browser._get_state_dict()
            }
    finally:
        await browser.close()
```

## Development Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install development dependencies:
```bash
pip install -e ".[dev]"
```
4. Install Playwright browsers:
```bash
donew-install-browsers
```

## Testing

Run the test suite:
```bash
pytest tests/
```

For more detailed testing options, including using local or remote httpbin, see the [Testing Documentation](docs/testing.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 