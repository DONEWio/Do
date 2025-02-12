# DoNew: Your Composable Task Library

[![PyPI version](https://badge.fury.io/py/donew.svg)](https://badge.fury.io/py/donew)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/donew)](https://pypi.org/project/donew/)
[![PyPI - License](https://img.shields.io/pypi/l/donew)](https://pypi.org/project/donew/)

## What is DoNew?

DoNew is a Python package designed to empower AI agents to interact effortlessly with web content and documents through high-level, intuitive interfaces. It bridges the gap between advanced AI capabilities and real-world applications by offering robust tools for web automation, document processing, and autonomous task execution. Developers can build isolated tasks that integrate seamlessly with their favorite tools, composing them either horizontally as peers or vertically as nested workflows to orchestrate complex operations, all while keeping each task self-contained and extendable.

## Quick Install

```bash
pip install donew
playwright install  # Install required browsers
```

## Key Features
- Task Planning & Multi-Step Operations: Organize complex workflows with clear planning and sequential execution.
- Provision Isolation: Context is isolated from the rest of the world. Each provision is a self-contained unit during runtime.
- Web Automation (DO.Browse): Leverage Playwright-based browser automation featuring smart element detection, visual debugging, and integrated knowledge graph extraction.
- Knowledge Processing: Utilize multi-model entity extraction (GLiNER), relationship mapping (GLiREL), advanced text analysis with spaCy, and robust graph storage via KuzuDB.
- Task Composition & Context Awareness: Achieve both horizontal (parallel provisions) and vertical (nested tasks) composition with comprehensive state and context management.
- Plug-and-Play Provisions: Comes with batteries included—ready-made tools such as MCP.run tasks and Restack workflows.

## Core Concepts

### Terminology
- **Doer**: A task executor that can be enacted. instance of DO.New
- **Provision**: A tool that can be used to enact a doer.
- **Realm**: The context in which a doer is enacted.
- **Envision**: The expected output of a doer.
- **Enact**: The act that triggers a doer to start its journey.

### Goals
0. Intuitive interface DO.New for task execution, DO.Browse for standalone web automation
1. Provide AI-first interfaces for web and document interaction
2. Enable autonomous decision-making and task execution
3. Maintain high reliability and performance
4. Ensure excellent developer experience
5. Support both simple and complex AI agent workflows
6. Provide batteries included provisions
7. Hide underlying complexity from the user

## Usage Guide

### 1. Creating a Task with DO.New

```python
from donew import DO

# Create a new task (doer) with a model
model = LitellmModel()
doer = DO.New(model, name='example_task', purpose='demonstrate basic usage')
```

### 2. Adding Context with realm()

```python
# Assume browser is a provision for web tasks
browser = DO.Browse(headless=False)

doer = doer.realm([browser])
```

### 3. Enforcing Output Structure with envision()

```python
from pydantic import BaseModel, Field

class Team(BaseModel):
    members: list[str] = Field(description='List of team members')

# This enforces that the output should match the Team schema
constrained_doer = doer.envision(Team)
```

### 4. Kickstarting the Workflow with enact()

```python
result = constrained_doer.enact('goto https://unrealists.com and find the team')
```

## Development Setup

### Requirements
- Python 3.11+ (required for Knowledge Graph functionality)
- uv package manager (recommended over pip)

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/DONEWio/donew.git
cd donew
```

2. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate virtual environment:
```bash
uv venv -p python3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
   - For basic usage:
   ```bash
   uv pip install pip
   uv pip install -e ".[dev]"
   ```

   - For Knowledge Graph functionality:
   ```bash
   uv pip install pip
   uv pip install -e "."
   uv pip install -e ".[kg,dev]"
   uv run -- spacy download en_core_web_md
   ```

5. Install Playwright browsers:
```bash
playwright install chromium
playwright install # or all browsers
```

## Testing

Run the test suite:
```bash
pytest tests/ --httpbin-url=https://httpbin.org
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Knowledge Graph Component

The Knowledge Graph component (`donew.see.graph`) provides entity and relationship extraction from text, with persistent storage in KuzuDB. This implementation is inspired by and adapted from the [GraphGeeks.org](https://live.zoho.com/PBOB6fvr6c) talk and [strwythura](https://raw.githubusercontent.com/DerwenAI/strwythura/refs/heads/main/demo.py).

## Features

- Named Entity Recognition using GLiNER
- Relationship Extraction using GLiREL 
- Graph storage and querying with KuzuDB
- Text processing and chunking with spaCy

## Graph Construction

The graph is built in layers:

1. **Base Layer**: Textual analysis using spaCy parse trees
2. **Entity Layer**: Named entities and noun chunks from GLiNER
3. **Relationship Layer**: Semantic relationships from GLiREL
4. **Storage Layer**: Persistent graph storage in KuzuDB

## Usage

```python
from donew.see.graph import KnowledgeGraph

# Initialize KG (in-memory or with persistent storage)
kg = KnowledgeGraph(db_path="path/to/db")  # or None for in-memory

# Analyze text
result = kg.analyze("""
OpenAI CEO Sam Altman has partnered with Microsoft.
The collaboration was announced in San Francisco.
""")

# Query the graph
ceo_relations = kg.query("""
MATCH (p:Entity)-[r:Relation]->(o:Entity)
WHERE p.label = 'Person' AND o.label = 'Company'
AND r.type = 'FOUNDER'
RETURN p.text as Founder, o.text as Company
ORDER BY Founder;
""") 