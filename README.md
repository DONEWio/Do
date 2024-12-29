# Python Package Template

A template for creating Python packages.

## Project Structure
```
.
├── docs/               # Documentation Site
├── src/               # Source code
│   └── package_name/  # Main package directory
├── tests/             # Test files
├── .gitignore        # Git ignore file
├── pyproject.toml    # Project metadata and dependencies
├── LICENSE           # License file
└── README.md         # This file
```

## Installation

1. Install the package:
   ```bash
   pip install donew
   ```

2. Install Playwright browsers:
   ```bash
   donew-install-browsers
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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 