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

## Testing

### Local Testing with httpbin

The test suite uses httpbin for testing HTTP interactions. You can run tests against either the public httpbin.org instance or a local Docker container.

#### Using Local httpbin

1. Start the httpbin container (runs on port 8070):
```bash
docker-compose up -d httpbin
```

2. Run tests with local httpbin:
```bash
pytest tests/  # Uses http://localhost:8070 by default
```

#### Using Remote httpbin

There are several ways to specify the httpbin URL:

1. Using command line option:
```bash
pytest tests/ --httpbin-url=https://httpbin.org
```

2. Using environment variable:
```bash
HTTPBIN_URL=https://httpbin.org pytest tests/
```

3. Using a different port:
```bash
pytest tests/ --httpbin-url=http://localhost:9000
# or
HTTPBIN_URL=http://localhost:9000 pytest tests/
```

The precedence order is:
1. HTTPBIN_URL environment variable
2. --httpbin-url command line option
3. Default value (http://localhost:8070)

### Test Coverage

The test suite includes comprehensive tests for:
- HTTP methods (GET, POST, etc.)
- Cookie management
- Local storage
- Response headers
- Status codes
- Image processing
- Element annotation
- Browser state management 