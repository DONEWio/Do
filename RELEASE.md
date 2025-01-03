# Release Process

When releasing a new version of DoNew, follow these steps:

1. Update version numbers:
   - Edit `pyproject.toml`: Update `version = "x.y.z"`
   - Edit `src/donew/__init__.py`: Update `__version__ = "x.y.z"`

2. Update CHANGELOG.md:
   ```markdown
   ## [x.y.z] - YYYY-MM-DD
   ### Added
   - New features
   
   ### Changed
   - Changes in existing functionality
   
   ### Fixed
   - Bug fixes
   ```

3. Build and test:
   ```bash
   pip install -e .
   pytest
   ```

4. Build distribution:
   ```bash
   python -m build
   ```

5. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

6. Create a git tag and push:
   ```bash
   git add .
   git commit -m "Release x.y.z"
   git tag -a vx.y.z -m "Version x.y.z"
   git push origin main --tags
   ```

Remember to replace `x.y.z` with the actual version number (e.g., "0.1.3"). 