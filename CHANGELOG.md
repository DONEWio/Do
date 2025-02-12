# Changelog

All notable changes to this project will be documented in this file.

## [0.1.6] - 2024-03-21

### Changed
- Experimental release with version bump

## [0.1.5] - 2024-01-03

### Fixed
- Fixed package distribution issues:
  - Corrected script file paths for installed package
  - Added proper wheel packaging configuration
  - Implemented dynamic script loading to work in both development and installed environments
- Restructured package assets to be properly included in distribution
- Added proper resource loading using importlib.resources

## [0.1.4] - 2025-01-03

### Fixed
- Fixed test discovery in VS Code by adding importlib mode
- Fixed package imports to work correctly in installed package

## [0.1.3] - 2025-01-03

### Other
- Version number is not read from pyproject.toml anymore.

## [0.1.2] - 2024-01-02

### Fixed
- Fixed element ID type from string to integer for better type safety
- Fixed element handling issues during website navigation
- Fixed Chromium bug handling for 404 status codes

### Other
- Added status badges to documentation
- Removed work-in-progress files

## [0.1.1] - 2023-12-29

- Initial release with core functionality 