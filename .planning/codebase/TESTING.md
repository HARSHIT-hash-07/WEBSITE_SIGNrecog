# Testing

This document logs the known automated test structures present in the codebase.

## Framework & Structure
- Currently, **no specialized testing framework** (like Pytest, Jest, or Cypress) is implemented across the stack. There are no test scripts in `<package.json>`.
- The frontend operates without a formal component testing layer.
- The `backend/` and `/sign-idd-api/` do not host typical `/tests/` directories.

## Mocking & Coverage
- Mocking logic doesn't functionally exist yet. 
- Some pseudo-tests run inside isolated execution loops, such as `/backend/verify_api.py`, which appears designed to script a mock request to the `/translate` endpoint to ensure the core API responds correctly. 
- However, continuous test coverage reporting is currently unconfigured.
- `/sign-idd-api/test_videos/` functions as a static asset fixture folder to test static file serving, but lacks matching test logic.
