# Development Guide

This guide outlines the main workflow and best practices for contributing to this project.

## Development Environment Setup

### 1. Install Development Dependencies

Install the project with development dependencies to get all necessary tools:

```bash
# Development installation with all tools
pip install -e ".[dev]" --config-settings editable_mode=strict

# This includes formatting tools (black, isort), testing tools (pytest), 
# and other development utilities
```

### 2. Editor Configuration

Use an editor with built-in linting capabilities for the best development experience:

- **VS Code**: Install the Python extension for automatic linting, formatting, and type checking
- **PyCharm**: Has built-in Python support with linting and type checking
- **Vim/Neovim**: Use plugins like ALE or CoC for Python linting
- **Sublime Text**: Install SublimeLinter-flake8 and other Python packages

Your editor should be configured to:
- Show type hints and errors in real-time
- Auto-format with `black` on save
- Sort imports with `isort`
- Highlight linting issues

### 3. Verify Installation

Test your setup:

```bash
# Check that development tools are available
black --version
isort --version
pytest --version

# Run style checks
black --config="pyproject.toml" .
isort . --settings-path pyproject.toml

# Run tests
pytest
```

## Workflow Overview

1. **Open an Issue**
    - Before starting work on a new feature or bugfix, open an issue describing your planned change.

2. **Branching**
    - **Never push directly to `main`.**
    - Create a new branch for each feature or fix, based on `main`.
    - Use descriptive branch names (e.g., `feature/add-forecasting`, `bugfix/fix-import-error`).

3. **Development**
    - Make your changes in your feature branch.
    - Ensure your code follows the style and contribution guidelines (see Code Style Guide).
    - Write tests for any new features or bugfixes.
    - Run all tests locally before pushing.

4. **Push and Pull Request**
    - Push your branch to the remote repository.
    - Open a pull request (PR) to merge your branch into `main`.
    - Link the PR to the relevant issue.
    - Request a code review and address any feedback.

5. **Continuous Integration (CI)**
    - All PRs are automatically checked for code style and tests.
    - PRs will be rejected if tests or style checks fail.

6. **Merge**
    - Only merge to `main` after review and all checks pass.

## Learning Git

If you are new to Git branching and workflow, try [Learn Git Branching](https://learngitbranching.js.org/) for an interactive introduction.

---

For more details on code style, testing, and documentation, see the other guides in this section.
