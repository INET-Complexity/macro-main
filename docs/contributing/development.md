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

## Repository Guidelines

### Keep the Repository Generic

This project is designed to be used by many teams and researchers. **The repository must remain generic and clean.**

#### What Should NOT be in This Repository

- **❌ Project-specific code**: Analysis scripts, visualizations, or post-processing tools for specific projects
- **❌ Specific simulation runs**: Python files or notebooks containing runs for specific research projects  
- **❌ Personal data files**: Any data files specific to individual projects or teams
- **❌ Team-specific utilities**: Code that only benefits one team or project
- **❌ Ad-hoc analysis scripts**: One-off scripts for data exploration or results processing

#### What IS Allowed

- **✅ Core framework code**: Generic agents, markets, data readers that benefit all users
- **✅ Example notebooks**: Educational examples in `examples/` folder showing how to use the framework
- **✅ Test data**: Sample data required for testing in `tests/test_macro_data/unit/sample_raw_data/`
- **✅ Documentation**: Contribution guides, API documentation, architectural explanations
- **✅ Generic utilities**: Tools that benefit all users of the framework

#### For Project-Specific Work

If you need to create project-specific content:

1. **Create your own repository** for project-specific analysis, visualizations, and post-processing. You can add these packages as dependencies in your own repository.
2. **Use this framework as a dependency** by importing the `macro_data` and `macromodel` packages
3. **Keep your specific configurations** in your own repository
4. **Share improvements back** to this repository only if they benefit all users

#### Jupyter Notebooks Policy

- **Only in `examples/` folder**: Notebooks must demonstrate framework usage, not specific research
- **Educational purpose**: Should teach users how to use the framework
- **Generic examples**: Use sample data and standard configurations
- **Well-documented**: Clear explanations of what each example demonstrates

#### Example Repository Structure for Your Project

```
your-project-repository/
├── analysis/                  # Your specific analysis scripts
├── visualizations/           # Your plotting and visualization code
├── data/                     # Your project-specific data
├── notebooks/               # Your research notebooks
├── configurations/          # Your custom model configurations
├── results/                 # Your simulation results
└── requirements.txt         # Including this framework as dependency
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
