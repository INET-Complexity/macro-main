# Installation Guide

This project requires **Python >=3.10**.

## System Dependencies

Before installing the Python package, you need to install some system dependencies:

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install libhdf5-dev
```

**macOS:**

```bash
brew install hdf5
```

## Python Package Installation

Clone the repository and install from the root directory:

```bash
# Basic installation (includes only core dependencies)
pip install .

# Development installation with all development tools
pip install -e ".[dev]"

# Documentation installation with all documentation tools
pip install -e ".[docs]"

# Install everything (core + dev + docs)
pip install -e ".[dev,docs]"
```

### Installation Options

- `[dev]`: Development tools (testing, formatting, etc.)
- `[docs]`: Documentation tools (mkdocs, mike, etc.)

For development, it is recommended to use editable mode:

```bash
pip install -e ".[dev]" --config-settings editable_mode=strict
```
