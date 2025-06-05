# Code Style Guide

This guide summarizes the main code style rules for this project. Following these rules ensures code is readable, maintainable, and consistent.

## Formatting

- **Use `black` for code formatting** and **`isort` for import sorting**.
- `black` automatically formats your Python code to a consistent style. `isort` sorts your imports into standard sections and order.
- The CI will check formatting and imports automatically. Code that does not pass will be rejected.
- **Default settings:** Both tools use the configuration in `pyproject.toml` (see the CI workflow in `.github/workflows/style_check.yml`).
- **How to use locally:**

  ```bash
  # Format all code with black (using pyproject.toml)
  black .

  # Sort imports with isort (using pyproject.toml)
  isort .
  ```

- To check formatting without making changes:

  ```bash
  black --check --config="pyproject.toml" .
  isort . --check-only --settings-path pyproject.toml
  ```

## Typing

- All function arguments and return values must have type hints.
- Use standard Python typing (e.g., `str`, `int`, `float`, `list[str]`, `Optional[Type]`).
- Use `@dataclass` for data containers where appropriate.

### Example: Bad vs. Good Function

**Bad Example (no typing, no docstring):**

```python
def add(a, b):
    return a + b
```

**Good Example (with typing and docstring):**

```python
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b
```

## Docstrings

- All public classes and functions must have clear, informative docstrings.
- Use Google or NumPy style docstrings for consistency.
- Include example usage in docstrings for main classes when possible.

## Simplicity and Readability

- Write simple, modular functions. Avoid long, complex methods.
- Use descriptive variable and function names.
- Avoid deep nesting and keep indentation levels reasonable.

## Project Structure

- Organize code by domain (e.g., `macro_data`, `macromodel`).
- Place preprocessing and matching logic in `macro_data`, and core simulation logic in `macromodel`.

---

For more on workflow and testing, see the Development Guide and Testing Guidelines.
