# Code Style Guide

This guide summarizes the main code style rules for this project. Following these rules ensures code is readable, maintainable, and consistent.

## Formatting

- **Use `black` for code formatting** and **`isort` for import sorting**.
- `black` automatically formats your Python code to a consistent style. `isort` sorts your imports into standard sections and order.
- The CI will check formatting and imports automatically. Code that does not pass will be rejected.
- **Default settings:** Both tools use the configuration in `pyproject.toml` (see the CI workflow in `.github/workflows/style_check.yml`).

### How to Use Locally

Always run these commands before committing:

```bash
# Format code with black and sort imports with isort
black --config="pyproject.toml" .
isort . --settings-path pyproject.toml

# Or use the convenience script
./run_style.sh
```

### Check Formatting Without Making Changes

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
- Use Google-style docstrings for consistency.
- Include example usage in docstrings for main classes when possible.

### Example: Complete Function with Docstring

```python
def calculate_emissions(fuel_consumption: float, emission_factor: float) -> float:
    """
    Calculate CO2 emissions from fuel consumption.
    
    Args:
        fuel_consumption: Amount of fuel consumed in liters
        emission_factor: CO2 emission factor in kg CO2 per liter
        
    Returns:
        float: Total CO2 emissions in kg
        
    Raises:
        ValueError: If fuel_consumption or emission_factor is negative
    """
    if fuel_consumption < 0 or emission_factor < 0:
        raise ValueError("Fuel consumption and emission factor must be non-negative")
    
    return fuel_consumption * emission_factor
```

## Variable Naming

- **Use snake_case**: `my_variable`, `data_frame`, `emission_factor`
- **No UPPERCASE**: Avoid `GDP_DATA`, use `gdp_data` instead
- **Descriptive names**: `unemployment_rate` not `ur`

## Simplicity and Readability

- Write simple, modular functions. Avoid long, complex methods.
- Use descriptive variable and function names.
- Avoid deep nesting and keep indentation levels reasonable.

## Project Structure

- Organize code by domain (e.g., `macro_data`, `macromodel`).
- Place preprocessing and matching logic in `macro_data`, and core simulation logic in `macromodel`.

---

For more on workflow and testing, see the Development Guide and Testing Guidelines.
