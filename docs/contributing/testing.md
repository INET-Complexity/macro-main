# Testing Guidelines

This guide explains how to write and run tests for this project, and what is required for contributions to be accepted.

## Writing Tests

- Use `pytest` for all unit and integration tests.
- Place tests in files named `test_*.py` or `*_test.py`.
- Write tests for any new features or bugfixes you add.
- Strive for good test coverage and meaningful assertions.

### Example

A simple function and its test:

```python
# my_module.py
def sum(a: int, b: int) -> int:
    return a + b
```

```python
# test_my_module.py
from my_module import sum

def test_sum():
    assert sum(2, 3) == 5
    assert sum(-1, 1) == 0
```

## Running Tests

- From the root of the repository, run:

```bash
pytest
```

- This will automatically discover and run all tests.

## Continuous Integration (CI)

- All pull requests are automatically checked for code style and tests.
- If tests fail or style checks do not pass, the pull request will be rejected until the issues are fixed.

## Best Practices

- Validate data and check for edge cases in your tests.
- Use fixtures for setup if needed.
- Tests also serve as documentation for expected behavior.

---

For more on workflow and style, see the Development Guide and Code Style Guide.
