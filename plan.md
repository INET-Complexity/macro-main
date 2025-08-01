# Project Coding and Documentation Plan

This document outlines the basic rules and best practices for contributing to this codebase. It is a living sketchpad and will be filled out in detail later.

## 1. Code Formatting and Style

- **Use `black` and `isort`** for code formatting and import sorting. The CI will check this automatically (see `.github/workflows/style_check.yml`).
- All code must be formatted with `black` and imports sorted with `isort` before committing.

## 2. Typing and Function Signatures

- **Type hints are required** for all function arguments and return values.
- Use Python's standard typing (e.g., `str`, `int`, `float`, `list[str]`, `Optional[Type]`, etc.).
- Use `@dataclass` for data containers where appropriate (see `DataWrapper`).

## 3. Docstrings and Documentation

- All public classes and functions must have clear, informative docstrings.
- Use Google or NumPy style docstrings for consistency.
- Example usage should be included in docstrings for main classes (see `DataWrapper`).

## 4. Simplicity and Readability

- Write simple, modular functions. Avoid long, complex methods.
- Use descriptive variable and function names.
- Avoid deep nesting and keep indentation levels reasonable.

## 5. Testing and Validation

- All new code should include tests where possible.
- Use `pytest` for unit testing.
- Validate data and check for edge cases.

### How to Use Tests

- **What is a test?**
  A test is a small piece of code that checks whether a function or class behaves as expected. For example:

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

- **How do you run tests?**
  - From the root of the repository, run:

    ```bash
    pytest
    ```

  - This will automatically discover and run all files named `test_*.py` or `*_test.py`.

- **Purpose of tests:**
  - Ensure your code works as intended and continues to work after future changes.
  - Catch bugs early and make refactoring safer.
  - Serve as documentation for expected behavior.

- **Continuous Integration (CI):**
  - We use GitHub/GitLab workflows to automatically check code style and run all tests on every pull request.
  - If tests fail or style checks do not pass, the pull request will be rejected until the issues are fixed.

- **Best Practice:**
  - Always write tests for any new features or bug fixes you add.
  - Strive for good test coverage and meaningful assertions.

## 6. Project Structure

- Keep code organized by domain (e.g., `macro_data`, `macromodel`).
- Place preprocessing and matching logic in `macro_data`, and core simulation logic in `macromodel`.

## 7. Documentation

- Keep documentation up to date with code changes.
- Use mike/mkdocs for API and user documentation.
- Reference code directly in docs using mkdocstrings (see `docs/index.md`).

## 8. Contribution Process

- Follow the development and style guides.
- Write clear commit messages.
- Open pull requests for all changes and request review.

## 9. Git Workflow

- **Never push directly to `main`.**
- When you want to add a feature or fix a bug:
  1. **Open an issue** describing the feature or bug.
  2. **Create a new branch** (or multiple branches for separate features/fixes) from `main` for your work.
  3. Make your changes and ensure all tests pass locally.
  4. Push your branch to the remote repository.
  5. **Open a pull request** (PR) to merge your branch into `main`.
  6. Request a code review and address any feedback.
- Only merge to `main` after review and all checks pass.
- For a hands-on introduction to Git branching and workflow, see [Learn Git Branching](https://learngitbranching.js.org/).

---

**This plan is a draft and will be expanded with concrete examples and more detailed rules.**
