# Costmodels Package Development Guide

This document outlines the general structure of the `costmodels` package and the development process using `pip`.

## Package Structure

The `costmodels` package is organized as follows:

*   **`src/`**: Contains the main source code of the package.
*   **`tests/`**: Contains all the unit tests for the package.
*   **`examples/`**: Contains example usage of the package.
*   **`pyproject.toml`**: Defines project metadata, dependencies, and build system configurations.
*   **`.gitlab-ci.yml`**: Defines project CI configuration.

## Development Process with pip

### 1. Install Dependencies

To set up your development environment and install the package in editable mode along with its testing dependencies, navigate to the project's root directory and execute the following command:

```bash
pip install -e ".[test]"
```

This command installs the package itself, making your local changes immediately available. It also installs dependencies listed under `[project.dependencies]` and the testing-specific dependencies listed under `[project.optional-dependencies.test]` in the `pyproject.toml` file.

### 2. Running Tests

Tests are managed and executed using `pytest`. After successfully installing the dependencies, you can run the test suite from the project's root directory with:

```bash
pytest
```

Pytest will automatically discover and run all tests located within the `tests/` directory. The configuration for pytest, including test paths and other options, can be found in the `pyproject.toml` file under the `[tool.pytest.ini_options]` section.

## Pre-commit Hooks (Optional)

This project utilizes pre-commit hooks to ensure code quality and consistency before commits are made. These hooks, defined in the `.pre-commit-config.yaml` file, typically handle tasks like code formatting.

To set up pre-commit hooks:

1.  **Install the git hooks:**
    ```bash
    pre-commit install
    ```

Once installed, the hooks will run automatically before each commit. However it's preffered to run `pre-commit run --all-files` before commiting anything.

## New API Guidelines

Cost models implemented with `src/costmodels/api.py` must define a static
``_run`` function. ``_run`` receives all inputs as ``jnp.ndarray`` values and
**must** behave as a pure function, i.e. no side effects or mutation of global
state. This is required so that JAX can trace the function and compute
gradients correctly.

The public :meth:`run` method handles conversion of input values (including
``pint.Quantity`` objects) to JAX arrays before invoking ``_run``. The return
value of ``_run`` must be an instance of ``CostModelOutput`` so that it can
participate in JAX transformations.

At the time of writing all shipped models still follow the old interface. They
will be gradually ported to the new API. Check ``examples/icostmodel.py`` for a
reference implementation of the new design.
