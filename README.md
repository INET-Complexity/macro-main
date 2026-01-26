# Macromodel

## Introduction

This repository contains a comprehensive macroeconomic simulation framework consisting of two main packages:

- `macro-data`: Handles data preprocessing, harmonization, and initialization
- `macromodel`: Implements the core simulation engine and economic behaviors

The framework supports multi-country simulations with:

- Explicit modeling of individual countries
- Rest of World (ROW) aggregation for non-simulated countries
- International trade and exchange rate dynamics
- Detailed industry-level interactions
- Environmental impact tracking


The documentation can be found in [this URL.](http://macrodocs.macrocosm.group)

## Installation

### Requirements

- **Python Version**: 3.10, 3.11, or 3.12 (⚠️ **Python 3.13+ is NOT supported** due to numba compatibility)
- **System Dependencies**: HDF5 libraries for data storage
  - **macOS**: `brew install hdf5`
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install libhdf5-dev`
  - **Linux (Fedora/RHEL)**: `sudo dnf install hdf5-devel`
  - **Linux (Arch)**: `sudo pacman -S hdf5`

### Primary Installation Method: uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the fastest and most reliable way to install this project. It handles complex dependencies better than pip and provides reproducible installations.

#### Installing uv

`uv` is a fast and reliable way to install Python packages. It resolves dependencies and provides reproducible installations in a more reliable way than pip or conda. You can install it with the following command:

```bash
# Unix/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# NOTE can also be installed with homebrew:
brew install uv
# 

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# You can also install uv with winget:
winget install --id=astral-sh.uv  -e
```

#### Installing the Project with uv

```bash
# Clone the repository
git clone <repository-url>
cd macro-main

# Install with exact versions from uv.lock (recommended for reproducibility)
uv sync --python 3.12 --all-extras

# Or install specific extras
uv sync --python 3.12              # Core dependencies only
uv sync --python 3.12 --extra dev  # With development tools
uv sync --python 3.12 --extra docs # With documentation tools
```

#### Running Commands with uv

When using `uv`, you need to use the `uv run` command to run any commands in the env. This is because `uv` creates a virtual environment and sets the `PYTHONPATH` to the virtual environment. Run this in any child directory of the project (e.g. in any directory `./macro-main/...` or `./macro-main/macro_data/...` or `./macro-main/macromodel/...`, etc.).

```bash
# Run Python scripts
uv run python your_script.py

# Run tests
uv run pytest

# Run Jupyter
uv run jupyter lab

# Format code
uv run black .
uv run isort .
```

You can also use the `uv shell` command to enter the virtual environment, and you can add any packages to the environment with `uv pip install <package>`.

### Alternative Installation Methods

#### Method 1: Conda (Best for Windows and Scientific Computing)

```bash
# Create environment from provided file
conda env create -f environment.yml
conda activate macromodel

# Or create environment manually
conda create -n macromodel python=3.12
conda activate macromodel
conda install -c conda-forge numba numpy pandas scipy scikit-learn
pip install -e .
```

#### Method 2: pip with venv

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from requirements.txt (exact versions)
pip install -r requirements.txt

# Or install from pyproject.toml (flexible versions)
pip install -e ".[dev]"
```

#### Method 3: pip with requirements (Reproducible)

```bash
# For exact reproduction of tested versions
pip install -r requirements.txt        # Core dependencies
pip install -r requirements-dev.txt    # With development tools
```

### Installation Options

- `[dev]`: Development tools (pytest, black, isort, coverage)
- `[docs]`: Documentation tools (mkdocs, notebook support)

### Checking the integrity of the installation

To check that the model runs in your machine, make sure you have installed the dev dependencies (see above). Then, run the following command at the root of the project:

```bash
uv run pytest .
```

This will run a series of tests to check that the model runs in your machine. If you get no errors, then the model is installed correctly. Don't worry about warnings.

### Known Issues and Solutions

#### numba/llvmlite Issues

**Problem**: `KeyError: 'LLVMPY_AddSymbol'` or similar errors  
**Cause**: Version mismatch or incompatible binary

**Solutions**:

```bash
# Solution 1: Force compatible versions
pip uninstall numba llvmlite -y
pip install numba==0.59.0 llvmlite==0.42.0

# Solution 2: Use conda (more reliable)
conda install -c conda-forge numba=0.59.0

# Solution 3: Clear cache and reinstall
pip cache purge
pip install --no-cache-dir numba==0.59.0
```

**Platform-specific**:

- Apple Silicon: May need `arch -arm64` prefix
- Windows: Use conda to avoid compilation issues

#### tables/HDF5 Issues

**Problem**: Build failures, HDF5 not found, Cython errors  
**Cause**: Missing HDF5 system libraries

**Solutions**:

```bash
# Install HDF5 first (see System Dependencies above)

# macOS: Set HDF5 directory
export HDF5_DIR=$(brew --prefix hdf5)
pip install tables

# All platforms: Use conda (includes HDF5)
conda install pytables

# If Cython issues persist
pip install "cython<3.0"
pip install tables
```

#### Python 3.13 Issues

**Python 3.13 is not supported yet due to numba compatibility issues.**

How to solve this? Use Python 3.12 or lower.

```bash
# With uv
uv sync --python 3.12

# With pyenv
pyenv install 3.12.7
pyenv local 3.12.7

# With conda
conda create -n macromodel python=3.12
```

#### torch Installation Issues

In case you get an error message telling you that no wheels are available for the platform, you can install the CPU version of torch (typically a problem on Apple Silicon).

```bash
# Install CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon with MPS support
pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Reproducible Installation

For exact reproduction of tested versions:

1. **Best Option**: Use `uv sync` with the provided `uv.lock` file
2. **Alternative**: Use exact versions from `requirements.txt`
3. The `uv.lock` file is the source of truth for working versions

### Development Workflow

```bash
# Install with development dependencies
uv sync --extra dev

# Run tests
uv run pytest
uv run pytest --cov=macro_data --cov=macromodel

# Format code
uv run black --config="pyproject.toml" .
uv run isort --settings-path pyproject.toml .

# Build documentation, but mind that there is a github action that builds the documentation for each push to the docs branch, done in the macrocosm repository.
uv sync --extra docs
uv run mkdocs serve
```

### Quick Start Commands

```bash
# One-line installation (requires uv)
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync --python 3.12 --all-extras

# Verify installation
uv run python -c "import macro_data, macromodel; print('Installation successful!')"

# Run tests 
uv run pytest . 
```

### Troubleshooting Tips

1. **Always check Python version first**: `python --version` (must be 3.10-3.12)
2. **For HDF5 errors**: Install system HDF5 libraries (see System Dependencies above)
3. **For version conflicts**: Use `uv sync` or conda instead of pip
4. **For Apple Silicon**: Some packages may need Rosetta (`arch -x86_64` prefix)
5. **For Windows**: Conda is strongly recommended to avoid compilation issues

If you encounter issues not covered here, please check the issue tracker or file a new issue with:

- Your Python version
- Your operating system
- The full error message
- The installation method you tried

## Data Preprocessing

The `DataWrapper` class serves as the primary interface for data preprocessing, handling:

1. Country-level economic data
2. Rest of World aggregation
3. Exchange rates and trade relationships
4. Emissions data
5. Market structure initialization

Basic usage:

```python
from macro_data.configuration_utils import default_data_configuration
from macro_data import DataWrapper

# Configure data preprocessing
data_config = default_data_configuration(
    countries=["FRA", "CAN", "USA"],
    proxy_country_dict={"CAN": "FRA", "USA": "FRA"}  # Use France as proxy for non-EU countries
)

# Create DataWrapper instance
creator = DataWrapper.from_config(
    configuration=data_config,
    raw_data_path="path/to/raw/data",
    single_hfcs_survey=True  # Use single survey for household finance data
)

# Save processed data
creator.save("path/to/save/data.pkl")
```

### Data Configuration Options

The `data_config` object supports various settings:

- Country selection and proxy relationships
- Industry aggregation levels
- Market structure parameters
- Data scaling and filtering
- Environmental impact tracking

## Running Simulations

The simulation engine supports:

- Multi-country economic interactions
- Global goods market clearing
- Exchange rate dynamics
- Environmental impact assessment
- Detailed metric tracking

Basic simulation setup:

```python
from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation
from pathlib import Path

# Load preprocessed data
data = DataWrapper.init_from_pickle("./data.pkl")

# Configure country-specific parameters
country_configurations = {
    "FRA": CountryConfiguration(),
    "CAN": CountryConfiguration(),
    "USA": CountryConfiguration()
}

# Create simulation configuration
configuration = SimulationConfiguration(
    country_configurations=country_configurations,
    t_max=20  # Number of time steps
)

# Initialize simulation
model = Simulation.from_datawrapper(
    datawrapper=data,
    simulation_configuration=configuration
)

# Run simulation and save results
model.run()
model.save(save_dir=Path("./output/"), file_name="multi_country_simulation.h5")
```

### Simulation Configuration

The `SimulationConfiguration` class supports extensive customization:

- Country-specific behavioral parameters
- Market clearing mechanisms
- Exchange rate dynamics
- Growth model settings
- Environmental impact factors

### Output Analysis

Simulation results are saved in HDF5 format, containing:

- Country-level economic metrics
- Global market conditions
- Trade relationships
- Environmental impacts
- Time series data

For less verbose output, configure logging:

```python
import logging
logging.basicConfig(level=logging.WARNING)
```

## Advanced Features

The framework includes support for:

- Custom country implementations
- Alternative market clearing mechanisms
- Extended environmental impact modeling
- Detailed metric tracking and analysis
- Data validation and consistency checks

## Documentation

For detailed documentation of specific components:

- Data preprocessing modules
- Simulation engine
- Market mechanisms
- Configuration options
- Analysis tools

Please refer to the module docstrings and inline documentation.

This will save the run data into an `output` directory, writing into `./output/can_usa_fra_run.h5`. If you want to have less verbose logs, you can use a logger to change the logging level to `logging.INFO` or `logging.WARNING`.

This simulation runs the model using default country configurations. You can modify them directly from the `CountryConfiguration`  object.

## Calibration Package

The repository also includes a calibration package (`macrocalib`) that provides tools for calibrating the macromodel using simulation-based inference (SBI). This package is installed as an optional dependency.

### Installing the Calibration Package

To install the calibration package along with its dependencies, use:

```bash
pip install -e ".[calibration]"
```

**Note:** If you encounter issues installing the `sbi` package, you may need to pass the `--use-pep517` flag:

```bash
pip install -e ".[calibration]" --use-pep517
```

This flag helps resolve build issues with some of the dependencies, particularly `nflows`.

### Using the Calibration Package

The calibration package provides tools for sampling from the macromodel and running simulations in parallel. For more details, refer to the `macrocalib` module documentation.


## Contributing

Contributions are welcome! By submitting a pull request or patch, you agree to license your contribution under the Apache License 2.0, consistent with this project's license.

Please ensure any contributions follow the guidelines in [the documentation](http://macrodocs.macrocosm.group).

For significant changes, please open an issue first to discuss the proposed changes.  

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This software includes technology developed at the University of Oxford and licensed from Oxford University Innovation Limited.

Copyright © 2025 Oxford University Innovation Limited  
Copyright © 2025 Macrocosm Limited


