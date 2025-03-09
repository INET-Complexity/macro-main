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

## Installation

Requires Python >=3.10. Clone the repository and install from the root directory:

```bash
pip install .
```

For development installation:

```bash
pip install -e ./ --config-settings editable_mode=strict
```

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
