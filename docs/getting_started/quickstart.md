# Quick Start Tutorial

This guide will help you get started with the **full pipeline**: data preprocessing and running a simulation using the macromodel framework.

## Pipeline Overview

The workflow consists of two main steps:

1. **Data Preprocessing (macro_data):**
    - The `macro_data` package preprocesses, harmonizes, and validates all economic data needed for the simulation.
    - You must run this step first and save the resulting data (or pass the object directly) before running any simulation.
2. **Simulation (macromodel):**
    - The `macromodel` package runs the simulation using the preprocessed data.
    - The simulation cannot run without this preprocessed data.

## Configuration Objects

Both data preprocessing and simulation use configuration objects to control parameters and structure:

- These are Python classes (using Pydantic for validation) with many preset parameters (defaults are not "tuned").
- You can instantiate them directly in Python, or load them from YAML files for reproducibility and sharing.
- Example (Python):

```python
from macro_data.configuration import DataConfiguration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration

# Instantiate with defaults
country_config = CountryConfiguration()
sim_config = SimulationConfiguration(country_configurations={"FRA": country_config})
```

- Example (YAML):

```yaml
# configuration.yaml
country_configurations:
  FRA:
    # ... parameters ...
```

You can load YAML configs using Pydantic or custom loaders.

---

## Data Preprocessing

The `DataWrapper` class is the main interface for data preprocessing. It handles country-level economic data, rest of world aggregation, exchange rates, emissions, and market structure initialization.

### Example: Creating and Saving Preprocessed Data

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

## Running a Simulation

The simulation engine supports multi-country interactions, market clearing, exchange rate dynamics, and environmental impact assessment.

### Example: Running a Basic Simulation

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

## Tips

- For less verbose output, configure logging:

```python
import logging
logging.basicConfig(level=logging.WARNING)
```

- You can customize country configurations and simulation parameters as needed.
