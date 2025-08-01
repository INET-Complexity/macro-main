# Basic Usage Examples

This guide walks through a typical workflow using the macromodel framework, from data preprocessing to running a simulation and extracting summary results.

## 1. Data Preprocessing with DataWrapper

First, preprocess your data using the `DataWrapper` class. This step harmonizes and prepares all the data needed for the simulation.

```python
from macro_data.configuration_utils import default_data_configuration
from macro_data import DataWrapper

# Configure data preprocessing
config = default_data_configuration(countries=["FRA"], proxy_country_dict={})

# Create DataWrapper instance
creator = DataWrapper.from_config(
    configuration=config,
    raw_data_path="path/to/raw/data",
    single_hfcs_survey=True
)

# Save processed data for later use
creator.save("./data_fra.pkl")
```

## 2. Running a Simulation

Once you have preprocessed data, you can run a simulation. Here's a step-by-step example:

```python
from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

# Load preprocessed data
preprocessed_data = DataWrapper.init_from_pickle("./data_fra.pkl")

# Set up simulation configuration
country_config = CountryConfiguration()
sim_config = SimulationConfiguration(country_configurations={"FRA": country_config})

# Initialize simulation
simulation = Simulation.from_datawrapper(
    datawrapper=preprocessed_data,
    simulation_configuration=sim_config
)

# Run the simulation (runs for the configured number of steps)
simulation.run()
```

## 3. Extracting and Interpreting Results

After running the simulation, you can extract summary results for each country using the `shallow_df_dict()` method:

```python
# Get a dictionary of shallow (summary) DataFrames for each country
shallow_outputs = simulation.shallow_df_dict()

# For France, get the summary DataFrame
shallow_output_df = shallow_outputs["FRA"]

# For example, get the total nominal production of all firms in France:
total_production = shallow_output_df["Production"]
print("Total nominal production (all firms):", total_production)
```

The `shallow_output_df` contains key economic indicators for the country, such as:

- `Production`: Total nominal production of all firms
- `Sales`, `Wages`, `Profits`, `Taxes Paid on Production`, etc.
- `Unemployment Rate`, `Imports`, `Exports`, and more

You can use these outputs for further analysis, validation, or visualization.

---

**Tip:**

- The `shallow_output` method (see API docs) provides a concise summary of the main economic metrics for each country, including GDP, inflation, unemployment, and sectoral outputs.
