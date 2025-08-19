# macromodel Package Documentation

The `macromodel` package implements the core economic modeling framework for Macrocosm. It defines the behaviors of economic agents, market mechanisms, policy implementations, and the simulation engine that drives the model.

## Overview

The package provides tools to:

- Simulate economic agent behaviors (firms, banks, households, governments, etc.)
- Model market mechanisms (credit, goods, housing, labor)
- Implement monetary and fiscal policy
- Run time-stepped simulations and process results
- Integrate exogenous data and forecasts

## API Reference

For detailed documentation of all classes, methods, and functions, see the [macromodel API Reference](api/index.md).

## Core Components

### Agents

The economic actors in the model, each with their own state, decision rules, and interactions. Each agent type has a dedicated module and a `func/` folder for specialized functions.

- Firms
- Banks
- Households
- Central Bank
- Central Government
- Government Entities
- Individuals
- Agent (base class and utilities)

### Markets

The main arenas for economic exchange and price formation. Each market has a dedicated module and a `func/` folder for market-specific logic.

- Credit Market
- Goods Market
- Housing Market
- Labour Market

### Country

Manages country-level aggregation, regional structure, and country-specific parameters.

### Economy

Coordinates the overall economic system, aggregates results, and manages macroeconomic indicators.

### Simulation

The simulation engine that advances the model through time, manages time series, and processes results.

## Package Structure

```text
macromodel/
├── agents/
│   ├── agent/
│   ├── banks/ (func/)
│   ├── central_bank/ (func/)
│   ├── central_government/ (func/)
│   ├── firms/ (func/)
│   ├── government_entities/ (func/)
│   ├── households/ (func/)
│   ├── individuals/ (func/)
├── country/
├── economy/ (func/)
├── exchange_rates/
├── exogenous/
├── forecaster/
├── markets/
│   ├── credit_market/ (func/)
│   ├── goods_market/ (func/)
│   ├── housing_market/ (func/)
│   ├── labour_market/ (func/)
├── rest_of_the_world/
├── simulation.py
├── timeseries.py
├── timestep.py
```

## Usage Example

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

## Best Practices

1. **Modularity**
    - Use agent and market modules for clear separation of logic
    - Place specialized functions in the appropriate `func/` subfolder

2. **Configuration**
    - Use country and economy modules for parameter management
    - Document all configuration options

3. **Performance**
    - Use efficient data structures and vectorized operations
    - Profile and optimize bottlenecks in agent and market logic

4. **Extensibility**
    - Add new agent types or market mechanisms by following the existing modular structure
    - Keep function modules focused and well-documented
