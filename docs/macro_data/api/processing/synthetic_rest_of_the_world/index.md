# SyntheticRestOfTheWorld

The `SyntheticRestOfTheWorld` class is an abstract base class that manages data related to the Rest of the World (ROW) agent, which represents all countries not explicitly simulated in the model.

## Core Functionality

The class handles:
1. Trade Relationships:
   - Aggregating exports to non-simulated countries
   - Tracking imports from non-simulated countries
   - Managing international price levels
   - Handling exchange rate effects

2. Data Aggregation:
   - Combining economic data from non-simulated countries
   - Scaling trade flows appropriately
   - Preserving global trade balance
   - Maintaining consistent units

3. Growth Modeling:
   - Processing historical growth patterns
   - Estimating export/import trends
   - Handling structural changes
   - Projecting future relationships

4. Market Structure:
   - Determining number of trading agents
   - Allocating market shares
   - Setting initial conditions
   - Preserving key relationships

## Key Attributes

- `country_name`: Always "ROW" for this class
- `year`: Reference year for the data
- `row_data`: Main DataFrame containing ROW economic data
- `exports_model`: Model for export growth trends
- `imports_model`: Model for import growth trends
- `n_exporters_by_industry`: Number of exporting agents by industry
- `n_importers`: Number of importing agents

## Data Structure

The ROW data is organized in a DataFrame with columns:
- Exports: Value of exports to simulated countries
- Imports in USD: Value of imports from simulated countries in USD
- Imports in LCU: Value of imports in local currency units
- Price in USD: Price levels in USD
- Price in LCU: Price levels in local currency units

# Implementation

::: macro_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world
    options:
        members:
            - SyntheticRestOfTheWorld

# DefaultSyntheticRestOfTheWorld

`DefaultSyntheticRestOfTheWorld` is a concrete implementation of `SyntheticRestOfTheWorld` that provides standard preprocessing of ROW data using common data sources.

## Data Source Integration

The class integrates data from multiple sources:
- Trade flow data
- Exchange rate information
- Industry-level statistics
- Historical growth patterns

## Initial State Processing

The class processes:
- Trade data aggregation
- Currency conversion
- Market structure initialization
- Growth model fitting

## Parameter Estimation

Key parameters estimated include:
- Export growth models
- Import growth models
- Exporter counts by industry
- Importer numbers

## Factory Methods

The class provides a factory method `from_readers` that creates a `DefaultSyntheticRestOfTheWorld` instance by:
1. Reading and processing data from various sources
2. Converting currencies using exchange rates
3. Fitting growth models if configured
4. Initializing market structure
5. Setting up initial conditions

# Implementation

::: macro_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world
    options:
        members:
            - DefaultSyntheticRestOfTheWorld
            - from_readers 