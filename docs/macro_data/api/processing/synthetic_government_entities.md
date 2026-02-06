# SyntheticGovernmentEntities

The `SyntheticGovernmentEntities` class is an abstract base class that serves as a container for preprocessed government entity data. It manages data about government entities that consume and invest in goods and services produced in the economy.

## Core Functionality

The class organizes data about:

1. Government Consumption:
    - Sectoral consumption patterns
    - Consumption in local currency and USD
    - Historical consumption growth trends
    - Consumption model parameter estimation

2. Entity Structure:
    - Number of government entities
    - Entity size distribution
    - Consumption allocation across entities
    - Entity-industry relationships

3. Environmental Impact:
    - Emissions from government consumption
    - Fuel-specific emission tracking

## Key Attributes

- `country_name`: Country identifier for data collection
- `year`: Base year for data preprocessing
- `number_of_entities`: Number of government entities
- `gov_entity_data`: Main DataFrame containing entity information
- `government_consumption_model`: Model for projecting consumption growth

## Main Methods

- `total_emissions`: Property that calculates total CO2 emissions from government consumption activities

# Implementation

::: macro_data.processing.synthetic_government_entities.synthetic_government_entities
    options:
        members:
            - SyntheticGovernmentEntities
            - total_emissions

# DefaultSyntheticGovernmentEntities

`DefaultSyntheticGovernmentEntities` is a concrete implementation of `SyntheticGovernmentEntities` that provides preprocessing of government entity data using standard data sources. This class handles the initialization and organization of data that will be used to initialize behavioral models in the simulation package.

## Data Source Integration

The class integrates data from multiple sources:

- OECD business demography statistics
- National accounts government consumption data
- Industry-level consumption patterns
- Environmental impact factors

## Initial State Processing

The class processes:

- Entity count calculation based on economic size
- Consumption allocation by industry
- Size-based distribution
- Growth rate calculation

## Parameter Estimation

Key parameters estimated include:

- Government consumption growth model
- Entity size distribution
- Industry-specific consumption patterns
- Environmental impact factors

## Factory Methods

The class provides a factory method `from_readers` that creates a `DefaultSyntheticGovernmentEntities` instance by:

1. Reading and processing data from various sources
2. Calculating entity counts and distributions
3. Setting up initial state data
4. Estimating growth models if requested
5. Processing environmental impact data if provided

# Implementation

::: macro_data.processing.synthetic_government_entities.default_synthetic_government_entities
    options:
        members:
            - DefaultSyntheticGovernmentEntities
            - from_readers
