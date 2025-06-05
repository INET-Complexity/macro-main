# SyntheticCentralGovernment

The `SyntheticCentralGovernment` class is an abstract base class that serves as a container for preprocessed central government data. It manages government data including revenue streams, social benefits, tax collections, and financial positions.

## Core Functionality

The class stores government data in a pandas DataFrame with the following key columns:

- Total Unemployment Benefits: Initial unemployment benefit levels
- Other Social Benefits: Initial levels of other social transfers
- Debt: Initial central government debt level
- Bank Equity Injection: Initial bank support levels
- Total Social Housing Rent: Initial social housing revenue
- Taxes on Production: Initial production tax revenue
- VAT: Initial value-added tax revenue
- Capital Formation Taxes: Initial capital tax revenue
- Export Taxes: Initial export tax revenue
- Corporate Taxes: Initial corporate tax revenue
- Employer SI Tax: Initial employer social insurance revenue
- Employee SI Tax: Initial employee social insurance revenue
- Income Taxes: Initial income tax revenue
- Rental Income Taxes: Initial rental tax revenue
- Revenue: Total initial revenue
- Taxes on Products: Initial product tax revenue

## Key Attributes

- `country_name`: Country identifier for data collection
- `year`: Reference year for preprocessing
- `central_gov_data`: Main DataFrame containing government information
- `other_benefits_model`: Model for estimating other benefits
- `unemployment_benefits_model`: Model for estimating unemployment benefits

## Main Methods

- `update_fields`: Updates preprocessed government data based on other economic agents
- `set_revenue`: Sets initial revenue values in the preprocessed government data

# Implementation

::: macro_data.processing.synthetic_central_government.synthetic_central_government
    options:
        members:
            - SyntheticCentralGovernment
            - update_fields
            - set_revenue

# DefaultSyntheticCentralGovernment

`DefaultSyntheticCentralGovernment` is a concrete implementation of `SyntheticCentralGovernment` that provides preprocessing of central government data using standard data sources. This class handles the initialization and organization of data that will be used to initialize behavioral models in the simulation package.

## Data Source Integration

The class integrates data from multiple sources:

- Historical benefits data
- Tax revenue data
- Government debt data
- Bank support data

## Initial State Processing

The class processes:

- Benefits data organization
- Tax revenue parameter estimation
- Initial state calculations
- Model parameter estimation

## Parameter Estimation

Key parameters estimated include:

- Benefits models (unemployment and other benefits)
- Tax revenue parameters
- Initial debt levels
- Bank support levels

## Factory Methods

The class provides a factory method `from_readers` that creates a `DefaultSyntheticCentralGovernment` instance by:

1. Reading and processing data from various sources
2. Estimating benefits models using historical data
3. Setting up initial state data
4. Calculating initial parameters

## Additional Methods

- `build_unemployment_model`: Estimates a model for preprocessing unemployment benefits data
- `build_other_benefits_model`: Estimates a model for preprocessing other benefits data

# Implementation

::: macro_data.processing.synthetic_central_government.default_synthetic_central_government
    options:
        members:
            - DefaultSyntheticCGovernment
            - from_readers
            - build_unemployment_model
            - build_other_benefits_model
