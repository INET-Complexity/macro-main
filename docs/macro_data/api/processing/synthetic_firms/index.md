# SyntheticFirms

The `SyntheticFirms` class is an abstract base class that serves as a container for preprocessed firm-level data. It manages firm data including industry assignments, employment, wages, production, financial positions, and various economic indicators.

## Core Functionality

The class stores firm data in a pandas DataFrame with the following key columns:
- Industry: The industry classification of the firm
- Number of Employees: Employee count
- Total Wages: Total wage bill
- Production: Production output (in local currency)
- Price: Product price (in local currency)
- Labour Inputs: Labor input quantities
- Inventory: Current inventory levels
- Demand: Current demand
- Deposits: Bank deposits
- Debt: Outstanding debt
- Equity: Firm equity
- Employees ID: List of employee IDs
- Corresponding Bank ID: Associated bank ID
- Taxes paid on Production: Production taxes
- Interest paid: Total interest payments
- Profits: Current profits
- Unit Costs: Production costs per unit
- Corporate Taxes Paid: Corporate tax payments
- Debt Installments: Scheduled debt payments

## Key Attributes

- `country_name`: Country identifier for data collection
- `scale`: Scaling factor for synthetic data
- `year`: Reference year for preprocessing
- `industries`: List of industry classifications
- `number_of_firms_by_industry`: Array of firm counts by industry
- `firm_data`: Main DataFrame containing firm information
- `total_firm_deposits`: Aggregate firm deposits
- `total_firm_debt`: Aggregate firm debt
- `capital_inputs_productivity_matrix`: Capital productivity parameters
- `intermediate_inputs_productivity_matrix`: Input productivity parameters
- `capital_inputs_depreciation_matrix`: Capital depreciation rates
- `labour_productivity_by_industry`: Labor productivity by industry

## Main Methods

- `reset_function_parameters`: Updates firm operational parameters
- `set_additional_initial_conditions`: Sets up initial conditions for firm operations
- `total_emissions`: Calculates total emissions from inputs and capital
- `number_of_firms`: Property returning total number of firms

# Implementation

::: macro_data.processing.synthetic_firms.synthetic_firms
    options:
        members:
            - SyntheticFirms
            - reset_function_parameters
            - set_additional_initial_conditions
            - total_emissions
            - number_of_firms

# DefaultSyntheticFirms

`DefaultSyntheticFirms` is a concrete implementation of `SyntheticFirms` that provides preprocessing of firm-level data using standard data sources (OECD, Eurostat, Compustat). This class handles the initialization and organization of data that will be used to initialize behavioral models in the simulation package.

## Data Source Integration

The class integrates data from multiple sources:
- OECD economic indicators
- Eurostat business statistics
- Compustat firm-level data
- National accounts data

## Initial State Processing

The class processes:
- Industry-level aggregates
- Firm size distributions
- Financial positions
- Production parameters

## Parameter Estimation

Key parameters estimated include:
- Productivity metrics
- Input-output relationships
- Tax rates
- Interest rates

## Factory Methods

The class provides a factory method `from_readers` that creates a `DefaultSyntheticFirms` instance by:
1. Reading and processing data from various sources
2. Initializing firm-level data based on industry statistics
3. Setting up financial relationships and parameters
4. Calculating initial conditions for firm operations

## Additional Methods

- `set_taxes_paid_on_production`: Sets production taxes based on rates
- `set_interest_paid`: Calculates interest payments on deposits and loans
- `set_firm_profits`: Computes firm profits from sales and costs
- `set_unit_costs`: Calculates per-unit production costs
- `set_corporate_taxes_paid`: Computes corporate tax payments
- `set_firm_debt_installments`: Sets up debt payment schedules

# Implementation

::: macro_data.processing.synthetic_firms.default_synthetic_firms
    options:
        members:
            - DefaultSyntheticFirms
            - from_readers
            - reset_function_parameters
            - set_taxes_paid_on_production
            - set_interest_paid
            - set_firm_profits
            - set_unit_costs
            - set_corporate_taxes_paid
            - set_firm_debt_installments
            - set_additional_initial_conditions 