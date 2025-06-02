# SyntheticBanks

The `SyntheticBanks` class is an abstract base class that defines the interface for preprocessing and organizing banking system data. It provides the core functionality for managing bank-level data and relationships.

## Core Functionality

The class handles:
- Bank balance sheet data organization
- Customer relationship mapping
- Initial state calculations
- Interest rate management
- Market share calculations

## Key Methods

- `initialise_deposits_and_loans`: Sets up initial deposits and loans for all banks
- `set_bank_equity`: Sets equity level for each bank
- `set_deposits_from_firms`: Sets initial deposits from firms
- `set_deposits_from_households`: Sets initial deposits from households
- `set_loans_to_firms`: Sets initial loans to firms
- `set_loans_to_households`: Sets initial loans to households
- `set_bank_deposits`: Sets total deposits for each bank
- `set_market_share`: Calculates and sets market share for each bank
- `set_liability`: Calculates and sets total liabilities for each bank

# Implementation

::: macro_data.processing.synthetic_banks.synthetic_banks
    options:
        members:
            - SyntheticBanks
            - initialise_deposits_and_loans
            - set_bank_equity
            - set_deposits_from_firms
            - set_deposits_from_households
            - set_loans_to_firms
            - set_loans_to_households
            - set_bank_deposits
            - set_market_share
            - set_liability

# DefaultSyntheticBanks

`DefaultSyntheticBanks` is a concrete implementation of `SyntheticBanks` that provides preprocessing of banking system data using standard data sources (OECD, Eurostat, Compustat). This class handles the initialization and organization of data that will be used to initialize behavioral models in the simulation package.

## Data Source Integration

The class integrates data from multiple sources:
- OECD economic indicators
- Eurostat banking statistics
- Compustat bank-level data
- National accounts data

## Initial State Processing

The class processes:
- Bank balance sheet data
- Historical rate parameters
- Customer relationship mappings
- Initial state calculations

## Parameter Estimation

Key parameters estimated include:
- Interest rate parameters
- Balance sheet ratios
- Market share calculations
- Risk premiums

## Factory Methods

The class provides two factory methods:
1. `from_readers`: Creates a `DefaultSyntheticBanks` instance using standard data sources
2. `from_readers_compustat`: Creates a `DefaultSyntheticBanks` instance using detailed Compustat data

# Implementation

::: macro_data.processing.synthetic_banks.default_synthetic_banks
    options:
        members:
            - DefaultSyntheticBanks
            - from_readers
            - from_readers_compustat 