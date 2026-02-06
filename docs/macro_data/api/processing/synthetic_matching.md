# Synthetic Matching

The Synthetic Matching modules handle the harmonization of data between different economic agents to create a consistent initial state for the model. These modules are not markets themselves, but rather data processing components that ensure consistency between different data sources.

## Core Functionality

The matching modules handle:

1. Individual-Firm Matching:
    - Harmonizing employment data
    - Reconciling wages with labor expenses
    - Matching workers to positions
    - Validating industry relationships

2. Firm-Bank Matching:
    - Reconciling financial data
    - Matching firms to banking relationships
    - Harmonizing deposit and loan balances
    - Validating balance sheet consistency

3. Household-Bank Matching:
    - Harmonizing retail banking data
    - Matching households to accounts
    - Reconciling deposit and loan balances
    - Validating customer relationships

4. Household-Housing Matching:
    - Reconciling property ownership
    - Matching tenants to properties
    - Harmonizing rental relationships
    - Processing social housing allocation

## Key Components

### Individual-Firm Matching

- `matching_individuals_with_firms.py`: Handles employment data harmonization
  - Validates employment counts
  - Reconciles wage totals
  - Computes position assignments
  - Adjusts for tax effects

### Firm-Bank Matching

- `matching_firms_with_banks.py`: Manages corporate banking relationships
  - Reconciles financial data
  - Allocates accounts
  - Validates balance sheets
  - Records relationships

### Household-Bank Matching

- `matching_households_with_banks.py`: Processes retail banking relationships
  - Harmonizes deposit data
  - Reconciles loan balances
  - Validates customer accounts
  - Records assignments

### Household-Housing Matching

- `matching_households_with_houses.py`: Handles property relationships
  - Processes ownership data
  - Manages rental market
  - Handles social housing
  - Validates tenure status

## Implementation

::: macro_data.processing.synthetic_matching.matching_individuals_with_firms
    options:
        members:
            - match_individuals_with_firms_country

::: macro_data.processing.synthetic_matching.matching_firms_with_banks
    options:
        members:
            - match_firms_with_banks_optimal

::: macro_data.processing.synthetic_matching.matching_households_with_banks
    options:
        members:
            - match_households_with_banks_optimal

::: macro_data.processing.synthetic_matching.matching_households_with_houses
    options:
        members:
            - set_housing_df
