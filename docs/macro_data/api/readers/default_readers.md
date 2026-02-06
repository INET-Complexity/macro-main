# DefaultReaders

## Overview

The `DefaultReaders` class serves as the central coordinator for all data reading operations in the macro_data package. It manages the initialization, configuration, and coordination of various specialized readers for different types of economic data.

## Specialized Readers

The following specialized readers are used internally by DefaultReaders:

### Core Economic Data Readers

#### ICIOReader

Handles Inter-Country Input-Output tables, providing methods for:

- Reading and processing ICIO data
- Calculating trade flows and proportions
- Managing industry-level data
- Computing value added and consumption

[View ICIOReader documentation →](icio_reader.md)

#### EuroStatReader

Processes Eurostat data for EU countries, including:

- Economic indicators
- Industry statistics
- Social and demographic data
- Financial statistics

[View EuroStatReader documentation →](eurostat.md)

#### WorldBankReader

Manages World Bank data for global economic indicators:

- GDP and growth rates
- Population statistics
- Trade data
- Development indicators

[View WorldBankReader documentation →](world_bank.md)

#### OECDEconData

Handles OECD economic data:

- Economic indicators
- Industry statistics
- Policy data
- Social indicators

[View OECDEconData documentation →](oecd.md)

#### IMFReader

Processes IMF data:

- Economic forecasts
- Financial statistics
- Balance of payments
- Exchange rates

[View IMFReader documentation →](imf.md)

### Financial and Population Data Readers

#### ExchangeRatesReader

Handles currency exchange rate data:

- Bilateral exchange rates
- Currency conversions
- Rate time series

[View ExchangeRatesReader documentation →](exchange_rates.md)

#### PolicyRatesReader

Processes central bank policy rates:

- Interest rates
- Monetary policy data
- Rate time series

[View PolicyRatesReader documentation →](policy_rates.md)

#### HFCSReader

Manages Household Finance and Consumption Survey data:

- Household wealth
- Income distribution
- Financial assets
- Liabilities

[View HFCSReader documentation →](hfcs.md)

#### CompustatFirmsReader

Processes firm-level financial data:

- Balance sheets
- Income statements
- Financial ratios
- Industry classification

[View CompustatFirmsReader documentation →](compustat_firms.md)

#### CompustatBanksReader

Handles bank-level financial data:

- Bank balance sheets
- Financial statements
- Banking indicators
- Risk metrics

[View CompustatBanksReader documentation →](compustat_banks.md)

### Environmental Data Readers

#### EmissionsReader

Manages environmental data:

- Emission factors
- Energy consumption
- Environmental indicators
- Climate data

[View EmissionsReader documentation →](emissions.md)

## Usage Example

```python
from macro_data.readers.default_readers import DataReaders
from pathlib import Path

# Initialize the central data reader
readers = DataReaders.from_raw_data(
    raw_data_path=Path("path/to/raw/data"),
    country_names=["FRA", "DEU", "USA"],
    simulation_year=2018,
    scale_dict={"FRA": 1.0, "DEU": 1.0, "USA": 1.0},
    industries=["C10T12", "C13T15"],
    aggregate_industries=True
)

# Access data through the unified interface
france_gdp = readers.world_bank.get_current_scaled_gdp("FRA", 2018)
german_trade = readers.icio[2018].get_trade("DEU", "FRA")
us_emissions = readers.emissions.get_emissions_data("USA")
```

## Best Practices

1. **Data Source Management**
    - Keep raw data in a consistent location
    - Document data source versions
    - Maintain data update schedules

2. **Reader Configuration**
    - Use appropriate aggregation levels
    - Configure proxy relationships
    - Set up proper error handling

3. **Performance Optimization**
    - Cache processed data
    - Use efficient data structures
    - Implement parallel processing when possible

4. **Data Quality**
    - Validate input data
    - Handle missing values appropriately
    - Document data transformations

::: macro_data.readers.default_readers
    options:
        members:
            - DataReaders
            - DataPaths
            - from_raw_data
            - get_investment_fractions
            - get_exogenous_data
            - get_benefits_inflation_data
            - get_total_benefits_lcu
            - get_total_unemployment_benefits_lcu
            - get_govt_debt_lcu
            - get_export_taxes
            - get_national_accounts_growth
            - expand_weights_by_income
