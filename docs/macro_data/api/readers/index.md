# Data Readers

The data reading system is centered around the `DefaultReaders` class, which provides a unified interface for accessing and coordinating all economic data sources. This class manages the initialization, configuration, and coordination of various specialized readers for different types of economic data.

## Main Component

### DefaultReaders

The central coordinator for all data reading operations. It provides:

- Unified interface for accessing all economic data
- Automatic reader initialization and configuration
- Data path management
- Reader coordination and data validation
- Exchange rate handling
- Industry aggregation
- Country-specific data processing

[View DefaultReaders documentation →](default_readers.md)

## Implementation Details

The following specialized readers are used internally by DefaultReaders:

### Core Economic Data

- ICIOReader: Inter-Country Input-Output tables
- EuroStatReader: EU economic statistics
- WorldBankReader: Global economic indicators
- OECDEconData: OECD economic data
- IMFReader: IMF economic data
- PolicyRatesReader: Central bank policy rates
- ExchangeRatesReader: Currency exchange rates

### Population and Financial Data

- HFCSReader: Household Finance and Consumption Survey
- CompustatFirmsReader: Firm-level financial data
- CompustatBanksReader: Bank-level financial data

### Environmental Data

- EmissionsReader: Environmental and emissions data

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
