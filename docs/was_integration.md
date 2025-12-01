# WAS (Wealth and Assets Survey) Integration

This document describes the integration of ONS Wealth and Assets Survey (WAS) data with the macro_data package, enabling the use of UK-specific household wealth and asset data in macroeconomic modeling.

## Overview

The WAS integration provides:

- **WAS Data Reader**: Loads and processes WAS survey data from multiple waves
- **Synthetic WAS Population**: Generates synthetic populations from WAS data
- **UK-Specific Variables**: Handles WAS-specific wealth and asset variables
- **Seamless Integration**: Works with existing model components

## Key Components

### 1. WASReader

The `WASReader` class handles loading and processing WAS survey data:

```python
from macro_data.readers.population_data.was_reader import WASReader

# Initialize WAS reader
was_reader = WASReader.from_stata(
    country_name="United Kingdom",
    country_name_short="GB",
    year=2022,
    was_data_path=Path("path/to/was/data"),
    exchange_rates=exchange_rates,
    round_number=7
)
```

**Features:**
- Supports Stata (.dta) and CSV file formats
- Handles multiple survey waves (Rounds 1-8)
- Converts monetary values to local currency units
- Maps WAS variables to standardized format

### 2. SyntheticWASPopulation

The `SyntheticWASPopulation` class extends the base synthetic population to handle WAS-specific data:

```python
from macro_data.processing.synthetic_population.was_synthetic_population import SyntheticWASPopulation

# Create synthetic WAS population
was_population = SyntheticWASPopulation.from_readers(
    readers=data_readers,
    country_name=Country.UNITED_KINGDOM,
    country_name_short="GB",
    scale=1000,
    year=2022,
    quarter=1,
    industry_data=industry_data,
    industries=industries,
    total_unemployment_benefits=1000000,
    exogenous_data=exogenous_data
)
```

**Features:**
- WAS-specific wealth computation
- UK-specific income and saving patterns
- Enhanced housing and property data
- WAS-specific variable mappings

### 3. Automatic Integration

The system automatically uses WAS data when available for UK (GBR) countries:

```python
# The system automatically detects and uses WAS data for UK
synthetic_country = SyntheticCountry.eu_synthetic_country(
    country=Country.UNITED_KINGDOM,
    year=2022,
    quarter=1,
    country_configuration=country_config,
    industries=industries,
    readers=data_readers,  # Contains WAS data
    exogenous_country_data=exogenous_data,
    country_industry_data=industry_data,
    year_range=1,
    goods_criticality_matrix=criticality_matrix
)
```

## WAS-Specific Variables

The integration includes support for WAS-specific variables:

### Individual Variables
- `Employee Income`: Gross and net employment income
- `Self-Employment Income`: Self-employment earnings
- `Investment Income`: Income from financial assets
- `Pension Income`: Occupational and private pension income

### Household Variables
- `Value of Household Vehicles`: Vehicle assets
- `Value of Household Valuables`: Collectibles and valuables
- `Value of Self-Employment Businesses`: Business assets
- `Formal Financial Assets`: Mutual funds, bonds, shares
- `Voluntary Pension`: Individual pension wealth
- `Outstanding Balance of Credit Card Debt`: Credit card debt
- `Household Debt Burden`: Overall debt burden
- `Non-Mortgage Debt Burden`: Non-mortgage debt burden

### Housing Variables
- `Number of Second Homes`: Additional properties
- `Number of Buy to Let Properties`: Rental properties
- `Number of Buildings`: Commercial properties
- `Number of Land Pieces`: Land ownership
- `Number of Overseas Properties`: International properties

## Data Processing

### Variable Mapping

WAS variables are mapped to standardized names compatible with the existing model:

```python
# Example mappings
"hvaluer7": "Value of the Main Residence"
"DVHValueR7": "Value of the Main Residence Alt"
"DVOPrValR7": "Value of other Properties"
"AllgdR7": "Value of Household Valuables"
"TOTPENR7": "Voluntary Pension"
```

### Currency Conversion

All monetary values are converted to local currency units (GBP):

```python
# Automatic conversion from GBP to local currency
df.loc[:, monetary_columns] = exchange_rates.from_eur_to_lcu(
    country=country_name,
    year=year
) * df.loc[:, monetary_columns]
```

### Data Cleaning

The system handles:
- Missing value imputation
- Outlier detection and removal
- Data type conversion
- Variable standardization

## Configuration

### Data Paths

WAS data paths are configured in the `DataPaths` class:

```python
datapaths = DataPaths.default_paths(
    raw_data_path=Path("path/to/raw/data"),
    icio_years=[2020, 2021, 2022]
)

# WAS data path
was_path = datapaths.was_path  # raw_data_path / "was"
```

### Country Configuration

UK is already supported in the country configuration:

```python
from macro_data.configuration.countries import Country

# UK is available as
Country.UNITED_KINGDOM  # "GBR"
```

## Usage Examples

### Basic Usage

```python
from macro_data import DataConfiguration, Country
from macro_data.data_wrapper import DataWrapper

# Create configuration with UK
config = DataConfiguration(
    year=2022,
    quarter=1,
    country_configs={
        Country.UNITED_KINGDOM: CountryDataConfiguration(
            firms_configuration=FirmsDataConfiguration(),
            banks_configuration=BanksDataConfiguration(),
            central_bank_configuration=CentralBankDataConfiguration(),
            single_bank=True,
            single_firm_per_industry=True,
            single_government_entity=True,
            scale=1000,
        )
    }
)

# Initialize data wrapper (automatically uses WAS data for UK)
data_wrapper = DataWrapper.from_config(
    configuration=config,
    raw_data_path="path/to/raw/data"
)
```

### Advanced Usage

```python
# Access WAS-specific data
uk_country = data_wrapper.synthetic_countries["GBR"]
was_population = uk_country.population

# Access WAS-specific variables
vehicle_wealth = was_population.household_data["Value of Household Vehicles"]
pension_wealth = was_population.household_data["Voluntary Pension"]
credit_card_debt = was_population.household_data["Outstanding Balance of Credit Card Debt"]

# Compute WAS-specific statistics
avg_vehicle_wealth = vehicle_wealth.mean()
total_pension_wealth = pension_wealth.sum()
debt_burden = credit_card_debt.sum() / was_population.household_data["Income"].sum()
```

## Testing

The integration includes comprehensive tests:

```bash
# Run WAS integration tests
pytest tests/test_was_integration.py -v
```

Test coverage includes:
- WAS reader initialization
- Synthetic population creation
- Wealth and income computation
- Variable mapping and conversion
- Data processing and cleaning

## File Structure

```
macrocb7/
├── macro_data/
│   ├── readers/
│   │   └── population_data/
│   │       └── was_reader.py              # WAS data reader
│   └── processing/
│       └── synthetic_population/
│           ├── was_synthetic_population.py # WAS synthetic population
│           ├── was_household_tools.py      # WAS household processing
│           └── was_individual_tools.py     # WAS individual processing
├── tests/
│   └── test_was_integration.py            # WAS integration tests
├── examples/
│   └── was_integration_example.py         # Usage example
└── docs/
    └── was_integration.md                 # This documentation
```

## Data Requirements

### WAS Data Files

The system expects WAS data files in the following format:

```
was/
├── was_round_7_person_eul_*.dta          # Individual data
├── was_round_7_hhold_eul_*.dta           # Household data
└── ...
```

### Required Variables

The system requires the following WAS variables to be present:

**Individual Variables:**
- `pidno`: Personal identifier
- `hholdr7`: Household identifier
- `sexr7`: Gender
- `dvager7`: Age
- `edlevelr7`: Education level
- `wrkingr7`: Working status
- `dvgrspayannualr7`: Employee income

**Household Variables:**
- `hvaluer7`: Main residence value
- `DVOPrValR7`: Other property values
- `DVSaValR7_SUM`: Savings
- `DVFFAssetsR7_SUM`: Financial assets
- `TotmortR7`: Mortgage debt
- `TOTCSCR7_aggr`: Credit card debt

## Troubleshooting

### Common Issues

1. **Missing WAS Data Files**
   ```
   FileNotFoundError: No person files found matching pattern
   ```
   **Solution**: Ensure WAS data files are in the correct directory and format.

2. **Missing Variables**
   ```
   KeyError: 'Variable not found in WAS data'
   ```
   **Solution**: Check that required WAS variables are present in the data files.

3. **Currency Conversion Issues**
   ```
   ValueError: Exchange rate not available
   ```
   **Solution**: Ensure exchange rate data is available for the specified year.

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your WAS integration code here
```

## Future Enhancements

Planned improvements include:

1. **Additional WAS Rounds**: Support for more WAS survey waves
2. **Enhanced Variable Mapping**: More comprehensive variable mappings
3. **Advanced Imputation**: More sophisticated missing value imputation
4. **Regional Disaggregation**: Support for UK regional data
5. **Time Series Analysis**: Longitudinal WAS data analysis

## Contributing

To contribute to the WAS integration:

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure backward compatibility

## References

- [ONS Wealth and Assets Survey](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/methodologies/wealthandassetssurvey)
- [WAS Data Dictionary](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/methodologies/wealthandassetssurvey)
- [macro_data Package Documentation](../README.md)

