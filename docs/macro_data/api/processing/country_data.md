# CountryData

The CountryData module provides data structures and utilities for managing country-specific economic data, particularly focusing on tax-related information.

## TaxData

The `TaxData` class is a container for various tax rates and financial metrics for a specific country. It aggregates different types of tax rates and financial metrics that are relevant for economic modeling and analysis.

### Key Features

- Comprehensive tax rate collection
- Integration with multiple data sources
- Support for both direct and indirect taxes
- Social insurance contribution tracking
- Risk premium management

### Attributes

- `value_added_tax` (float): Value Added Tax (VAT) rate
- `export_tax` (float): Tax rate applied to exports
- `employer_social_insurance_tax` (float): Social insurance tax rate paid by employers
- `employee_social_insurance_tax` (float): Social insurance tax rate paid by employees
- `profit_tax` (float): Corporate profit tax rate
- `income_tax` (float): Personal income tax rate
- `capital_formation_tax` (float): Tax rate on capital formation
- `risk_premium` (float): Risk premium rate for firms

### Factory Methods

#### from_readers

```python
@classmethod
def from_readers(cls, readers: DataReaders, country: Country, year: int) -> TaxData
```

Creates a TaxData instance by fetching data from various data readers.

**Parameters:**

- `readers` (DataReaders): Collection of data readers for accessing different data sources
- `country` (Country): Country for which to fetch tax data
- `year` (int): Year for which to fetch tax data

**Returns:**

- `TaxData`: Instance containing tax rates and financial metrics for the specified country and year

### Data Sources

The class integrates data from multiple sources:

- World Bank: VAT rates and export taxes
- OECD: Social insurance and corporate tax rates
- Eurostat: Risk premiums and capital formation taxes

### Usage Example

```python
from macro_data import DataReaders
from macro_data.configuration.countries import Country
from macro_data.processing.country_data import TaxData

# Initialize data readers
readers = DataReaders.from_raw_data(...)

# Create tax data for France in 2023
france_tax_data = TaxData.from_readers(
    readers=readers,
    country=Country.FRANCE,
    year=2023
)

# Access tax rates
vat_rate = france_tax_data.value_added_tax
corporate_tax = france_tax_data.profit_tax
```

::: macro_data.processing.country_data
    options:
        members:
            - TaxData
