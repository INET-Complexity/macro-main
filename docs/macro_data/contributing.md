# Contributing to macro_data Package

This guide covers how to contribute to the `macro_data` package, which handles data preprocessing, harmonization, and initialization for the macroeconomic simulation framework.

## Package Overview

The `macro_data` package transforms raw economic data from various sources into standardized formats that the `macromodel` package can use for simulations. It follows a specific architecture pattern:

1. **Readers**: Interface with external data sources
2. **DataReaders**: Aggregate all readers into a single container
3. **Processing modules**: Transform reader data into synthetic agents/markets
4. **DataWrapper**: Orchestrates the entire pipeline

## Architecture Flow

```
Raw Data → Readers → DataReaders → Processing Modules → SyntheticCountry → DataWrapper
```

The key insight is that **readers provide high-level methods to access complex data** (like `reader.get_quarterly_gdp(country, year, quarter)`), and **processing modules use readers through the `from_readers` class method pattern** to extract and harmonize data from multiple sources.

## Directory Structure

```
macro_data/
├── readers/                     # Data source interfaces
│   ├── economic_data/          # Economic indicators
│   ├── emissions/              # Environmental data
│   ├── population_data/        # Demographics
│   └── default_readers.py      # DataReaders container
├── processing/                 # Data transformation modules
│   ├── synthetic_country.py   # Country-level aggregation
│   ├── synthetic_banks/       # Bank agent generation
│   ├── synthetic_firms/       # Firm agent generation
│   └── synthetic_*/           # Other synthetic components
└── data_wrapper.py            # Main orchestration class
```

## Adding New Data Sources

### Step 1: Create a Reader Class

Create your reader in the appropriate `readers/` subdirectory:

```python
# File: macro_data/readers/emissions/methane_reader.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from macro_data.configuration.countries import Country

@dataclass
class MethaneReader:
    """
    Reader for methane emissions data from EPA/UNFCCC sources.
    
    Args:
        emissions_data: DataFrame with methane emissions by country/year/sector
        
    Attributes:
        emissions_data: Processed emissions data
    """
    
    emissions_data: pd.DataFrame
    
    @classmethod
    def from_data(cls, data_path: Path) -> "MethaneReader":
        """
        Create reader from methane emissions data files.
        
        Args:
            data_path: Path to directory containing methane data files
            
        Returns:
            MethaneReader: New instance with loaded and processed data
        """
        emissions_df = pd.read_csv(data_path / "methane_emissions.csv")
        emissions_df = emissions_df.set_index(["country", "year", "sector"])
        
        return cls(emissions_data=emissions_df)
    
    def get_total_emissions(self, country: Country, year: int) -> float:
        """Get total methane emissions for a country and year."""
        try:
            country_data = self.emissions_data.loc[(str(country), year)]
            return country_data["methane_emissions"].sum()
        except KeyError:
            return 0.0
```

### Step 2: Add to DataReaders Container

The `DataReaders` class in `default_readers.py` acts as a container for all readers:

```python
# File: macro_data/readers/default_readers.py

from macro_data.readers.emissions.methane_reader import MethaneReader

@dataclass
class DataReaders:
    """Container for all data readers."""
    
    # ... existing readers
    methane: Optional[MethaneReader] = None
    
    @classmethod
    def from_config(cls, config: DataConfiguration, data_path: Path) -> "DataReaders":
        """Create DataReaders from configuration."""
        
        # ... existing reader initialization
        
        # Add methane reader if configured
        methane_reader = None
        if config.include_methane_data:
            methane_reader = MethaneReader.from_data(data_path / "methane")
        
        return cls(
            # ... existing readers
            methane=methane_reader,
        )
```

### Step 3: Use in Processing Modules

Processing modules use the `from_readers` pattern to access data:

```python
# File: macro_data/processing/synthetic_firms/default_synthetic_firms.py

class DefaultSyntheticFirms:
    
    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        year: int,
        # ... other parameters
    ) -> "DefaultSyntheticFirms":
        """
        Create synthetic firms from reader data.
        
        Args:
            readers: Container with all data readers
            country_name: Country to process
            year: Year to process
            
        Returns:
            DefaultSyntheticFirms: Synthetic firms for the country
        """
        
        # Extract data from readers
        gdp_data = readers.world_bank.get_historic_gdp(country_name, year)
        
        # NEW: Use methane data if available
        methane_emissions = 0.0
        if readers.methane is not None:
            methane_emissions = readers.methane.get_total_emissions(country_name, year)
        
        # Process and harmonize data
        firm_data = cls._process_firm_data(gdp_data, methane_emissions)
        
        return cls(firm_data=firm_data)
    
    @staticmethod
    def _process_firm_data(gdp_data: float, methane_emissions: float) -> pd.DataFrame:
        """Process and harmonize firm data."""
        # Implementation that combines different data sources
        # This is where you harmonize methane data with existing firm data
        pass
```

### Step 4: Integration in SyntheticCountry

The `SyntheticCountry` class uses processing modules through `from_readers`:

```python
# File: macro_data/processing/synthetic_country.py

@dataclass
class SyntheticCountry:
    
    @classmethod
    def eu_synthetic_country(
        cls,
        country: Country,
        year: int,
        readers: DataReaders,  # All readers passed here
        # ... other parameters
    ) -> "SyntheticCountry":
        """Create synthetic country from readers."""
        
        # Processing modules get readers and extract what they need
        synthetic_firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            # ... other parameters
        )
        
        # Other processing modules also use from_readers pattern
        synthetic_banks = DefaultSyntheticBanks.from_readers(
            readers=readers,
            country_name=country,
            year=year,
        )
        
        return cls(
            synthetic_firms=synthetic_firms,
            synthetic_banks=synthetic_banks,
            # ... other components
        )
```

## Key Architecture Patterns

### 1. Minimal `__init__` Pattern
```python
# Good: Only store attributes
def __init__(self, data: pd.DataFrame):
    self.data = data

# Bad: Processing in __init__
def __init__(self, data_path: Path):
    self.data = pd.read_csv(data_path)  # Processing belongs in @classmethod
```

### 2. `@classmethod` Constructor Pattern
```python
@classmethod
def from_data(cls, data_path: Path) -> "MyReader":
    """Create reader from data files."""
    # All data loading and processing happens here
    data = pd.read_csv(data_path)
    processed_data = cls._process_data(data)
    return cls(data=processed_data)
```

### 3. `from_readers` Pattern
```python
@classmethod
def from_readers(
    cls,
    readers: DataReaders,
    country: Country,
    year: int,
) -> "MyProcessor":
    """Create processor from readers."""
    # Extract needed data from readers
    gdp = readers.world_bank.get_historic_gdp(country, year)
    emissions = readers.emissions.get_emissions_factors(year)
    
    # Harmonize data from different sources
    harmonized_data = cls._harmonize_data(gdp, emissions)
    
    return cls(data=harmonized_data)
```

## Data Harmonization

When adding new data sources, you often need to harmonize them with existing data:

```python
@staticmethod
def _harmonize_emissions_data(
    co2_emissions: float,
    methane_emissions: float,
    conversion_factor: float = 25.0
) -> float:
    """
    Harmonize different types of emissions data.
    
    Args:
        co2_emissions: CO2 emissions in metric tons
        methane_emissions: Methane emissions in metric tons
        conversion_factor: CH4 to CO2 equivalent conversion factor
        
    Returns:
        float: Total CO2 equivalent emissions
    """
    methane_co2_equiv = methane_emissions * conversion_factor
    return co2_emissions + methane_co2_equiv
```

## Testing Your Integration

Create comprehensive tests for the integration:

```python
# File: tests/test_macro_data/test_integration/test_methane_integration.py

import pytest
from pathlib import Path
from macro_data.readers.default_readers import DataReaders
from macro_data.processing.synthetic_firms.default_synthetic_firms import DefaultSyntheticFirms
from macro_data.configuration.countries import Country

def test_methane_integration(tmp_path):
    """Test that methane data integrates properly with firm processing."""
    
    # Create test data
    methane_data = pd.DataFrame({
        "country": ["USA", "USA"],
        "year": [2020, 2020],
        "sector": ["agriculture", "energy"],
        "methane_emissions": [100.0, 50.0]
    })
    
    # Set up test files
    methane_dir = tmp_path / "methane"
    methane_dir.mkdir()
    methane_data.to_csv(methane_dir / "methane_emissions.csv", index=False)
    
    # Create data configuration
    config = DataConfiguration(include_methane_data=True)
    
    # Test reader creation
    readers = DataReaders.from_config(config, tmp_path)
    assert readers.methane is not None
    
    # Test processing module integration
    synthetic_firms = DefaultSyntheticFirms.from_readers(
        readers=readers,
        country_name=Country.USA,
        year=2020,
    )
    
    # Verify methane data was used
    assert synthetic_firms.total_emissions > 0
```

## Code Style Guidelines

Follow the same patterns as the existing codebase:

### Variable Naming
- Use `snake_case`: `methane_emissions`, `co2_equivalent`
- No `UPPERCASE`: Use `methane_factor` not `METHANE_FACTOR`
- Be descriptive: `total_methane_emissions` not `total_me`

### Type Hints
```python
def get_emissions(self, country: Country, year: int) -> float:
    """All parameters and return values need type hints."""
    return 0.0
```

### Documentation
Use Google-style docstrings with clear examples:

```python
def harmonize_emissions(
    self, 
    co2_data: float, 
    methane_data: float
) -> float:
    """
    Harmonize CO2 and methane emissions data.
    
    Args:
        co2_data: CO2 emissions in metric tons
        methane_data: Methane emissions in metric tons
        
    Returns:
        float: Total CO2 equivalent emissions
        
    Example:
        >>> reader = MethaneReader.from_data(data_path)
        >>> total = reader.harmonize_emissions(100.0, 50.0)
        >>> print(total)  # 1350.0 (100 + 50*25)
    """
    return co2_data + (methane_data * 25.0)
```

## Common Pitfalls to Avoid

1. **Storing readers in DataWrapper**: DataWrapper doesn't store readers as attributes
2. **Complex `__init__` methods**: Keep initialization simple, use `@classmethod` for processing
3. **Missing `from_readers` methods**: Processing modules must implement `from_readers`
4. **Poor data harmonization**: Always consider how new data relates to existing data
5. **Missing integration tests**: Test the full pipeline, not just individual components
6. **Inconsistent data formats**: Use standardized country codes and date formats

## Summary

The macro_data architecture follows a clear pattern:
1. **Readers** interface with raw data sources
2. **DataReaders** aggregates all readers
3. **Processing modules** use `from_readers` to extract and harmonize data
4. **DataWrapper** orchestrates the entire pipeline

When adding new data sources, follow this pattern and ensure your data harmonizes properly with existing sources. The key is understanding that readers are passed through the system, not stored as attributes.