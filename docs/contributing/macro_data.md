# Contributing to macro_data Package

This guide explains how to contribute to the `macro_data` package, which handles data preprocessing, harmonization, and initialization for the macroeconomic simulation framework.

## Package Overview

The `macro_data` package transforms raw economic data from various sources into standardized formats that the `macromodel` package can use for simulations. It follows a specific architecture pattern focused on data readers, processing modules, and synthetic country generation.

## macro_data Architecture Flow

The macro_data package follows a clear pattern:

```
Raw Data → Readers → DataReaders → Processing Modules → SyntheticCountry → DataWrapper
```

**Key Insight**: Readers provide high-level methods to access complex data (like `reader.get_quarterly_gdp(country, year, quarter)`), and processing modules use readers through the `from_readers` class method pattern to extract and harmonize data from multiple sources.

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
├── configuration/             # Configuration classes
└── data_wrapper.py            # Main orchestration class
```

## Adding New Data Sources

When adding a new data source (e.g., methane emissions data), follow these steps:

### 1. Create a Reader Class

All data sources must have a corresponding **reader** class in the `macro_data/readers/` directory. Readers provide high-level methods that abstract away data complexity.

**Directory structure for readers:**
```
macro_data/readers/
├── economic_data/          # Economic indicators (GDP, inflation, etc.)
├── emissions/             # Environmental data (CO2, methane, etc.)
├── population_data/       # Demographics and household data
├── io_tables/            # Input-output tables
├── socioeconomic_data/   # Social and economic statistics
└── criticality_data/     # Supply chain criticality data
```

### 2. Reader Class Structure

Follow these coding patterns for all reader classes:

#### Constructor Pattern
```python
class MyDataReader:
    """
    Reader for my data source.
    
    Args:
        data_param: Description of the data parameter
        
    Attributes:
        data_param: Stored data parameter
    """
    
    def __init__(self, data_param: pd.DataFrame):
        """
        Initialize reader with data.
        
        Args:
            data_param: The data to store
            
        Note:
            Only store attributes in __init__. No processing logic here.
        """
        self.data_param = data_param
    
    @classmethod
    def from_data(cls, data_path: Path, **kwargs) -> "MyDataReader":
        """
        Create reader instance from data files.
        
        Args:
            data_path: Path to data directory
            **kwargs: Additional parameters
            
        Returns:
            MyDataReader: New instance with loaded data
        """
        # Load and process data here
        data = pd.read_csv(data_path / "my_data.csv")
        return cls(data_param=data)
    
    @classmethod
    def from_config(cls, config: MyDataConfig) -> "MyDataReader":
        """
        Create reader from configuration object.
        
        Args:
            config: Configuration containing data parameters
            
        Returns:
            MyDataReader: New instance from configuration
        """
        # Process configuration and create instance
        return cls(data_param=config.data)
```

#### Key Patterns:
- **Minimal `__init__`**: Only store attributes, no processing logic
- **`@classmethod` constructors**: Use `from_data()`, `from_config()`, etc. for actual construction
- **Type hints**: All parameters and return values must have type hints
- **Docstrings**: Use Google-style docstrings for all public methods

### 3. Example: Adding Methane Emissions Data

Here's how you would add methane emissions data:

```python
# File: macro_data/readers/emissions/methane_reader.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from macro_data.configuration.countries import Country

@dataclass
class MethaneReader:
    """
    Reader for methane emissions data.
    
    Args:
        emissions_df: DataFrame containing methane emissions by country/year
        
    Attributes:
        emissions_df: Methane emissions data indexed by country and year
    """
    
    emissions_df: pd.DataFrame
    
    @classmethod
    def from_data(cls, data_path: Path) -> "MethaneReader":
        """
        Create reader from methane emissions data files.
        
        Args:
            data_path: Path to directory containing methane data files
            
        Returns:
            MethaneReader: New instance with loaded emissions data
        """
        # Load methane emissions data
        emissions_df = pd.read_csv(data_path / "methane_emissions.csv")
        
        # Process and clean data
        emissions_df["year"] = pd.to_datetime(emissions_df["year"], format="%Y")
        emissions_df = emissions_df.set_index(["country", "year"])
        
        return cls(emissions_df=emissions_df)
    
    def get_methane_emissions(self, country: Country, year: int) -> float:
        """
        Get methane emissions for a specific country and year.
        
        Args:
            country: Country to get emissions for
            year: Year to get emissions for
            
        Returns:
            float: Methane emissions in metric tons CO2 equivalent
        """
        try:
            return self.emissions_df.loc[(str(country), f"{year}")]["methane_emissions"]
        except KeyError:
            return 0.0  # Default value if no data available
    
    def get_sectoral_methane_emissions(self, country: Country, year: int, sector: str) -> float:
        """Get methane emissions for a specific sector."""
        try:
            return self.emissions_df.loc[(str(country), f"{year}", sector)]["methane_emissions"]
        except KeyError:
            return 0.0
```

### 4. Add to DataReaders Container

Add your reader to the `DataReaders` class in `macro_data/readers/default_readers.py`:

```python
# In macro_data/readers/default_readers.py
from macro_data.readers.emissions.methane_reader import MethaneReader

@dataclass
class DataReaders:
    """Container for all data readers."""
    
    # ... existing readers
    methane: Optional[MethaneReader] = None
    
    @classmethod
    def from_raw_data(cls, data_path: Path, ...) -> "DataReaders":
        """Create DataReaders from raw data files."""
        
        # ... existing reader initialization
        
        # Add methane reader
        methane_reader = None
        if (data_path / "methane").exists():
            methane_reader = MethaneReader.from_data(data_path / "methane")
        
        return cls(
            # ... existing readers
            methane=methane_reader,
        )
```

### 5. Use in Processing Modules

Processing modules access readers through the `from_readers` pattern:

```python
# In macro_data/processing/synthetic_firms/default_synthetic_firms.py

class DefaultSyntheticFirms:
    
    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        year: int,
        # ... other parameters
    ) -> "DefaultSyntheticFirms":
        """Create synthetic firms from reader data."""
        
        # Extract data from different readers
        gdp_data = readers.world_bank.get_historic_gdp(country_name, year)
        firm_deposits = readers.eurostat.get_total_nonfin_firm_deposits(country_name, year)
        
        # NEW: Use methane data if available
        methane_emissions = 0.0
        if readers.methane is not None:
            methane_emissions = readers.methane.get_methane_emissions(country_name, year)
        
        # Harmonize data from different sources
        firm_data = cls._harmonize_firm_data(gdp_data, firm_deposits, methane_emissions)
        
        return cls(firm_data=firm_data)
    
    @staticmethod
    def _harmonize_firm_data(gdp: float, deposits: float, methane: float) -> pd.DataFrame:
        """Harmonize data from different sources."""
        # Implementation that combines different data sources
        pass
```

### 6. Integration in SyntheticCountry

The `SyntheticCountry` class uses processing modules through `from_readers`:

```python
# In macro_data/processing/synthetic_country.py

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
        
        return cls(
            synthetic_firms=synthetic_firms,
            # ... other components
        )
```

## Key Architecture Patterns

### 1. Reader High-Level Methods
Readers provide high-level methods that abstract away data complexity:

```python
# Good: High-level methods
def get_quarterly_gdp(self, country: Country, year: int, quarter: int) -> float:
def get_methane_emissions(self, country: Country, year: int) -> float:
def get_total_nonfin_firm_deposits(self, country: Country, year: int) -> float:

# Bad: Exposing internal data structures
def get_data(self) -> pd.DataFrame:  # Too generic
def raw_emissions_data(self) -> pd.DataFrame:  # Exposes internals
```

### 2. from_readers Pattern
Processing modules consistently use `from_readers` to access data:

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
    emissions = readers.methane.get_methane_emissions(country, year)
    
    # Harmonize data from different sources
    harmonized_data = cls._harmonize_data(gdp, emissions)
    
    return cls(data=harmonized_data)
```

### 3. Data Harmonization
When combining data from different sources, create static methods:

```python
@staticmethod
def _harmonize_emissions_data(
    co2_emissions: float,
    methane_emissions: float,
    conversion_factor: float = 25.0
) -> float:
    """Harmonize different types of emissions data."""
    methane_co2_equiv = methane_emissions * conversion_factor
    return co2_emissions + methane_co2_equiv
```

## Configuration System

The `macro_data` package uses a comprehensive configuration system based on Pydantic BaseModel classes. These configurations provide type safety, validation, and flexible parameter management.

### Configuration Classes

Configuration classes are located in `macro_data/configuration/` and use Pydantic BaseModel:

```python
# Example: macro_data/configuration/data_configuration.py
from pydantic import BaseModel, Field
from typing import Optional, List
from macro_data.configuration.countries import Country

class DataConfiguration(BaseModel):
    """Configuration for data processing parameters."""
    
    countries: List[Country] = Field(
        default=[Country.USA, Country.GBR, Country.FRA],
        description="List of countries to process"
    )
    
    base_year: int = Field(
        default=2020,
        description="Base year for data processing"
    )
    
    include_methane_data: bool = Field(
        default=False,
        description="Whether to include methane emissions data"
    )
    
    scaling_factor: Optional[float] = Field(
        default=None,
        description="Optional scaling factor for economic data"
    )
```

### Benefits of Pydantic BaseModel

1. **Type Safety**: Your editor will provide autocompletion and type checking
2. **Validation**: Automatic validation of data types and constraints
3. **Default Values**: Easy to specify default values for parameters
4. **Documentation**: Field descriptions serve as built-in documentation
5. **Serialization**: Easy conversion to/from JSON and YAML

### Usage Patterns

#### Creating Configurations

```python
# Using defaults
config = DataConfiguration()

# Overriding specific parameters
config = DataConfiguration(
    countries=[Country.USA, Country.DEU],
    base_year=2019,
    include_methane_data=True
)

# From dictionary
config_dict = {
    "countries": ["USA", "DEU"],
    "base_year": 2019,
    "include_methane_data": True
}
config = DataConfiguration(**config_dict)
```

#### Reading from YAML

```python
import yaml
from pathlib import Path

# Read configuration from YAML file
with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
    
config = DataConfiguration(**config_dict)
```

#### Using in Processing

```python
def process_data(config: DataConfiguration) -> None:
    """Process data according to configuration."""
    
    # Your editor knows the types and provides autocompletion
    for country in config.countries:  # Type: List[Country]
        year = config.base_year      # Type: int
        
        if config.include_methane_data:  # Type: bool
            # Process methane data
            pass
```

### Adding New Configuration Parameters

When adding new functionality, extend the appropriate configuration class:

```python
class DataConfiguration(BaseModel):
    # ... existing fields
    
    # Add new parameter
    new_data_source: bool = Field(
        default=False,
        description="Whether to include new data source"
    )
    
    new_parameter: Optional[str] = Field(
        default=None,
        description="Optional new parameter"
    )
```

## Testing Your Reader

### Sample Test Data Requirements

**MANDATORY**: When adding a new data source, you MUST provide sample test data.

#### Adding Sample Test Data

1. **Create data directory** in `tests/test_macro_data/unit/sample_raw_data/[your_data_source]/`

2. **Include representative data** with the same format as the full dataset:
   ```
   sample_raw_data/
   └── my_new_data_source/
       ├── main_data.csv              # Primary data file
       ├── metadata.json             # Supporting metadata if needed
       └── [year_folders]/           # Year-specific data if applicable
   ```

3. **Follow the subset pattern**:
   - **Primary country**: Include France (FRA) data
   - **Additional countries**: At least one other country (CAN, GBR, or USA)
   - **Time period**: Focus on 2014 with relevant historical data
   - **Real values**: Use actual data values, not synthetic or dummy data

#### Example Sample Data Structure

```python
# Example: tests/test_macro_data/unit/sample_raw_data/methane/methane_emissions.csv
country,year,sector,methane_emissions
FRA,2014,agriculture,1500.5
FRA,2014,energy,800.2
CAN,2014,agriculture,2100.8
CAN,2014,energy,1200.3
```

#### Test Integration

Your reader tests must work with the sample data:

```python
# File: tests/test_macro_data/readers/test_methane_reader.py
import pytest
from macro_data.readers.emissions.methane_reader import MethaneReader

def test_methane_reader_with_sample_data(readers):
    """Test methane reader using sample data."""
    
    # Test specific values from sample data
    fra_agriculture = readers.methane.get_methane_emissions("FRA", 2014, "agriculture")
    assert fra_agriculture == pytest.approx(1500.5, abs=0.1)
    
    # Test data availability
    assert readers.methane.has_data_for_country("FRA")
    assert readers.methane.has_data_for_country("CAN")
    
    # Test total emissions
    total_fra = readers.methane.get_total_methane_emissions("FRA", 2014)
    assert total_fra == pytest.approx(2300.7, abs=0.1)  # sum of sectors
```

#### Sample Data Validation

**Required test patterns:**

1. **Value validation**: Test against specific known values from sample data
2. **Data availability**: Verify data exists for test countries
3. **Format consistency**: Ensure data structure matches expectations
4. **Error handling**: Test behavior with missing data

```python
def test_sample_data_coverage(readers):
    """Ensure sample data covers required test cases."""
    
    # Required countries
    required_countries = ["FRA"]  # Minimum requirement
    for country in required_countries:
        assert readers.my_reader.has_data_for_country(country)
    
    # Required years  
    assert readers.my_reader.has_data_for_year(2014)
    
    # Data consistency
    data = readers.my_reader.get_data_for_country("FRA", 2014)
    assert len(data) > 0
    assert not data.isna().all()
```

### Comprehensive Testing

Create comprehensive tests for your reader following these patterns. See the [Testing Guidelines](testing.md) for detailed information on testing requirements and the sample data structure.

## Common Architecture Pitfalls to Avoid

- **Don't put logic in `__init__`**: Use `@classmethod` constructors instead
- **Don't expose internal data structures**: Provide high-level methods instead
- **Don't forget the `from_readers` pattern**: All processing modules must implement this
- **Don't store readers in DataWrapper**: Readers are passed through the pipeline, not stored
- **Don't forget data harmonization**: Consider how new data relates to existing data

## Architecture Summary

The macro_data architecture follows this clear pattern:
1. **Readers** provide high-level methods to access raw data
2. **DataReaders** aggregates all readers into a single container
3. **Processing modules** use `from_readers` to extract and harmonize data
4. **SyntheticCountry** orchestrates agent creation through processing modules
5. **DataWrapper** manages the entire pipeline
6. **Configuration** classes provide type-safe parameter management

This architecture ensures consistency, maintainability, and proper separation of concerns across the codebase.