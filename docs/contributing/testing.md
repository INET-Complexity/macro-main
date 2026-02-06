# Testing Guidelines

This guide explains how to write and run tests for this project, and what is required for contributions to be accepted.

## Writing Tests

- Use `pytest` for all unit and integration tests.
- Place tests in files named `test_*.py` or `*_test.py`.
- Write tests for any new features or bugfixes you add.
- Strive for good test coverage and meaningful assertions.

### Example

A simple function and its test:

```python
# my_module.py
def sum(a: int, b: int) -> int:
    return a + b
```

```python
# test_my_module.py
from my_module import sum

def test_sum():
    assert sum(2, 3) == 5
    assert sum(-1, 1) == 0
```

## Running Tests

- From the root of the repository, run:

```bash
pytest
```

- This will automatically discover and run all tests.

## Continuous Integration (CI)

- All pull requests are automatically checked for code style and tests.
- If tests fail or style checks do not pass, the pull request will be rejected until the issues are fixed.

## Sample Test Data for macro_data

### Overview

The `macro_data` package includes comprehensive sample test data located in:
```
tests/test_macro_data/unit/sample_raw_data/
```

This sample data mirrors the structure and format of the full datasets used by the simulation framework, but contains only a subset of countries and time periods to keep tests fast and the repository size manageable.

### Sample Data Structure

The sample data is organized by data source:

```
sample_raw_data/
├── eurostat/                 # European statistics (GDP, debt, CPI, etc.)
├── world_bank/              # World Bank indicators (inflation, population, etc.)
├── oecd_econ/               # OECD economic data (employment, taxes, etc.)
├── icio/                    # Input-Output tables
├── wiod_sea/                # World Input-Output Database
├── hfcs/2014/               # Household Finance and Consumption Survey
├── compustat/               # Corporate financial data
├── exchange_rates/          # Currency exchange rates
├── emissions/               # Energy price data
└── [other_data_sources]/    # Additional data sources
```

### Sample Data Characteristics

- **Primary test country**: France (FRA)
- **Additional countries**: Canada (CAN), UK (GBR), USA
- **Time focus**: Primarily 2014 (simulation year) with historical data 2010-2018
- **Representative values**: Real data values, not synthetic or dummy data
- **File formats**: CSV, PKL, JSON, XLSB depending on source

### Testing Requirements for New Data Sources

**When adding a new data source to macro_data, you MUST:**

1. **Add sample data** to `tests/test_macro_data/unit/sample_raw_data/[your_data_source]/`
2. **Follow the subset pattern**: Include data for France (FRA) and at least one other country
3. **Include test years**: Focus on 2014 with some historical context if relevant
4. **Use real data format**: Same structure and format as the full dataset
5. **Keep size reasonable**: Subset the data to essential test cases

### Example: Adding New Data Source Sample Data

```
sample_raw_data/
└── my_new_data_source/
    ├── emissions_data.csv         # Main data file
    ├── metadata.json             # Supporting metadata
    └── 2014/
        └── detailed_data.csv     # Year-specific data if needed
```

### Sample Data Integration

**Reader Integration:**
```python
# In your reader class
@classmethod
def from_data(cls, data_path: Path) -> "MyNewReader":
    """Create reader from data directory."""
    # Reader should work with both full data and sample data
    data_file = data_path / "my_data_file.csv"
    data = pd.read_csv(data_file)
    return cls(data=data)
```

**Test Integration:**
```python
# In your test file
def test_my_reader_functionality(readers):
    """Test reader with sample data."""
    # Use the readers fixture which provides sample data
    result = readers.my_new_reader.get_data_for_country("FRA", 2014)
    
    # Validate against expected sample data values
    assert result == pytest.approx(expected_value, abs=tolerance)
```

### Validation Testing Pattern

All sample data should be validated with specific value tests:

```python
def test_sample_data_values(readers):
    """Test that sample data returns expected values."""
    
    # Test specific known values from sample data
    gdp_fra_2014 = readers.eurostat.get_quarterly_gdp("FRA", 2014, 1)
    assert gdp_fra_2014 == pytest.approx(535467e6, abs=1e6)
    
    # Test data availability
    assert readers.my_new_reader.has_data_for_country("FRA")
    
    # Test data consistency
    assert len(readers.my_new_reader.get_country_list()) >= 1
```

### Sample Data Maintenance

- **Keep synchronized**: Sample data should reflect the structure of full data
- **Update when needed**: If full data format changes, update sample data accordingly
- **Document changes**: Note any changes to sample data in test documentation
- **Validate regularly**: Ensure sample data tests continue to pass

## Best Practices

- Validate data and check for edge cases in your tests.
- Use fixtures for setup if needed.
- Tests also serve as documentation for expected behavior.
- **For macro_data**: Always include sample test data for new data sources.
- **Test with real values**: Use actual data values in sample data, not synthetic ones.
- **Validate specific values**: Test against known values from sample data to catch regressions.

---

For more on workflow and style, see the Development Guide and Code Style Guide.
