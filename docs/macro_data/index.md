# macro_data Package Documentation

The macro_data package is responsible for creating and managing synthetic economic data for the macromodel. It handles data preprocessing, validation, and transformation to create a comprehensive dataset that represents the economic structure of multiple countries.

## Overview

The package provides tools to:

- Create synthetic country-level economic data
- Process and validate input-output tables
- Handle trade flows and exchange rates
- Manage emissions data
- Generate calibration data

## Core Components

### DataWrapper

The main container class that manages synthetic economic data for multiple countries. It serves as the primary interface for data preprocessing and management, coordinating all economic agents, markets, and their relationships.

[View DataWrapper documentation →](api/data_wrapper.md)

### Data Readers

A collection of specialized readers for different types of economic data, handling data ingestion, preprocessing, and validation.

[View Data Readers documentation →](api/readers/index.md)

## Data Structure

The package organizes data in a hierarchical structure:

1. **DataWrapper**
   - Manages all synthetic economic data
   - Handles data preprocessing and validation
   - Coordinates between different data sources

2. **Synthetic Countries**
   - Individual country data containers
   - Contains all economic agents and markets
   - Manages country-specific data and relationships

3. **Economic Agents**
   - Households
   - Firms
   - Banks
   - Government institutions

4. **Markets**
   - Credit market
   - Housing market
   - Goods market
   - Labor market

## Usage Example

```python
from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration

# Create configuration for multiple countries
data_config = default_data_configuration(
    countries=["FRA", "CAN", "USA"],
    proxy_country_dict={"CAN": "FRA", "USA": "FRA"}
)

# Initialize DataWrapper with configuration
creator = DataWrapper.from_config(
    configuration=data_config,
    raw_data_path="path/to/raw/data",
    single_hfcs_survey=True
)

# Save processed data
creator.save("path/to/save/data.pkl")
```

## Best Practices

1. **Data Validation**
   - Always validate input data before processing
   - Check for missing or inconsistent values
   - Verify economic relationships

2. **Configuration Management**
   - Use configuration files for reproducible results
   - Document all configuration parameters
   - Version control configuration files

3. **Performance Optimization**
   - Use appropriate data structures
   - Implement efficient data processing
   - Cache intermediate results when possible
