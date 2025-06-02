# DataWrapper

The `DataWrapper` class is the main container for managing synthetic economic data across multiple countries. It coordinates the creation and initialization of synthetic countries and their components.

## Core Components

### Synthetic Countries

The `SyntheticCountry` class manages individual country data and coordinates all economic agents and markets. It provides:

- Unified interface for creating synthetic economic data
- Agent initialization and configuration
- Market setup and coordination
- Data validation and consistency checks
- Economic relationship management

[View SyntheticCountry documentation →](processing/synthetic_country.md)

### Economic Agents

- [SyntheticFirms](processing/synthetic_firms/index.md): Firm-level data and behavior
- [SyntheticBanks](processing/synthetic_banks/index.md): Banking sector data and operations
- [SyntheticPopulation](processing/synthetic_population/index.md): Household and individual data
- [SyntheticCentralBank](processing/synthetic_central_bank/index.md): Monetary policy and operations
- [SyntheticCentralGovernment](processing/synthetic_central_government/index.md): Fiscal policy and operations
- [SyntheticGovernmentEntities](processing/synthetic_government_entities/index.md): Government agencies and institutions
- [SyntheticRestOfTheWorld](processing/synthetic_rest_of_the_world/index.md): External economic relationships

### Markets and Data Harmonization

- [SyntheticCreditMarket](processing/synthetic_credit_market/index.md): Credit and lending operations
- [SyntheticHousingMarket](processing/synthetic_housing_market/index.md): Housing market operations
- [SyntheticGoodsMarket](processing/synthetic_goods_market/index.md): Goods and services market
- [SyntheticMatching](processing/synthetic_matching/index.md): Data harmonization between agents

## Usage Example

```python
from macro_data import DataWrapper
from macro_data.configuration.countries import Country

# Initialize a synthetic country
country = SyntheticCountry(
    country=Country("FRA"),
    data_readers=data_readers,
    configuration=config
)

# Create the synthetic economic system
country.create()

# Access different components
firms = country.firms
banks = country.banks
population = country.population
central_bank = country.central_bank
```

## Best Practices

1. **Data Consistency**
   - Ensure all economic relationships are properly initialized
   - Validate agent-level data against aggregate statistics
   - Maintain accounting identities

2. **Configuration Management**
   - Use appropriate country-specific settings
   - Configure proxy relationships for missing data
   - Document all configuration parameters

3. **Performance Optimization**
   - Use efficient data structures
   - Implement parallel processing where possible
   - Cache intermediate results

4. **Data Quality**
   - Validate input data
   - Handle missing values appropriately
   - Document data transformations

::: macro_data.data_wrapper.DataWrapper
    options:
        members:
            - DataWrapper
            - from_config
            - init_from_pickle
            - save
            - all_country_names
            - n_industries
            - calibration_before
            - calibration_during
