# SyntheticCentralBank

The SyntheticCentralBank module provides data structures and utilities for preprocessing and organizing central bank data that will be used to initialize behavioral models in the simulation package.

## SyntheticCentralBank

The `SyntheticCentralBank` class is an abstract base class that provides a framework for collecting and organizing central bank data. It is not used for simulating central bank behavior - it only handles data preprocessing.

### Key Features

- Historical policy rate collection
- Inflation and growth data aggregation
- Parameter estimation from historical data
- Data validation and organization
- Country-specific data handling

### Attributes

- `country_name` (str): Country identifier for data collection
- `year` (int): Reference year for data preprocessing
- `central_bank_data` (pd.DataFrame): Preprocessed central bank data containing:
  - Policy rates
  - Additional metrics (implementation-specific)

## DefaultSyntheticCentralBank

The `DefaultSyntheticCentralBank` class is a concrete implementation that preprocesses and organizes central bank data by estimating Taylor rule parameters from historical data.

### Key Features

- Taylor rule parameter estimation
- Interest rate smoothing calculation
- Response coefficients estimation
- Natural rate computation
- Zero lower bound enforcement

### Attributes

The preprocessed data DataFrame contains:
- `policy_rate` (float): Historical/initial policy rate
- `targeted_inflation_rate` (float): Reference inflation target
- `rho` (float): Estimated interest rate smoothing parameter
- `r_star` (float): Estimated natural real interest rate
- `xi_pi` (float): Estimated inflation response coefficient
- `xi_gamma` (float): Estimated growth response coefficient

### Factory Methods

#### from_readers

```python
@classmethod
def from_readers(
    cls,
    country_name: str,
    year: int,
    quarter: int,
    readers: DataReaders,
    exogenous_data: ExogenousCountryData,
    central_bank_configuration: CentralBankDataConfiguration
) -> DefaultSyntheticCentralBank
```

Creates a preprocessed central bank data container using historical data.

**Parameters:**
- `country_name` (str): Country to preprocess data for
- `year` (int): Reference year for preprocessing
- `quarter` (int): Reference quarter (1-4)
- `readers` (DataReaders): Data source readers
- `exogenous_data` (ExogenousCountryData): External economic data
- `central_bank_configuration` (CentralBankDataConfiguration): Configuration settings

**Returns:**
- `DefaultSyntheticCentralBank`: Container with preprocessed parameters

### Parameter Estimation

The class estimates Taylor rule parameters using the form:
```
r_t = ρr_{t-1} + (1-ρ)[r* + π* + ξ_π(π_t - π*) + ξ_γγ_t]
```
where:
- r_t: historical policy rate
- ρ: smoothing parameter
- r*: natural rate
- π*: inflation target
- π_t: historical inflation
- γ_t: historical growth

### Usage Example

```python
from macro_data import DataReaders, ExogenousCountryData
from macro_data.configuration.dataconfiguration import CentralBankDataConfiguration
from macro_data.processing.synthetic_central_bank import DefaultSyntheticCentralBank

# Initialize data readers and configuration
readers = DataReaders.from_raw_data(...)
exogenous_data = ExogenousCountryData(...)
config = CentralBankDataConfiguration(inflation_target=0.02)

# Create central bank data for France in 2023 Q1
france_central_bank = DefaultSyntheticCentralBank.from_readers(
    country_name="FRA",
    year=2023,
    quarter=1,
    readers=readers,
    exogenous_data=exogenous_data,
    central_bank_configuration=config
)

# Access estimated parameters
policy_rate = france_central_bank.central_bank_data["policy_rate"]
inflation_response = france_central_bank.central_bank_data["xi_pi"]
```

::: macro_data.processing.synthetic_central_bank.synthetic_central_bank
    options:
        members:
            - SyntheticCentralBank
            - DefaultSyntheticCentralBank 