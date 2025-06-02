# SyntheticPopulation

The SyntheticPopulation module provides data structures and utilities for preprocessing and organizing population data that will be used to initialize behavioral models in the simulation package.

## SyntheticPopulation

The `SyntheticPopulation` class is an abstract base class that provides a framework for collecting and organizing population data. It is not used for simulating population behavior - it only handles data preprocessing.

### Key Features

- Household and individual data management
- Income and wealth computation
- Consumption and investment patterns
- Labor market integration
- Social transfer processing
- Data validation and cleaning

### Attributes

- `country_name` (str): Country identifier
- `country_name_short` (str): Short country code
- `scale` (int): Population scaling factor
- `year` (int): Reference year
- `industries` (list[str]): List of industries
- `individual_data` (pd.DataFrame): Individual-level data containing:
  - Demographics (age, gender, education)
  - Employment status and industry
  - Income sources
  - Household and firm associations
- `household_data` (pd.DataFrame): Household-level data containing:
  - Household composition
  - Income and wealth components
  - Housing tenure and property
  - Financial assets and debt
  - Consumption patterns

### Abstract Methods

#### compute_household_income

```python
@abstractmethod
def compute_household_income(
    self,
    total_social_transfers: float,
    independents: Optional[list[str]] = None
) -> None
```

Computes household income from all sources:
- Employee income
- Social transfers
- Rental income
- Financial asset returns

#### compute_household_wealth

```python
@abstractmethod
def compute_household_wealth(
    self,
    independents: Optional[list[str]] = None
) -> None
```

Computes household wealth components:
- Real assets (property, vehicles, businesses)
- Financial assets (deposits, investments)
- Debt obligations
- Net wealth position

#### set_debt_installments

```python
@abstractmethod
def set_debt_installments(
    self,
    consumption_installments: np.ndarray,
    ce_installments: np.ndarray,
    mortgage_installments: np.ndarray
) -> None
```

Sets household debt payment schedules:
- Consumption loan payments
- Consumer electronics installments
- Mortgage payments

#### set_household_saving_rates

```python
@abstractmethod
def set_household_saving_rates(
    self,
    independents: Optional[list[str]] = None
) -> None
```

Computes household saving rates based on:
- Income levels
- Wealth position
- Household characteristics

## SyntheticHFCSPopulation

The `SyntheticHFCSPopulation` class is a concrete implementation that preprocesses population data using the Household Finance and Consumption Survey (HFCS) as its primary data source.

### Key Features

- HFCS data integration
- Household sampling and scaling
- Industry employment allocation
- Wealth and income modeling
- Consumption pattern estimation

### Factory Methods

#### from_readers

```python
@classmethod
def from_readers(
    cls,
    readers: DataReaders,
    country_name: Country,
    country_name_short: str,
    scale: int,
    year: int,
    quarter: int,
    industry_data: dict[str, pd.DataFrame],
    industries: list[str],
    total_unemployment_benefits: float,
    exogenous_data: ExogenousCountryData,
    rent_as_fraction_of_unemployment_rate: float = 0.25,
    n_quantiles: int = 5,
    population_ratio: float = 1.0,
    exch_rate: float = 1.0,
    proxied_country: str | Country = None,
    yearly_factor: float = 4.0
) -> "SyntheticHFCSPopulation"
```

Creates a synthetic population using HFCS data and additional sources.

**Parameters:**
- `readers` (DataReaders): Data source readers
- `country_name` (Country): Target country
- `country_name_short` (str): Country code
- `scale` (int): Population scaling factor
- `year` (int): Reference year
- `quarter` (int): Reference quarter
- `industry_data` (dict): Industry-level data
- `industries` (list[str]): Target industries
- `total_unemployment_benefits` (float): Total benefits to distribute
- `exogenous_data` (ExogenousCountryData): External economic data
- `rent_as_fraction_of_unemployment_rate` (float): Rent parameter
- `n_quantiles` (int): Income quantiles for analysis
- `population_ratio` (float): Population scaling ratio
- `exch_rate` (float): Exchange rate for currency conversion
- `proxied_country` (str|Country): Proxy country for missing data
- `yearly_factor` (float): Annual to sub-annual conversion factor

**Returns:**
- `SyntheticHFCSPopulation`: Configured population instance

### Usage Example

```python
from macro_data import DataReaders, ExogenousCountryData
from macro_data.processing.synthetic_population import SyntheticHFCSPopulation

# Initialize data readers and configuration
readers = DataReaders.from_raw_data(...)
exogenous_data = ExogenousCountryData(...)
industry_data = {...}

# Create synthetic population for France in 2023 Q1
france_population = SyntheticHFCSPopulation.from_readers(
    country_name="FRA",
    country_name_short="FR",
    scale=1000,
    year=2023,
    quarter=1,
    readers=readers,
    industry_data=industry_data,
    industries=["C10T12", "C13T15"],
    total_unemployment_benefits=1e9,
    exogenous_data=exogenous_data
)

# Compute household wealth and income
france_population.compute_household_wealth()
france_population.compute_household_income(total_social_transfers=5e8)
```

::: macro_data.processing.synthetic_population.synthetic_population
    options:
        members:
            - SyntheticPopulation
            - SyntheticHFCSPopulation
            - set_individual_labour_inputs
            - industry_consumption_before_vat
            - investment_weights
            - compute_household_income
            - number_of_households
            - number_employees_by_industry
            - set_consumption_weights
            - set_debt_installments
            - set_household_saving_rates
            - compute_household_wealth
            - set_income
            - set_household_investment_rates
            - normalise_household_investment 