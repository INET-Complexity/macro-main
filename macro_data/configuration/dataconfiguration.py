"""
This module defines the configuration classes for the macroeconomic model's data generation.
It provides a hierarchical configuration structure that controls how synthetic data is
generated for different economic agents (firms, banks, central banks) across countries.

The configuration system is built using Pydantic models for validation and type safety,
with support for:
- Country-specific configurations
- Financial institution parameters
- Interest rate settings
- Industry aggregation options
- Emissions and carbon pricing

Example:
    ```python
    from macro_data.configuration import DataConfiguration
    from macro_data.configuration.countries import Country

    # Create configurations for different components
    firms_config = FirmsDataConfiguration(
        zero_initial_deposits=True,
        zero_initial_debt=True,
        initial_inventory_to_input_fraction=0.1
    )

    banks_config = BanksDataConfiguration(
        long_term_firm_loan_maturity=60,
        mortgage_maturity=120
    )

    # Create country configuration
    france_config = CountryDataConfiguration(
        firms_configuration=firms_config,
        banks_configuration=banks_config,
        central_bank_configuration=CentralBankDataConfiguration(),
        single_bank=True,
        single_firm_per_industry=True,
        single_government_entity=True,
        scale=1000
    )

    # Create main configuration
    config = DataConfiguration(
        year=2023,
        quarter=1,
        country_configs={Country.FRANCE: france_config},
        aggregate_industries=False
    )
    ```
"""

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .countries import Country
from .region import Region


class FirmsDataConfiguration(BaseModel):
    """
    Configuration settings for firm behavior and initial conditions.

    This class controls how synthetic firm data is generated, including initial
    financial conditions and operational parameters.

    Attributes:
        constructor (Literal["Compustat", "Default"]): The data constructor to use
        zero_initial_deposits (bool): Whether firms start with zero deposits
        zero_initial_debt (bool): Whether firms start with zero debt
        initial_inventory_to_input_fraction (float): Ratio of initial inventory to inputs
        intermediate_inputs_utilisation_rate (float): Rate at which intermediate inputs are used
        capital_inputs_utilisation_rate (float): Rate at which capital inputs are used
    """

    constructor: Literal["Compustat", "Default"] = "Compustat"
    zero_initial_deposits: bool = True
    zero_initial_debt: bool = True
    initial_inventory_to_input_fraction: float = 0
    intermediate_inputs_utilisation_rate: float = 1.0
    capital_inputs_utilisation_rate: float = 1.0


class InterestRates(BaseModel):
    """
    Configuration for interest rate markups on different types of loans.

    This class defines the markup rates that banks add to the base interest rate
    for different types of loans.

    Attributes:
        consumption_loans_markup (float): Additional markup for consumption loans
        mortgage_markup (float): Additional markup for mortgages
        household_overdraft_markup (float): Additional markup for household overdrafts
    """

    consumption_loans_markup: float = 0.01
    mortgage_markup: float = 0.1
    household_overdraft_markup: float = 0.01


class BanksDataConfiguration(BaseModel):
    """
    Configuration for bank behavior and loan parameters.

    This class controls how synthetic bank data is generated, including loan
    maturities and interest rate settings.

    Attributes:
        constructor (Literal["Compustat", "Default"]): The data constructor to use
        long_term_firm_loan_maturity (int): Maturity period (months) for long-term firm loans
        consumption_exp_loan_maturity (int): Maturity period (months) for consumption loans
        mortgage_maturity (int): Maturity period (months) for mortgages
        interest_rates (InterestRates): Interest rate markup configuration
    """

    constructor: Literal["Compustat", "Default"] = "Compustat"
    long_term_firm_loan_maturity: int = 60
    consumption_exp_loan_maturity: int = 12
    mortgage_maturity: int = 120
    interest_rates: InterestRates = InterestRates()


class CentralBankDataConfiguration(BaseModel):
    """
    Configuration for central bank parameters.

    This class defines the monetary policy parameters for the central bank.

    Attributes:
        inflation_target (float): Target inflation rate (between 0 and 1)
    """

    inflation_target: float = Field(0.02, ge=0.0, le=1.0)


class CountryDataConfiguration(BaseModel):
    """
    Comprehensive configuration for a single country's economic agents and parameters.

    This class combines configurations for all major economic agents (firms, banks,
    central bank) along with country-specific settings.

    Attributes:
        firms_configuration (FirmsDataConfiguration): Configuration for firms
        banks_configuration (BanksDataConfiguration): Configuration for banks
        central_bank_configuration (CentralBankDataConfiguration): Configuration for central bank
        single_bank (bool): Whether to use a single bank for the country
        single_firm_per_industry (bool): Whether to use one firm per industry
        single_government_entity (bool): Whether to use a single government entity
        scale (int): Number of agents represented by each synthetic agent
        eu_proxy_country (Country, optional): EU country to use as proxy for non-EU countries
        carbon_price (float): Carbon price per tonne of CO2 in local currency units
    """

    firms_configuration: FirmsDataConfiguration
    banks_configuration: BanksDataConfiguration
    central_bank_configuration: CentralBankDataConfiguration
    single_bank: bool
    single_firm_per_industry: bool
    single_government_entity: bool
    scale: int
    eu_proxy_country: Optional[Country] = None
    carbon_price: float = 0.0


class ROWDataConfiguration(BaseModel):
    """
    Configuration for Rest of World (ROW) data generation.

    This class controls how the rest of world is modeled in terms of trade
    and economic interactions.

    Attributes:
        fit_imports (bool): Whether to fit a statistical model for imports
        fit_exports (bool): Whether to fit a statistical model for exports
        assume_one_exporter_by_industry (bool): Whether to assume one exporter per industry
    """

    fit_imports: bool = False
    fit_exports: bool = False
    assume_one_exporter_by_industry: bool = True


class DataConfiguration(BaseModel):
    """
    Master configuration class for the entire data generation process.

    This class is the top-level configuration that controls all aspects of
    synthetic data generation for the macroeconomic model.

    Attributes:
        year (int): Initial year for the simulation
        quarter (int): Initial quarter (1-4) for the simulation
        time_unit (int): Time unit for the simulation (1-12), in months
        prune_date (date, optional): Date to prune data before
        country_configs (dict[Country, CountryDataConfiguration]): Per-country configurations
        row_data_config (ROWDataConfiguration): Rest of world configuration
        purpose (str): Description of the simulation's purpose
        author (str): Author of the simulation configuration
        aggregate_industries (bool): Whether to aggregate industries
        can_disaggregation (bool): Whether to enable Canadian industry disaggregation
        seed (int, optional): Random seed for reproducibility
        aggregation_structure (dict[Country, list[Country | Region]], optional):
            Maps parent entities to their components (regions/countries)

    Example:
        ```python
        config = DataConfiguration(
            year=2023,
            quarter=1,
            country_configs={
                Country.FRANCE: CountryDataConfiguration(...),
                Country.GERMANY: CountryDataConfiguration(...)
            },
            aggregate_industries=False,
            seed=42
        )
        ```
    """

    year: int
    quarter: int = 1
    time_unit: int = Field(4, ge=1, le=12)
    prune_date: Optional[date] = None
    country_configs: dict[Country, CountryDataConfiguration]
    row_data_config: ROWDataConfiguration = ROWDataConfiguration()
    purpose: str = ""
    author: str = "INET"
    aggregate_industries: bool = False
    can_disaggregation: bool = False
    seed: Optional[int] = None
    aggregation_structure: Optional[dict[Country, list[Country | Region]]] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Validate the configuration after initialization.

        This method ensures that:
        1. Non-EU countries have proxy EU countries specified
        2. All components in aggregation_structure have corresponding configs
        3. No component appears in multiple parent entities

        Args:
            __context (Any): Pydantic initialization context

        Raises:
            ValueError: If validation fails
        """
        # Check EU proxy countries
        for country, country_config in self.country_configs.items():
            if country_config.eu_proxy_country is None and not country.is_eu_country:
                raise ValueError(f"{country} is not in EU: please set an EU proxy country.")

        # Validate aggregation structure if present
        if self.aggregation_structure:
            # Check that all components have configs
            all_components = set()
            for components in self.aggregation_structure.values():
                all_components.update(components)

            missing_configs = all_components - set(self.country_configs.keys())
            if missing_configs:
                raise ValueError(f"Missing configurations for components: {missing_configs}")

            # Check for duplicate components
            component_counts = {}
            for components in self.aggregation_structure.values():
                for component in components:
                    component_counts[component] = component_counts.get(component, 0) + 1

            duplicates = {comp: count for comp, count in component_counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"Components appear in multiple parent entities: {duplicates}")

    @property
    def countries(self) -> list[Country]:
        """
        Get the list of countries in the configuration.

        Returns:
            list[Country]: List of configured countries
        """
        return list(self.country_configs.keys())

    def get_components(self, entity: Country | Region) -> list[Country | Region]:
        """
        Get all components (regions/countries) for a given entity.

        Args:
            entity (Country | Region): The parent entity to get components for

        Returns:
            list[Country | Region]: List of components, or empty list if not found
        """
        if self.aggregation_structure and entity in self.aggregation_structure:
            return self.aggregation_structure[entity]
        return []

    def is_aggregated(self, entity: Country | Region) -> bool:
        """
        Check if an entity is disaggregated into components.

        Args:
            entity (Country | Region): The entity to check

        Returns:
            bool: True if the entity has components, False otherwise
        """
        return bool(self.get_components(entity))

    def get_parent(self, component: Country | Region) -> Optional[Country]:
        """
        Get the parent entity for a component.

        Args:
            component (Country | Region): The component to find the parent for

        Returns:
            Optional[Country]: The parent entity if found, None otherwise
        """
        if not self.aggregation_structure:
            return None
        for parent, components in self.aggregation_structure.items():
            if component in components:
                return parent
        return None
