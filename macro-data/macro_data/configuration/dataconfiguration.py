from datetime import date
from typing import Optional, Any

from pydantic import BaseModel

from .countries import Country


class FirmsDataConfiguration(BaseModel):
    """
    Configuration settings for firms.

    Attributes:
        zero_initial_deposits (bool): Whether to set initial deposits to zero.
        zero_initial_debt (bool): Whether to set initial debt to zero.
        initial_inventory_to_input_fraction (float): Initial inventory to input fraction.
        intermediate_inputs_utilisation_rate (float): Intermediate inputs utilisation rate.
        capital_inputs_utilisation_rate (float): Capital inputs utilisation rate.
    """

    zero_initial_deposits: bool = True
    zero_initial_debt: bool = True
    initial_inventory_to_input_fraction: float = 0
    intermediate_inputs_utilisation_rate: float = 1.0
    capital_inputs_utilisation_rate: float = 1.0


class InterestRates(BaseModel):
    """
    Represents the interest rates for different types of loans.

    Attributes:
        consumption_loans_markup (float): Markup for consumption loans.
        mortgage_markup (float): Markup for mortgages.
        household_overdraft_markup (float): Markup for household overdrafts.
    """

    consumption_loans_markup: float = 0.01
    mortgage_markup: float = 0.1
    household_overdraft_markup: float = 0.01


class BanksDataConfiguration(BaseModel):
    """
    Configuration class for banks.

    Attributes:
        long_term_firm_loan_maturity (int): The maturity period for long-term firm loans.
        consumption_exp_loan_maturity (int): The maturity period for household consumption expansion loans.
        mortgage_maturity (int): The maturity period for mortgages.
        interest_rates (InterestRates): The interest rates configuration.
    """

    long_term_firm_loan_maturity: int = 60
    consumption_exp_loan_maturity: int = 12
    mortgage_maturity: int = 120
    interest_rates: InterestRates = InterestRates()


class CountryDataConfiguration(BaseModel):
    """
    Represents the configuration for a country.

    Attributes:
        firms_configuration (FirmsDataConfiguration): The configuration for firms.
        banks_configuration (BanksDataConfiguration): The configuration for banks.
        single_bank (bool): Single bank flag.
        single_firm_per_industry (bool): Single firm per industry flag.
        single_government_entity (bool): Single government entity flag.
        scale (int): scale of the country (number of agents represented by a synthetic agent).
        eu_proxy_country (Country): EU proxy country (optional, if the country is not in the EU, part of the data will
                                    be generated using the EU proxy country).
    """

    firms_configuration: FirmsDataConfiguration
    banks_configuration: BanksDataConfiguration
    single_bank: bool
    single_firm_per_industry: bool
    single_government_entity: bool
    scale: int
    eu_proxy_country: Optional[Country] = None


class DataConfiguration(BaseModel):
    """
    Represents a configuration object for the data package.

    Attributes:
        industries (list[str]): List of industries.
        year (int): Initial year.
        quarter (int): Initial Quarter.
        prune_date (date): Prune date value.
        country_configs (dict[Country, CountryDataConfiguration]): Dictionary of country configurations.
        purpose (str): Purpose for this simulation.
        author (str): Author of this simulation.
    """

    industries: list[str]
    year: int
    quarter: int = 1
    prune_date: date
    country_configs: dict[Country, CountryDataConfiguration]
    purpose: str = ""
    author: str = "INET"

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization method.

        Args:
            __context (Any): The context.
        """
        # for each country config, raise an error if country is not in eu and eu proxy country is not set
        for country, country_config in self.country_configs.items():
            if country_config.eu_proxy_country is None and not country.is_eu_country:
                raise ValueError(f"{country} is not in EU: please set an EU proxy country.")

    @property
    def countries(self) -> list[Country]:
        """
        Get the list of countries.

        Returns:
            list: List of countries.
        """
        return list(self.country_configs.keys())
