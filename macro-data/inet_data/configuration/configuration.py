from datetime import date

from pydantic import BaseModel

from .countries import Country


class FirmsConfiguration(BaseModel):
    """
    Configuration settings for firms.

    Attributes:
        zero_initial_deposits (bool): Whether to set initial deposits to zero.
        zero_initial_debt (bool): Whether to set initial debt to zero.
    """

    zero_initial_deposits: bool
    zero_initial_debt: bool


class InterestRates(BaseModel):
    """
    Represents the interest rates for different types of loans.

    Attributes:
        consumption_loans_markup (float): Markup for consumption loans.
        mortgage_markup (float): Markup for mortgages.
        household_overdraft_markup (float): Markup for household overdrafts.
    """

    consumption_loans_markup: float
    mortgage_markup: float
    household_overdraft_markup: float


class BanksConfiguration(BaseModel):
    """
    Configuration class for banks.

    Attributes:
        long_term_firm_loan_maturity (int): The maturity period for long-term firm loans.
        consumption_exp_loan_maturity (int): The maturity period for household consumption expansion loans.
        mortgage_maturity (int): The maturity period for mortgages.
        interest_rates (InterestRates): The interest rates configuration.
    """

    long_term_firm_loan_maturity: int
    consumption_exp_loan_maturity: int
    mortgage_maturity: int
    interest_rates: InterestRates


class CountryConfiguration(BaseModel):
    """
    Represents the configuration for a country.

    Attributes:
        firms_configuration (FirmsConfiguration): The configuration for firms.
        banks_configuration (BanksConfiguration): The configuration for banks.
    """

    firms_configuration: FirmsConfiguration
    banks_configuration: BanksConfiguration


class Configuration(BaseModel):
    """
    Represents a configuration object for the data package.

    Attributes:
        industries (list[str]): List of industries.
        scale (int): Scale value.
        year (int): Year value.
        prune_date (date): Prune date value.
        single_bank (bool): Single bank flag.
        single_firm_per_industry (bool): Single firm per industry flag.
        single_government_entity (bool): Single government entity flag.
        country_configs (dict[Country, CountryConfiguration]): Dictionary of country configurations.
        purpose (str): Purpose for this simulation.
        author (str): Author of this simulation.
    """

    industries: list[str]
    scale: int
    year: int
    prune_date: date
    single_bank: bool
    single_firm_per_industry: bool
    single_government_entity: bool
    country_configs: dict[Country, CountryConfiguration]
    purpose: str
    author: str = "INET/Macrocosm"

    @property
    def countries(self) -> list[Country]:
        """
        Get the list of countries.

        Returns:
            list: List of countries.
        """
        return list(self.country_configs.keys())
