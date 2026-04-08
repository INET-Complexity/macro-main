from typing import Optional

from pydantic import BaseModel

from .bank_configuration import BanksConfiguration
from .central_bank_configuration import CentralBankConfiguration
from .central_government_configuration import CentralGovernmentConfiguration
from .credit_market_configuration import CreditMarketConfiguration
from .economy_configuration import EconomyConfiguration
from .exchange_rates_configuration import ExchangeRatesConfiguration
from .firms_configuration import FirmsConfiguration
from .government_entities_configuration import GovernmentEntitiesConfiguration
from .households_configuration import HouseholdsConfiguration
from .housing_market_configuration import HousingMarketConfiguration
from .individuals_configuration import IndividualsConfiguration
from .labour_market_configuration import LabourMarketConfiguration


class CountryConfiguration(BaseModel):
    """
    The configuration settings for the country.
    Attributes:
        economy: The configuration settings for the economy.
        individuals: The configuration settings for the individuals.
        households: The configuration settings for the households.
        firms: The configuration settings for the firms.
        government_entities: The configuration settings for the government entities.
        central_government: The configuration settings for the central government.
        central_bank: The configuration settings for the central bank.
        banks: The configuration settings for the banks.
        exchange_rates: The configuration settings for the exchange rates.
        labour_market: The configuration settings for the labour market.
        housing_market: The configuration settings for the housing market.
        credit_market: The configuration settings for the credit market.
        forecasting_window: The number of periods to forecast.
        assume_zero_growth: Whether to assume zero growth in the economy.
        assume_zero_noise: Whether to assume zero noise in the economy.
        use_consumer_carbon_reg: Whether to use the consumer carbon regulation.
        use_obps_reg: Whether to use the OBPS regulation.
        use_emission_multiplier: Whether to use the emission multiplier.
    """

    economy: EconomyConfiguration = EconomyConfiguration()
    individuals: IndividualsConfiguration = IndividualsConfiguration()
    households: HouseholdsConfiguration = HouseholdsConfiguration()
    firms: FirmsConfiguration = FirmsConfiguration()
    government_entities: GovernmentEntitiesConfiguration = GovernmentEntitiesConfiguration()
    central_government: CentralGovernmentConfiguration = CentralGovernmentConfiguration()
    central_bank: CentralBankConfiguration = CentralBankConfiguration()
    banks: BanksConfiguration = BanksConfiguration()
    exchange_rates: ExchangeRatesConfiguration = ExchangeRatesConfiguration()
    labour_market: LabourMarketConfiguration = LabourMarketConfiguration()
    housing_market: HousingMarketConfiguration = HousingMarketConfiguration()
    credit_market: CreditMarketConfiguration = CreditMarketConfiguration()

    forecasting_window: int = 60
    assume_zero_growth: bool = False
    assume_zero_noise: bool = False

    use_consumer_carbon_reg: bool = False  
    use_obps_reg: bool = False  

    use_emission_multiplier: bool = False

    @classmethod
    def n_industry_default(
        cls,
        n_industries: int,
        firms_bundles: Optional[list[list[int]]] = None,
        household_bundles: Optional[list[list[int]]] = None,
    ):
        firms_configuration = FirmsConfiguration.n_industries_default(n_industries=n_industries, bundles=firms_bundles)
        households_configuration = HouseholdsConfiguration.n_industries_default(
            n_industries=n_industries, bundles=household_bundles
        )
        # all other variables are default
        return cls(firms=firms_configuration, households=households_configuration)
