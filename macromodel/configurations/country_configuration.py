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
