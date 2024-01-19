from typing import Optional

from pydantic import BaseModel, Field

from configurations import (
    CountryConfiguration,
    RestOfTheWorldConfiguration,
    GoodsMarketConfiguration,
    ExchangeRatesConfiguration,
)


class SimulationConfiguration(BaseModel):
    """
    Configuration for the simulation.

    Attributes:
    - max_time (int): The span of the simulation (in months).
    - seed (int, optional): The seed for the random number generator. Defaults to None.
    - country_configurations (dict[str, CountryConfiguration]): The configuration for each country.
    - row_configuration (RestOfTheWorldConfiguration): The configuration for the rest of the world.
    - goods_market_configuration (GoodsMarketConfiguration): The configuration for the goods market.
    - exchange_rates_configuration (ExchangeRatesConfiguration): The configuration for the exchange rates.
    """

    t_max: int = 20
    seed: Optional[int] = None
    country_configurations: dict[str, CountryConfiguration] = {}
    row_configuration: RestOfTheWorldConfiguration = RestOfTheWorldConfiguration()
    goods_market_configuration: GoodsMarketConfiguration = GoodsMarketConfiguration()
    exchange_rates_configuration: ExchangeRatesConfiguration = ExchangeRatesConfiguration()
