from dataclasses import dataclass

import numpy as np
from inet_data import DataWrapper

from configurations import SimulationConfiguration
from country import Country
from exchange_rates import ExchangeRates
from goods_market import GoodsMarket
from rest_of_the_world import RestOfTheWorld


@dataclass
class Simulation:
    countries: dict[str, Country]
    rest_of_the_world: RestOfTheWorld
    goods_market: GoodsMarket
    exchange_rates: ExchangeRates

    @classmethod
    def from_datawrapper(
        cls,
        datawrapper: DataWrapper,
        simulation_configuration: SimulationConfiguration,
    ):
        countries_without_row = list(set(datawrapper.all_country_names) - {"ROW"})
        countries_with_row = datawrapper.all_country_names

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=simulation_configuration.exchange_rates_configuration,
            initial_year=datawrapper.configuration.year,
            country_names=countries_without_row,
        )

        countries = {
            country_name: Country.from_pickled_country(
                synthetic_country=datawrapper.synthetic_countries[country_name],
                country_configuration=simulation_configuration.country_configurations[country_name],
                exchange_rates=exchange_rates,
                country_name=country_name,
                all_country_names=countries_with_row,
                industries=datawrapper.configuration.industries,
                initial_year=datawrapper.configuration.year,
                t_max=simulation_configuration.t_max,
            )
            for country_name in countries_without_row
        }

        average_ppi_inflation = np.mean(
            [countries[country_name].economy.ts.current("ppi_inflation")[0] for country_name in countries_without_row]
        )

        rest_of_the_world = RestOfTheWorld.from_pickled_row(
            country_name="ROW",
            all_country_names=countries_with_row,
            n_industries=datawrapper.n_industries,
            synthetic_row=datawrapper.synthetic_rest_of_the_world,
            configuration=simulation_configuration.row_configuration,
            average_ppi_inflation=average_ppi_inflation,
        )

        goods_market = GoodsMarket.from_data(
            n_industries=datawrapper.n_industries,
            trade_proportions=datawrapper.trade_proportions,
            configuration=simulation_configuration.goods_market_configuration,
            goods_market_participants={**countries, "ROW": rest_of_the_world},
        )

        return cls(
            countries=countries,
            rest_of_the_world=rest_of_the_world,
            goods_market=goods_market,
            exchange_rates=exchange_rates,
        )
