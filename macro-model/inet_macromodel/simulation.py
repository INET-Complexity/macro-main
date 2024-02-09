import h5py
import logging
import numpy as np
from dataclasses import dataclass
from inet_data import DataWrapper
from pathlib import Path

from inet_macromodel.configurations import SimulationConfiguration
from inet_macromodel.country import Country
from inet_macromodel.exchange_rates import ExchangeRates
from inet_macromodel.goods_market import GoodsMarket
from inet_macromodel.rest_of_the_world import RestOfTheWorld
from inet_macromodel.timestep import Timestep


@dataclass
class Simulation:
    countries: dict[str, Country]
    rest_of_the_world: RestOfTheWorld
    goods_market: GoodsMarket
    exchange_rates: ExchangeRates
    timestep: Timestep
    configuration: SimulationConfiguration

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

        goods_market_participants = {
            country_name: country.get_goods_market_participants() for country_name, country in countries.items()
        }

        goods_market_participants["ROW"] = [rest_of_the_world]

        goods_market = GoodsMarket.from_data(
            n_industries=datawrapper.n_industries,
            trade_proportions=datawrapper.trade_proportions,
            configuration=simulation_configuration.goods_market_configuration,
            goods_market_participants=goods_market_participants,
        )

        if simulation_configuration.seed:
            np.random.seed(simulation_configuration.seed)

        timestep = Timestep(year=datawrapper.configuration.year, month=1)

        return cls(
            countries=countries,
            rest_of_the_world=rest_of_the_world,
            goods_market=goods_market,
            exchange_rates=exchange_rates,
            timestep=timestep,
            configuration=simulation_configuration,
        )

    @property
    def t_max(self):
        return self.configuration.t_max

    @property
    def random_seed(self):
        return self.configuration.seed

    def iterate(self):
        self.exchange_rates.set_current_exchange_rates(current_year=self.timestep.year)

        for ind, country in enumerate(self.countries.values()):
            logging.info("Country: %s", country.country_name)
            country.initialisation_phase(exchange_rate_usd_to_lcu=self.exchange_rates.ts.current("exchange_rates")[ind])
            country.estimation_phase()
            country.target_setting_phase()
            country.clear_labour_market()
            country.update_planning_metrics()

            # Clearing the housing and the credit market
            logging.info("Clearing the housing and the credit market")
            country.prepare_housing_market_clearing()
            country.clear_housing_market()
            country.prepare_credit_market_clearing()
            country.clear_credit_market()
            country.process_housing_market_clearing()
            country.process_credit_market_clearing()

            # Prepare goods market clearing
            logging.info("Prepare goods market clearing")
            country.prepare_goods_market_clearing()

        # Prepare goods market clearing
        logging.info("Prepare goods market clearing (ROW)")
        self.rest_of_the_world.update_planning_metrics(
            average_country_ppi_inflation=np.mean(
                [self.countries[c].economy.ts.current("ppi_inflation")[0] for c in self.countries.keys()]
            ),
        )

        logging.info("Clearing the goods market")
        # Clearing the goods market
        self.goods_market.prepare()
        self.goods_market.clear()
        self.goods_market.record()

        logging.info("Updating metrics")
        # After goods market clearing
        self.rest_of_the_world.record_bought_goods()
        for country in self.countries.values():
            country.update_realised_metrics()
            country.update_population_structure()

        # Next month
        self.timestep.step()

    def run(self):
        for _ in range(self.t_max):
            self.iterate()

    def save_random_seed(self, h5_file: h5py.File) -> None:
        if self.random_seed:
            h5_file.attrs["random_seed"] = self.random_seed
        else:
            h5_file.attrs["random_seed"] = "no_seed"

    def save_configuration(self, h5_file: h5py.File) -> None:
        conf_string = self.configuration.model_dump()
        h5_file.attrs["configuration"] = str(conf_string)

    def save(self, save_dir: Path, file_name: str):
        with h5py.File(save_dir / file_name, "w") as f:
            self.save_random_seed(f)
            self.save_configuration(f)
            self.exchange_rates.save_to_h5(f)
            self.rest_of_the_world.save_to_h5(f)
            self.goods_market.save_to_h5(f)
            for country in self.countries.values():
                country.save_to_h5(f)
