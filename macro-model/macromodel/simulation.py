from copy import deepcopy

import h5py
import logging
import numpy as np
from dataclasses import dataclass
from macro_data import DataWrapper
from pathlib import Path

from macro_data.configuration import CountryDataConfiguration

from macromodel.configurations import SimulationConfiguration, CountryConfiguration
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates
from macromodel.goods_market import GoodsMarket
from macromodel.rest_of_the_world import RestOfTheWorld
from macromodel.timestep import Timestep


@dataclass
class Simulation:
    countries: dict[str, Country]
    rest_of_the_world: RestOfTheWorld
    goods_market: GoodsMarket
    exchange_rates: ExchangeRates
    timestep: Timestep
    configuration: SimulationConfiguration
    initial_year: int
    aggregate_country_price_index: float = 1.0

    @classmethod
    def from_datawrapper(
        cls,
        datawrapper: DataWrapper,
        simulation_configuration: SimulationConfiguration,
    ):
        data_configuration = datawrapper.configuration
        for country, country_sim_conf in simulation_configuration.country_configurations.items():
            if country not in data_configuration.country_configs:
                raise ValueError(
                    f"Country {country} not found in the data configuration. " "Please use a valid data configuration."
                )
            if not check_compatibility(data_configuration.country_configs[country], country_sim_conf):  # type: ignore
                datawrapper.synthetic_countries[country].reset_firm_function_dependent(
                    **country_sim_conf.firms.reset_params,
                    zero_initial_debt=False,
                    zero_initial_deposits=False,
                )

        countries_without_row = list(set(datawrapper.all_country_names) - {"ROW"})
        countries_with_row = datawrapper.all_country_names

        running_multi_country = len(countries_without_row) > 1

        model_dict = {
            country_name: country.synthetic_goods_market.exchange_rates_model
            for country_name, country in datawrapper.synthetic_countries.items()
        }

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=simulation_configuration.exchange_rates_configuration,
            initial_year=datawrapper.configuration.year,
            country_names=countries_without_row,
            exchange_rates_model=model_dict,
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
                running_multiple_countries=running_multi_country,
            )
            for country_name in countries_without_row
        }

        rest_of_the_world = RestOfTheWorld.from_pickled_row(
            country_name="ROW",
            all_country_names=countries_with_row,
            n_industries=datawrapper.n_industries,
            synthetic_row=datawrapper.synthetic_rest_of_the_world,
            configuration=simulation_configuration.row_configuration,
            calibration_data_before=datawrapper.calibration_before,
            calibration_data_during=datawrapper.calibration_during,
        )

        goods_market_participants = {
            country_name: country.get_goods_market_participants() for country_name, country in countries.items()
        }

        goods_market_participants["ROW"] = [rest_of_the_world]

        row_index = sorted(countries_with_row).index("ROW")

        goods_market = GoodsMarket.from_data(
            n_industries=datawrapper.n_industries,
            configuration=simulation_configuration.goods_market_configuration,
            goods_market_participants=goods_market_participants,
            origin_trade_proportions=datawrapper.origin_trade_proportions.values,
            destin_trade_proportions=datawrapper.destination_trade_proportions.values,
            row_index=row_index,
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
            configuration=deepcopy(simulation_configuration),
            initial_year=datawrapper.configuration.year,
        )

    def reset(self, configuration: SimulationConfiguration) -> None:

        self.timestep = Timestep(year=self.initial_year, month=1)

        reset_row_params = configuration.row_configuration != self.configuration.row_configuration
        self.rest_of_the_world.reset(configuration.row_configuration, reset_row_params)

        reset_good_market_params = (
            configuration.goods_market_configuration != self.configuration.goods_market_configuration
        )
        self.goods_market.reset(configuration.goods_market_configuration, reset_good_market_params)

        self.exchange_rates.reset()

        for country in self.countries.values():
            country.reset(configuration.country_configurations[country.country_name])

        self.configuration = deepcopy(configuration)

    @property
    def t_max(self):
        return self.configuration.t_max

    @property
    def random_seed(self):
        return self.configuration.seed

    def iterate(self):
        # self.exchange_rates.set_current_exchange_rates(current_year=self.timestep.year)

        for ind, country in enumerate(self.countries.values()):
            exchange_rate = self.exchange_rates.get_current_exchange_rates_from_usd_to_lcu(
                country_name=country.country_name,
                current_year=self.timestep.year,
                prev_inflation=country.economy.ts.current("ppi_inflation")[0],
                prev_growth=country.economy.ts.current("total_growth")[0],
            )
            logging.info("Country: %s", country.country_name)
            country.initialisation_phase(exchange_rate_usd_to_lcu=exchange_rate)
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

        # Prepare goods market clearing
        aggregate_country_production_index = self.production_price_index
        total_real_production = self.total_real_production
        if total_real_production > 0:
            self.aggregate_country_price_index = self.aggregate_nominal_production / total_real_production
        self.rest_of_the_world.update_planning_metrics(
            aggregate_country_production_index=aggregate_country_production_index,
            aggregate_country_price_index=self.aggregate_country_price_index,
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

    @property
    def aggregate_nominal_production(self) -> float:
        return np.sum(
            [
                (
                    self.countries[c].firms.ts.current("price")
                    / self.countries[c].firms.ts.initial("price")
                    * (self.countries[c].firms.ts.current("production") + self.countries[c].firms.ts.prev("inventory"))
                ).sum()
                for c in self.countries.keys()
            ]
        )

    @property
    def total_real_production(self) -> float:
        return np.sum(
            [
                (self.countries[c].firms.ts.current("production") + self.countries[c].firms.ts.prev("inventory")).sum()
                for c in self.countries.keys()
            ]
        )

    @property
    def production_price_index(self) -> float:
        current_production = [self.countries[c].firms.ts.current("production").sum() for c in self.countries.keys()]
        initial_production = [self.countries[c].firms.ts.initial("production").sum() for c in self.countries.keys()]
        return np.sum(current_production) / np.sum(initial_production)

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

    def save(self, save_dir: Path | str, file_name: str):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        with h5py.File(save_dir / file_name, "w") as f:
            self.save_random_seed(f)
            self.save_configuration(f)
            # self.exchange_rates.save_to_h5(f)
            self.rest_of_the_world.save_to_h5(f)
            self.goods_market.save_to_h5(f)
            for country in self.countries.values():
                country.save_to_h5(f)

    def shallow_df_dict(self):
        df_dict = {country: self.countries[country].shallow_output() for country in self.countries}
        return df_dict

    def shallow_hdf_save(self, save_dir: Path | str, file_name: str):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        for country_name, country in self.countries.items():
            df = country.shallow_output()
            df.to_hdf(save_dir / file_name, key=country_name, mode="a")

    def get_country_shallow_output(self, country: str):
        return self.countries[country].shallow_output()


def check_compatibility(
    country_data_configuration: CountryDataConfiguration, country_sim_configuration: CountryConfiguration
) -> bool:
    """
    Check the compatibility of the datawrapper and the simulation configuration for a given country.

    Args:
        country_data_configuration (CountryDataConfiguration): The data configuration.
        country_sim_configuration (SimulationConfiguration): The simulation configuration.

    Returns:
        bool: True if the datawrapper and the simulation configuration are compatible, False otherwise.
    """
    firm_data_conf = country_data_configuration.firms_configuration

    firm_sim_reset_params = country_sim_configuration.firms.reset_params

    test_cases = [
        firm_data_conf.initial_inventory_to_input_fraction
        == firm_sim_reset_params["initial_inventory_to_input_fraction"],
        firm_data_conf.capital_inputs_utilisation_rate == firm_sim_reset_params["capital_inputs_utilisation_rate"],
        firm_data_conf.intermediate_inputs_utilisation_rate
        == firm_sim_reset_params["intermediate_inputs_utilisation_rate"],
    ]

    return all(test_cases)
