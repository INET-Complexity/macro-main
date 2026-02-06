import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
from sbi.utils import BoxUniform

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macrocalib.sampler import PriorSampler, Sampler
from macromodel.configurations import SimulationConfiguration
from macromodel.simulation import Simulation


@pytest.fixture(scope="module", name="datawrapper")
def instantiate_datawrapper() -> DataWrapper:
    data_config = default_data_configuration(countries=["FRA"])
    raw_data_path = Path(__file__).parent.parent.parent / "test_macro_data" / "unit" / "sample_raw_data"
    return DataWrapper.from_config(data_config, raw_data_path, single_hfcs_survey=True)


@pytest.fixture(scope="module", name="configuration_updater")
def fixture_configuration_updater() -> Callable[[SimulationConfiguration, np.ndarray], SimulationConfiguration]:
    # a function that updates the configuration, taking 5 parameters as input
    # and updating the configuration of a single country (5 parameters per country here)
    def configuration_updater(
        base_configuration: SimulationConfiguration, params: np.ndarray
    ) -> SimulationConfiguration:
        new_configuration = deepcopy(base_configuration)

        for i, country in enumerate(new_configuration.country_configurations):
            update_country_configuration(params[5 * i : 5 * (i + 1)], new_configuration, country)

        return new_configuration

    # a function that updates the configuration of a single country
    def update_country_configuration(params: np.ndarray, new_configuration: SimulationConfiguration, country: str):
        price_gf = params[0]
        price_dp = params[1]
        price_cp = params[2]

        firing_prob = 10 ** params[3]
        quitting_temp = 10 ** params[4]

        country_configuration = new_configuration.country_configurations[country]

        country_configuration.firms.functions.prices.parameters["price_setting_speed_gf"] = price_gf
        country_configuration.firms.functions.prices.parameters["price_setting_speed_dp"] = price_dp
        country_configuration.firms.functions.prices.parameters["price_setting_speed_cp"] = price_cp

        country_configuration.labour_market.functions.clearing.parameters["random_firing_probability"] = firing_prob
        country_configuration.labour_market.functions.clearing.parameters["individuals_quitting_temperature"] = (
            quitting_temp
        )

    return configuration_updater


@pytest.fixture(scope="module", name="observer")
def fixture_observer() -> Callable[[Simulation], np.ndarray]:
    def observer(simulation: Simulation) -> np.ndarray:
        # country = simulation.countries["CAN"]
        countries = [simulation.countries["FRA"]]
        sim_data = np.concatenate([country_data_array(country) for country in countries])
        return sim_data

    def country_data_array(country):
        model_df = pd.DataFrame(
            {
                "GDP": np.log(country.economy.gdp_output()),
                "Unemployment": np.log(country.economy.unemployment_rate()),
                "CPI": np.log(country.economy.total_cpi_inflation()),
            }
        )

        results = {
            "avg_gdp": model_df["GDP"].diff().mean(),
            "avg_unemp": model_df["Unemployment"].diff().mean(),
            "avg_cpi": model_df["CPI"].diff().mean(),
            "std_gdp": model_df["GDP"].diff().std(),
            "std_unemp": model_df["Unemployment"].diff().std(),
            "std_cpi": model_df["CPI"].diff().std(),
        }
        # add stds of the quantities

        # return a tensor with the means and stds of the variables
        data_array = np.array(
            [
                results["avg_gdp"],
                results["std_gdp"],
                results["avg_unemp"],
                results["std_unemp"],
                results["avg_cpi"],
                results["std_cpi"],
            ]
        )

        return data_array

    return observer


@pytest.fixture(scope="module", name="prior_sampler")
def fixture_prior_sampler() -> PriorSampler:
    low = [0, 0, 0, -4, -4]
    high = [1, 1, 1, -1, -1]

    prior = BoxUniform(
        low=torch.tensor(low),
        high=torch.tensor(high),
    )

    def prior_sampler(n_samples: int):
        return prior.sample((n_samples,)).numpy()  # type: ignore

    return prior_sampler


@pytest.fixture(scope="module", name="sampler")
def fixture_sampler(datawrapper, configuration_updater, observer):
    # first, save the datawrapper to a temp pickle file
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tmp_file = tmp / "datawrapper.pkl"
        datawrapper.save(tmp_file)

        # then, instantiate the sampler
        sampler = Sampler.default(configuration_updater=configuration_updater, observer=observer, pickle_path=tmp_file)

    return sampler
