# %%
import pickle as pkl
from copy import deepcopy

import numpy as np

from macrocalib.sampler import Sampler
from macromodel.configurations.country_configuration import CountryConfiguration
from macromodel.configurations.simulation_configuration import SimulationConfiguration
from macromodel.simulation import Simulation

PKL_PATH = "data.pkl"

# %%


def update_country_conf(country_conf: CountryConfiguration, params: np.ndarray) -> CountryConfiguration:
    country_conf = deepcopy(country_conf)

    # set whichever value you want
    # firm price paremeters
    #  "price_setting_noise_std": 0.05,
    # "price_setting_speed_gf": 1.0,
    # "price_setting_speed_dp": 0.0,
    # "price_setting_speed_cp": 0.0,
    country_conf.firms.functions.prices.parameters["price_setting_noise_std"] = params[0]
    country_conf.firms.functions.prices.parameters["price_setting_speed_gf"] = params[1]
    country_conf.firms.functions.prices.parameters["price_setting_speed_dp"] = params[2]
    country_conf.firms.functions.prices.parameters["price_setting_speed_cp"] = params[3]
    return country_conf


def configuration_updater(configuration: SimulationConfiguration, params: np.ndarray) -> SimulationConfiguration:
    for country_name, country_configuration in configuration.country_configurations.items():
        configuration.country_configurations[country_name] = update_country_conf(country_configuration, params)
    return configuration


def observer(simulation: Simulation) -> np.ndarray:
    country = simulation.countries["CAN"]

    gdp_growth = np.diff(np.log(country.economy.gdp_output()))
    unemp_growth = np.diff(np.log(country.economy.unemployment_rate()))

    return np.array([np.nanmean(gdp_growth), np.nanmean(unemp_growth)])


# %%
sampler = Sampler.default(
    configuration_updater=configuration_updater,
    observer=observer,
    pickle_path=PKL_PATH,
)

# %%


def prior_sampler(n_samples: int) -> np.ndarray:
    return np.random.uniform(0, 1, size=(n_samples, 4))


# %%

samples = sampler.parallel_run(n_runs=20, prior_sampler=prior_sampler)

# %%

# dump to pickle

with open("samples.pkl", "wb") as f:
    pkl.dump(samples, f)


# %%
