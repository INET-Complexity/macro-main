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
    # "individuals_quitting_temperature" = 1
    # "random_firing_probability" = 0

    # "firm_growth_adjustment_speed" : 0.0
    # "sectoral_growth_adjustment_speed" : 0.0
    # "target_inventory_to_production_fraction" : 0.0
    # "intermediate_inputs_target_considers_labour_inputs" : 0.0
    # "intermediate_inputs_target_considers_intermediate_inputs" : 0.0
    # "capital_inputs_target_considers_labour_inputs" : 0.0
    # "capital_inputs_target_considers_intermediate_inputs" : 0.0

    """
    #use this to try and retrieve these values
    country_conf.firms.functions.prices.parameters["price_setting_noise_std"] = 0.05
    country_conf.firms.functions.prices.parameters["price_setting_speed_gf"] = 1.0
    country_conf.firms.functions.prices.parameters["price_setting_speed_dp"] = 0
    country_conf.firms.functions.prices.parameters["price_setting_speed_cp"] = 0
    """

    # use this to genuinely calibrate
    country_conf.firms.functions.prices.parameters["price_setting_noise_std"] = params[0]
    country_conf.firms.functions.prices.parameters["price_setting_speed_gf"] = params[1]
    country_conf.firms.functions.prices.parameters["price_setting_speed_dp"] = params[2]
    country_conf.firms.functions.prices.parameters["price_setting_speed_cp"] = params[3]
    country_conf.labour_market.functions.clearing.parameters["individuals_quitting_temperature"] = params[4]
    country_conf.labour_market.functions.clearing.parameters["random_firing_probability"] = params[5]

    """
    country_conf.firms.functions.demand_estimator.parameters["firm_growth_adjustment_speed"] = params[4]
    country_conf.firms.functions.demand_estimator.parameters["sectoral_growth_adjustment_speed"] = params[5]
    country_conf.firms.functions.target_production.parameters["target_inventory_to_production_fraction"] = params[6]
    country_conf.firms.functions.target_production.parameters["intermediate_inputs_target_considers_labour_inputs"] = params[7]
    country_conf.firms.functions.target_production.parameters["intermediate_inputs_target_considers_intermediate_inputs"] = params[8]
    country_conf.firms.functions.target_production.parameters["capital_inputs_target_considers_labour_inputs"] = params[9]
    country_conf.firms.functions.target_production.parameters["capital_inputs_target_considers_intermediate_inputs"] = params[10]       

    """

    return country_conf


def configuration_updater(configuration: SimulationConfiguration, params: np.ndarray) -> SimulationConfiguration:
    for country_name, country_configuration in configuration.country_configurations.items():
        configuration.country_configurations[country_name] = update_country_conf(country_configuration, params)
    return configuration


def observer(simulation: Simulation) -> np.ndarray:
    country = simulation.countries["FRA"]

    gdp_growth = np.diff(np.log(country.economy.gdp_output()))
    unemp_growth = np.diff(np.log(country.economy.unemployment_rate()))
    # cpi_growth = np.diff(np.log(country.economy.total_cpi_inflation()))

    return np.array([np.nanmean(gdp_growth), np.nanmean(unemp_growth)])


# %%
# %%


def prior_sampler(n_samples: int) -> np.ndarray:
    return np.random.uniform(0, 1, size=(n_samples, 6))


if __name__ == "__main__":
    sampler = Sampler.default(
        configuration_updater=configuration_updater,
        observer=observer,
        pickle_path=PKL_PATH,
    )

    # %%
    # these are the samples to be changed
    samples = sampler.parallel_run(n_runs=40, prior_sampler=prior_sampler)

    # %%

    # dump to pickle

    with open("samples.pkl", "wb") as f:
        pkl.dump(samples, f)

    # %%
