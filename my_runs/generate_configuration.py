# %%
from typing import Optional

import numpy as np
import yaml

from macromodel.configurations import CountryConfiguration, SimulationConfiguration

COUNTRY_CONF = "./labour_country_conf.yaml"

SAMPLES = "./labour_small_samples.npy"

with open(COUNTRY_CONF, "r") as f:
    country_conf_dict = yaml.safe_load(f)

samples = np.load(SAMPLES)

N_INDUSTRIES = 46
BUNDLES = ["B05a", "B05b", "B05c", "C19", "D"]


def generate_base_configuration(
    country_conf_dict_: dict = country_conf_dict, n_industries: int = None, bundles: list[list[int]] = None
) -> CountryConfiguration:
    # Create base configuration with bundled industries if provided
    if n_industries is not None and bundles is not None:
        country_conf = CountryConfiguration.n_industry_default(n_industries=n_industries, bundles=bundles)
    else:
        country_conf = CountryConfiguration(**country_conf_dict)
    return country_conf


def update_country_configuration(
    country_configuration: CountryConfiguration, params: np.ndarray
) -> CountryConfiguration:
    price_gf = params[0]
    price_dp = params[1]
    price_cp = params[2]

    firing_prob = 10 ** params[3]
    quitting_temp = 10 ** params[4]

    country_configuration.firms.functions.prices.parameters["price_setting_speed_gf"] = price_gf
    country_configuration.firms.functions.prices.parameters["price_setting_speed_dp"] = price_dp
    country_configuration.firms.functions.prices.parameters["price_setting_speed_cp"] = price_cp

    country_configuration.labour_market.functions.clearing.parameters["random_firing_probability"] = firing_prob
    country_configuration.labour_market.functions.clearing.parameters["individuals_quitting_temperature"] = (
        quitting_temp
    )

    return country_configuration


# use_carbon_price: bool = False
# asymptotic_carbon_price: float = 5.0
# carbon_price_time_constant: float = 10.0
# carbon_price_t_init: int = 0


def sample_configuration(
    samples_: np.ndarray = samples,
    t_max: int = 20,
    seed: int = 0,
    use_carbon_price: bool = False,
    asymptotic_carbon_price: float = 5.0,
    carbon_price_time_constant: float = 10.0,
    carbon_price_t_init: int = 0,
    n_industries: int = N_INDUSTRIES,
    bundles: list[list[int]] = BUNDLES,
    fixed_sample: Optional[int] = 42,
) -> SimulationConfiguration:
    country_conf = generate_base_configuration(n_industries=n_industries, bundles=bundles)

    if use_carbon_price:
        country_conf.use_carbon_price = True
        country_conf.asymptotic_carbon_price = asymptotic_carbon_price
        country_conf.carbon_price_time_constant = carbon_price_time_constant
        country_conf.carbon_price_t_init = carbon_price_t_init

    n_samples = samples_.shape[0]

    if fixed_sample is not None:
        sample = samples_[fixed_sample]
    else:
        sampled_index = np.random.choice(n_samples)
        sample = samples_[sampled_index]

    country_conf = update_country_configuration(country_conf, sample)
    return SimulationConfiguration(country_configurations={"CAN": country_conf}, t_max=t_max, seed=seed)


# %%

# Example usage with bundled industries:
# simulation_configuration = sample_configuration(
#     n_industries=len(industries),
#     bundles=substitution_bundles
# )
