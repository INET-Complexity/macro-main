# %%

import pickle as pkl
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from macro_data import DataWrapper, data_wrapper
from macromodel.configurations import (
    ExchangeRatesConfiguration,
    FirmsConfiguration,
    GoodsMarketConfiguration,
    RestOfTheWorldConfiguration,
)
from macromodel.configurations.country_configuration import CountryConfiguration
from macromodel.configurations.firms_configuration import FirmsConfiguration
from macromodel.configurations.simulation_configuration import SimulationConfiguration
from macromodel.simulation import Simulation

with open("data.pkl", "rb") as f:
    data_wrapper = pkl.load(f)


data = DataWrapper.init_from_pickle("data.pkl")

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


# %%
row_configurations = RestOfTheWorldConfiguration()
Goods_Market_Configuration = GoodsMarketConfiguration()
Exchange_Rates_Configuration = ExchangeRatesConfiguration(exchange_rate_type="constant")


country_configurations = {
    "FRA": CountryConfiguration.n_industry_default(data.n_industries),
}
n_industries = data.n_industries

firms_configuration = FirmsConfiguration.n_industries_default(n_industries=n_industries)

configuration = SimulationConfiguration(
    country_configurations=country_configurations,
    firms_configuration=firms_configuration,
    row_configuration=row_configurations,
    goods_market_configuration=Goods_Market_Configuration,
    exchange_rates_configuration=Exchange_Rates_Configuration,
)

configuration = configuration_updater(
    configuration, [0.03700569, 0.61192334, 0.35643813, 0.24625966, 0.617393, 0.00518311]
)
# the jump occurs for a lot of parameters, here is one example

model = Simulation.from_datawrapper(
    datawrapper=data,
    simulation_configuration=configuration,
)

model.run()
# print(model.countries["FRA"].economy.gdp_output())
# shallow_outputs = model.shallow_df_dict()


# attempt to plot

real_data = [
    5.2560126e11,
    5.2359846e11,
    5.2519946e11,
    5.2510276e11,
    5.2607646e11,
    5.3058476e11,
    5.2935046e11,
    5.3248206e11,
    5.3254336e11,
    5.3382316e11,
    5.3714106e11,
    5.3568016e11,
    5.3875986e11,
    5.3909316e11,
    5.4045046e11,
    5.4170016e11,
    5.4337436e11,
    5.4243796e11,
    5.4316946e11,
    5.4646356e11,
    5.5011346e11,
    5.5421146e11,
    5.5804696e11,
    5.6131926e11,
    5.6115956e11,
    5.6258546e11,
    5.6522166e11,
    5.6931776e11,
    5.7452906e11,
    5.7761656e11,
    5.7754036e11,
    5.7475006e11,
    5.4626526e11,
    4.8121946e11,
    5.5312706e11,
    5.5248286e11,
]


# Create a unified x-axis: shared time period


quarters2 = [
    "2012 Q1",
    "2012 Q2",
    "2012 Q3",
    "2012 Q4",
    "2013 Q1",
    "2013 Q2",
    "2013 Q3",
    "2013 Q4",
    "2014 Q1",
    "2014 Q2",
    "2014 Q3",
    "2014 Q4",
    "2015 Q1",
    "2015 Q2",
    "2015 Q3",
    "2015 Q4",
    "2016 Q1",
    "2016 Q2",
    "2016 Q3",
    "2016 Q4",
    "2017 Q1",
    "2017 Q2",
    "2017 Q3",
    "2017 Q4",
    "2018 Q1",
    "2018 Q2",
    "2018 Q3",
    "2018 Q4",
    "2019 Q1",
    "2019 Q2",
    "2019 Q3",
    "2019 Q4",
    "2020 Q1",
    "2020 Q2",
    "2020 Q3",
    "2020 Q4",
]


quarters1 = [
    "2014 Q1",
    "2014 Q2",
    "2014 Q3",
    "2014 Q4",
    "2015 Q1",
    "2015 Q2",
    "2015 Q3",
    "2015 Q4",
    "2016 Q1",
    "2016 Q2",
    "2016 Q3",
    "2016 Q4",
    "2017 Q1",
    "2017 Q2",
    "2017 Q3",
    "2017 Q4",
    "2018 Q1",
    "2018 Q2",
    "2018 Q3",
    "2018 Q4",
    "2019 Q1",
]

plt.figure(figsize=(12, 6))
# plt.plot(quarters1, real_data[8:29], label="Real GDP (Observed)", color = 'black',linestyle='--')
plt.plot(quarters2[:29], real_data[:29], label="Real GDP (Observed)", color="black", linestyle="--")
plt.plot(quarters1, model.countries["FRA"].economy.gdp_output(), color="blue", label="Median", linewidth=2)


plt.xlabel("Year")
plt.ylabel("GDP (€)")
plt.title("Real vs Model GDP Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(["2012 Q1", "2013 Q1", "2014 Q1", "2015 Q1", "2016 Q1", "2017 Q1", "2018 Q1", "2019 Q1"])
plt.savefig("Jumptest")
plt.show()

# %%

# dump the configuration to a yaml file, configuration is a pydantic basemodel

import yaml

with open("configuration.yaml", "w") as f:
    yaml.dump(configuration.model_dump(), f)

# %%
