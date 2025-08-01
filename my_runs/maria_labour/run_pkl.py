# %%
import pickle as pkl

import numpy as np
import pandas as pd

from macro_data import DataWrapper
from macro_data.configuration.countries import Country as CountryName
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation, check_compatibility

can_disagg_datawrapper = DataWrapper.init_from_pickle("canada_all_disagg.pkl")

# %%

n_industries = len(can_disagg_datawrapper.industries)
bundled_industries = ["B05a", "B05b", "B05c", "C19", "D01a", "D01b", "D01c", "D01d", "D01e"]
industries = can_disagg_datawrapper.industries
energy_bundle = [list(industries).index(ind) for ind in bundled_industries]


substitution_bundles = [energy_bundle]

configuration = SimulationConfiguration(
    country_configurations={
        "CAN": CountryConfiguration.n_industry_default(
            n_industries=n_industries,
            bundles=substitution_bundles,
        )
    },
    t_max=40,
    seed=0,
)


def get_production_dataframe(sim: Simulation, country: str = "CAN"):
    production = np.stack(sim.countries[country].firms.ts.historic("production"))
    price = np.stack(sim.countries[country].firms.ts.historic("price"))
    production_price = production * price
    return pd.DataFrame(production_price, columns=can_disagg_datawrapper.industries)


configuration.country_configurations["CAN"].asymptotic_carbon_price = 50.0
configuration.country_configurations["CAN"].carbon_price_time_constant = 10.0
configuration.country_configurations["CAN"].carbon_price_t_init = 4 * 3

# %%

configuration.country_configurations["CAN"].use_carbon_price = True
configuration.country_configurations["CAN"].asymptotic_carbon_price = 100.0
configuration.country_configurations["CAN"].carbon_price_time_constant = 3.0
configuration.country_configurations["CAN"].carbon_price_t_init = 4 * 3

simulation_ct = Simulation.from_datawrapper(datawrapper=can_disagg_datawrapper, simulation_configuration=configuration)

simulation_ct.run()

production_ct = get_production_dataframe(simulation_ct)

# %%

configuration.country_configurations["CAN"].use_carbon_price = False
configuration.country_configurations["CAN"].use_carbon_price = True
configuration.country_configurations["CAN"].asymptotic_carbon_price = 0.0
configuration.country_configurations["CAN"].carbon_price_time_constant = 10.0
configuration.country_configurations["CAN"].carbon_price_t_init = 4 * 3
simulation_no_ct = Simulation.from_datawrapper(
    datawrapper=can_disagg_datawrapper, simulation_configuration=configuration
)

simulation_no_ct.run()

production_no_ct = get_production_dataframe(simulation_no_ct)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
production_ct["B05a"].plot(ax=ax)
production_no_ct["B05a"].plot(ax=ax)
ax.legend(["CT", "No CT"])
plt.show()

# %%
index = pd.date_range(start="2014-01-01", periods=len(production_ct), freq="QS", name="GDP (no carbon price)")
production_ct.index = index
production_no_ct.index = index


# %%
production_ct.to_csv("production_ct.csv")
production_no_ct.to_csv("production_no_ct.csv")


# %%


def plot_sector(sector_name: str):
    fig, ax = plt.subplots()
    production_ct[sector_name].plot(ax=ax)
    production_no_ct[sector_name].plot(ax=ax)
    ax.legend(["CT", "No CT"])
    plt.show()


# %%
plot_sector("B05a")

# %%
plot_sector("B05b")
# %%
plot_sector("B05c")
# %%
plot_sector("D01a")
# %%
plot_sector("D01b")
# %%
plot_sector("D01c")
# %%
plot_sector("D01d")
# %%
