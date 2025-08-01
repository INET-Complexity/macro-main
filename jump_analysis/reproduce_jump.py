# %%
import pickle as pkl

import yaml

from macro_data import DataWrapper
from macromodel.configurations import SimulationConfiguration
from macromodel.simulation import Simulation

# %%

with open("data.pkl", "rb") as f:
    data_wrapper = pkl.load(f)


data = DataWrapper.init_from_pickle("data.pkl")

# %%

with open("configuration.yaml", "r") as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)


configuration = SimulationConfiguration(**configuration)

configuration.t_max = 5
# %%

model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)
# %%

model.run()

# %%

gdp = model.countries["FRA"].economy.gdp_output()


# %%


def get_gdp(configuration: SimulationConfiguration):
    model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)
    model.run()
    return model.countries["FRA"].economy.gdp_output()


# %%
gdps = [get_gdp(configuration) for _ in range(10)]

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for gdp in gdps:
    ax.plot(gdp)

ax.set_xlabel("Time")
ax.set_ylabel("GDP")
ax.set_title("GDP Over Time")


# %%
