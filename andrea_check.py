from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

data = DataWrapper.init_from_pickle("./data.pkl")


with open("/Users/jmoran/Projects/macrocosm/macromodel/runs/configurations/default_country_conf.yaml", "r") as f:
    country_conf_dict = yaml.safe_load(f)

country_conf = CountryConfiguration(**country_conf_dict)


country_configurations = {
    "FRA": country_conf,
}

configuration = SimulationConfiguration(country_configurations=country_configurations, t_max=15)

# dump the configuration
with open("configuration.yaml", "w") as f:
    yaml.dump(configuration.model_dump(), f)

# load the configuration
with open("configuration.yaml", "r") as f:
    configuration_dict = yaml.safe_load(f)

configuration = SimulationConfiguration(**configuration_dict)

# instantiate the model
# model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)
