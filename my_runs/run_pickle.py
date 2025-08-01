# %%
INPUT_PATH = "/Users/jmoran/Projects/macrocosm/inet/data/raw_data/"
PKL_PATH = "data.pkl"
import os

import numpy as np
import pandas as pd

import macro_data
import macromodel
from macro_data import configuration_utils
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

print(INPUT_PATH)
print(PKL_PATH)
# %%


def create_pickle(configuration, filename):
    creator = macro_data.DataWrapper.from_config(
        configuration=configuration, raw_data_path=INPUT_PATH, single_hfcs_survey=False
    )

    creator.save(filename)


representative_year: int = 2014
aggregate_industries = False
single_firm_per_industry = True
use_disagg_can_2014_reader = True
scale = 10000
seed = 1
# seed = None if seed == -1 else seed

data_configuration = configuration_utils.default_data_configuration(
    countries=["CAN"],
    proxy_country_dict={"CAN": "FRA"},
    year=representative_year,
    aggregate_industries=aggregate_industries,
    single_firm_per_industry=single_firm_per_industry,
    scale=scale,
    seed=seed,
    use_disagg_can_2014_reader=use_disagg_can_2014_reader,
)

# %%
create_pickle(data_configuration, PKL_PATH)

# %%
