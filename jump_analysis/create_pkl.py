# %%
INPUT_PATH = "/Users/jmoran/Projects/macrocosm/inet/data/raw_data"
PKL_PATH = "./data.pkl"

import macro_data
from macro_data import configuration_utils

print(INPUT_PATH)
print(PKL_PATH)
# %%


def create_pickle(configuration, filename):
    creator = macro_data.DataWrapper.from_config(
        configuration=configuration, raw_data_path=INPUT_PATH, single_hfcs_survey=True
    )

    creator.save(filename)


representative_year: int = 2014
aggregate_industries = False
single_firm_per_industry = False
use_disagg_can_2014_reader = False
scale = 10000
seed = 1
# seed = None if seed == -1 else seed

data_configuration = configuration_utils.default_data_configuration(
    countries=["FRA"],
    year=representative_year,
    aggregate_industries=aggregate_industries,
    single_firm_per_industry=single_firm_per_industry,
    seed=seed,
    use_disagg_can_2014_reader=False,
)

# %%


create_pickle(data_configuration, PKL_PATH)

# %%
