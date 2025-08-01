import numpy as np
import pytest
import yaml

from macro_data import DataWrapper, SyntheticCountry
from macro_data.configuration import DataConfiguration
from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers import AGGREGATED_INDUSTRIES, ALL_INDUSTRIES
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.exogenous_data import (
    ExogenousCountryData,
    create_all_exogenous_data,
)
from macro_data.readers.util.industry_extraction import (
    compile_exogenous_industry_data,
    compile_industry_data,
)

DATA_PATH = "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data"
DATA_CONFIG = (
    "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/default_data_config.yaml"
)

canada = Country("CAN")
france = Country("FRA")

reader = DataReaders.from_raw_data(
    raw_data_path=DATA_PATH,
    country_names=[canada],
    simulation_year=2014,
    scale_dict={canada: 100000},
    industries=ALL_INDUSTRIES,
    force_single_hfcs_survey=True,
    single_icio_survey=True,
    aggregate_industries=False,
    proxy_country_dict={canada: france},
    use_provincial_can_reader=False,
    override_icio_filename="can_all_disagg.csv",
)


with open(DATA_CONFIG, "r") as f:
    config_dict = yaml.safe_load(f)
# not necessary to do the country splitting here
# since the fixture used only has one country key
configuration = DataConfiguration(**config_dict)
configuration.can_disaggregation = False
configuration.aggregate_industries = False

canada = Country("CAN")
france = Country("FRA")

configuration.country_configs[france].single_firm_per_industry = True
configuration.country_configs[france].single_bank = True
configuration.country_configs[france].single_government_entity = True

configuration.country_configs[canada] = configuration.country_configs[france]

configuration.country_configs[canada].eu_proxy_country = france

del configuration.country_configs[france]


# Check if there is a file in raw data path
creator = DataWrapper.from_config(
    configuration=configuration,
    raw_data_path=DATA_PATH,
    single_hfcs_survey=True,
    override_icio_filename="can_all_disagg.csv",
)

creator.save("canada_all_disagg.pkl")
