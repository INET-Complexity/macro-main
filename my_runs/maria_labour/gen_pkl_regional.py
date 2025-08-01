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

configuration = DataConfiguration(**config_dict)
configuration.can_disaggregation = False
configuration.aggregate_industries = False
configuration.prune_date = None
configuration.seed = 0

# Get the base configuration (France's config) to copy for all regions
france = Country("FRA")
base_config = configuration.country_configs[france]

base_config.single_firm_per_industry = True
base_config.single_bank = True
base_config.single_government_entity = True

base_config.firms_configuration.constructor = "Default"

base_config.scale = 1000

# Define Canadian provinces
provinces = [
    Region.from_code("CAN_AB", "Alberta"),
    Region.from_code("CAN_BC", "British Columbia"),
    Region.from_code("CAN_MB", "Manitoba"),
    Region.from_code("CAN_NB", "New Brunswick"),
    Region.from_code("CAN_NL", "Newfoundland and Labrador"),
    Region.from_code("CAN_NS", "Nova Scotia"),
    Region.from_code("CAN_ON", "Ontario"),
    Region.from_code("CAN_PE", "Prince Edward Island"),
    Region.from_code("CAN_QC", "Quebec"),
    Region.from_code("CAN_SK", "Saskatchewan"),
]

# Add Canada as the parent country
canada = Country("CAN")
configuration.country_configs[canada] = base_config
configuration.country_configs[canada].eu_proxy_country = france

# Add configurations for all provinces
for province in provinces:
    configuration.country_configs[province] = base_config
    configuration.country_configs[province].eu_proxy_country = france

# Set up the aggregation structure
configuration.aggregation_structure = {canada: provinces}

# Remove France's config since we don't need it for this test
del configuration.country_configs[france]


# Check if there is a file in raw data path
creator = DataWrapper.from_config(
    configuration=configuration,
    raw_data_path=DATA_PATH,
    single_hfcs_survey=True,
    override_icio_filename="can_regional_energy_disagg.csv",
)

creator.save("canada_regional_disagg.pkl")
