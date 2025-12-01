import pathlib

import pytest
import yaml

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

PARENT = pathlib.Path(__file__).parent.resolve()
DATA_PATH = PARENT / "sample_raw_data"


@pytest.fixture(scope="module", name="data_path")
def data_path():
    return DATA_PATH


@pytest.fixture(scope="module", name="configuration")
def configuration():
    data_config_path = PARENT / "default_data_config.yaml"
    with open(data_config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
    configuration = DataConfiguration(**config_dict)
    return configuration


@pytest.fixture(scope="module", name="data_config_path")
def configuration2():
    return PARENT / "default_data_config.yaml"


@pytest.fixture(scope="module", name="gbr_data_config_path")
def gbr_data_config_path():
    return PARENT / "gbr_data_config.yaml"


@pytest.fixture(scope="module", name="gen_data_config_path")
def configuration3():
    return PARENT / "data_config_gen.yaml"


@pytest.fixture(scope="module", name="readers")
def readers(data_path):
    france = Country("FRA")
    readers = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=[Country("FRA")],
        simulation_year=2014,
        # need to put in Afghanistan because that is used in tests...
        scale_dict={france: 100000, "AFG": 100000},
        industries=AGGREGATED_INDUSTRIES,
        force_single_hfcs_survey=True,
        single_icio_survey=True,
    )
    return readers


@pytest.fixture(scope="module", name="readers_disagg_can")
def readers_disagg_can(data_path):
    canada = Country("CAN")
    france = Country("FRA")
    reader = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=[Country("CAN")],
        simulation_year=2014,
        scale_dict={canada: 100000},
        industries=ALL_INDUSTRIES,
        force_single_hfcs_survey=True,
        single_icio_survey=True,
        aggregate_industries=False,
        proxy_country_dict={canada: france},
        use_disagg_can_2014_reader=True,
    )

    return reader


@pytest.fixture(scope="module", name="readers_provincial_can")
def readers_provincial_can(data_path):
    canada = Country("CAN")
    france = Country("FRA")

    regions_list = [
        "CAN_AB",
        "CAN_BC",
        "CAN_MB",
        "CAN_NB",
        "CAN_NL",
        "CAN_NS",
        "CAN_ON",
        "CAN_PE",
        "CAN_QC",
        "CAN_SK",
    ]

    regions = [Region.from_code(region) for region in regions_list]

    regions_dict = {canada: regions}

    reader = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=[Country("CAN")],
        simulation_year=2014,
        scale_dict={canada: 100000},
        industries=ALL_INDUSTRIES,
        force_single_hfcs_survey=True,
        single_icio_survey=True,
        aggregate_industries=False,
        proxy_country_dict={canada: france},
        use_provincial_can_reader=True,
        regions_dict=regions_dict,
    )

    return reader


@pytest.fixture(scope="module", name="all_readers")
def all_readers(data_path):
    france = Country("FRA")
    all_readers = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=[Country("FRA")],
        simulation_year=2014,
        scale_dict={france: 100000},
        industries=ALL_INDUSTRIES,
        force_single_hfcs_survey=True,
        single_icio_survey=True,
        aggregate_industries=False,
    )
    return all_readers


@pytest.fixture(scope="module", name="all_industries_readers")
def all_industries_readers(data_path):
    france = Country("FRA")
    all_industries_readers = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=[Country("FRA")],
        simulation_year=2014,
        scale_dict={france: 100000},
        industries=ALL_INDUSTRIES,
        force_single_hfcs_survey=True,
        single_icio_survey=True,
        aggregate_industries=False,
    )
    return all_industries_readers


@pytest.fixture(scope="module", name="multic_readers")
def gen_multic_readers(data_path):
    france = Country("FRA")
    multic_readers = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=[Country("FRA"), Country("USA"), Country("CAN")],
        simulation_year=2014,
        # need to put in Afghanistan because that is used in tests...
        scale_dict={france: 100000, "AFG": 100000},
        industries=[
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R_S",
        ],
        force_single_hfcs_survey=True,
        single_icio_survey=True,
        proxy_country_dict={Country("CAN"): Country("FRA"), Country("USA"): Country("FRA")},
    )
    return multic_readers


@pytest.fixture(scope="module", name="industry_data")
def industry_data(readers):
    return compile_industry_data(
        year=2014, readers=readers, country_names=[Country("FRA")], single_firm_per_industry={"FRA": True}
    )


@pytest.fixture(scope="module", name="exogenous_data")
def exogenous_data(readers, industry_data):
    return ExogenousCountryData.from_data_readers(
        country_name=Country("FRA"),
        readers=readers,
        year=2014,
        quarter=1,
        industry_vectors=industry_data["FRA"]["industry_vectors"],
    )


@pytest.fixture(scope="module", name="multic_industry_data")
def multic_industry_data(multic_readers):
    return compile_industry_data(
        year=2014,
        readers=multic_readers,
        country_names=[Country("FRA"), Country("USA"), Country("CAN")],
        single_firm_per_industry={"FRA": True, "USA": True, "CAN": True},
    )


@pytest.fixture(scope="module", name="exogenous_industry_data")
def exogenous_industry_data(readers):
    country_names = ["FRA"]
    exogenous_industry_data = compile_exogenous_industry_data(readers, country_names)
    return exogenous_industry_data


@pytest.fixture(scope="module", name="all_exogenous_data")
def all_exogenous_data(readers):
    country_names = [Country("FRA")]
    all_exogenous_data = create_all_exogenous_data(readers, country_names)
    return all_exogenous_data


@pytest.fixture
def canada_disagg_config(data_config_path):
    """Fixture for Canadian provincial disaggregation configuration."""
    with open(data_config_path, "r") as f:
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

    return configuration
