import pathlib


import pytest
import yaml

from macro_data.configuration import DataConfiguration
from macro_data.configuration.countries import Country
from macro_data.readers import AGGREGATED_INDUSTRIES, ALL_INDUSTRIES
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.exogenous_data import create_all_exogenous_data, ExogenousCountryData
from macro_data.readers.util.industry_extraction import compile_industry_data, compile_exogenous_industry_data

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
