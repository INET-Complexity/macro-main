import pathlib

import pytest

from inet_data.readers.default_readers import DataReaders
from inet_data.readers.util.exogenous_data import create_all_exogenous_data
from inet_data.readers.util.industry_extraction import compile_industry_data, compile_exogenous_industry_data
from inet_data.configuration.process_config import process_config

PARENT = pathlib.Path(__file__).parent.resolve()
DATA_PATH = PARENT / "sample_raw_data"


@pytest.fixture(scope="module", name="data_path")
def data_path():
    return DATA_PATH


@pytest.fixture(scope="module", name="configuration")
def configuration():
    return process_config(config_path=PARENT / "default_unit_test.yaml")


@pytest.fixture(scope="module", name="data_config_path")
def configuration2():
    return PARENT / "default_data_config.yaml"


@pytest.fixture(scope="module", name="gen_data_config_path")
def configuration3():
    return PARENT / "data_config_gen.yaml"


@pytest.fixture(scope="module", name="readers")
def readers(data_path):
    readers = DataReaders.from_raw_data(
        raw_data_path=data_path,
        country_names=["FRA"],
        country_names_short=["FR"],
        simulation_year=2014,
        scale=100000,
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
    )
    return readers


@pytest.fixture(scope="module", name="industry_data")
def industry_data(readers):
    return compile_industry_data(year=2014, readers=readers, country_names=["FRA"], single_firm_per_industry=True)


@pytest.fixture(scope="module", name="exogenous_industry_data")
def exogenous_industry_data(readers):
    country_names = ["FRA"]
    exogenous_industry_data = compile_exogenous_industry_data(readers, country_names)
    return exogenous_industry_data


@pytest.fixture(scope="module", name="all_exogenous_data")
def all_exogenous_data(readers):
    country_names = ["FRA"]
    all_exogenous_data = create_all_exogenous_data(readers, country_names)
    return all_exogenous_data
