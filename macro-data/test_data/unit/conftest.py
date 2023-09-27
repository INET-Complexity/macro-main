import pytest
import pathlib

from data.readers.handle_readers import init_readers
from data.readers.util.matching_iot_with_sea import compile_industry_data

PARENT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_PATH = PARENT / "data" / "sample_raw_data"


@pytest.fixture(scope="module", name="data_path")
def data_path():
    return DATA_PATH


@pytest.fixture(scope="module", name="readers")
def readers(data_path):
    return init_readers(
        raw_data_path=data_path,
        country_names=["FRA"],
        country_names_short=["FR"],
        year=2014,
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
        testing=True,
    )


@pytest.fixture(scope="module", name="industry_data")
def industry_data(data_path, readers):
    return compile_industry_data(
        current_icio_reader=readers["icio"][2014],
        sea_reader=readers["wiod_sea"],
        econ_reader=readers["oecd_econ"],
        exchange_rates=readers["exchange_rates"],
        country_names=["FRA"],
        config={"model": {"single_firm_per_industry": {"value": True}}},
    )
