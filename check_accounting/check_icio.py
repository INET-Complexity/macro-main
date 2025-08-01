# %%
from macro_data.configuration.countries import Country
from macro_data.readers import AGGREGATED_INDUSTRIES, ALL_INDUSTRIES
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.exogenous_data import (
    ExogenousCountryData,
    create_all_exogenous_data,
)
from macro_data.readers.io_tables.icio_reader import ICIOReader
from macro_data.readers.util.industry_extraction import (
    compile_exogenous_industry_data,
    compile_industry_data,
)

DATA_PATH = "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data"


def get_icio_readers() -> ICIOReader:
    france = Country("FRA")
    readers = DataReaders.from_raw_data(
        raw_data_path=DATA_PATH,
        country_names=[Country("FRA")],
        simulation_year=2014,
        scale_dict={france: 100000},
        industries=AGGREGATED_INDUSTRIES,
        force_single_hfcs_survey=True,
        single_icio_survey=True,
    )
    return readers.icio[2014]


# %%

icio_reader = get_icio_readers()

# %%

icio_reader.iot.loc[["FRA", "ROW"]]
