import pandas as pd

from inet_data.readers.default_readers import DataReaders
from inet_data.readers.util.industry_extraction import compile_exogenous_industry_data


def create_all_exogenous_data(
    readers: DataReaders,
    country_names: list[str],
    year_min: int = 2010,
    year_max: int = 2019,
) -> dict[str, dict[str, pd.DataFrame]]:
    exogenous_industry_data = compile_exogenous_industry_data(readers, country_names, year_min, year_max)

    # get the set intersection of country_names and the keys of exogenous_industry_data
    exog_countries = list(set(country_names).intersection(exogenous_industry_data.keys()))
    exogenous_data = {
        country_name: {
            "log_inflation": readers.world_bank.get_log_inflation(country_name),
            "sectoral_growth": readers.eurostat.get_perc_sectoral_growth(country_name),
            "unemployment_rate": readers.oecd_econ.get_unemployment_rate(country_name),
            "house_price_index": readers.oecd_econ.get_house_price_index(country_name),
            "vacancy_rate": readers.oecd_econ.get_vacancy_rate(country_name),
            "total_firm_deposits_and_debt": readers.eurostat.get_total_industry_debt_and_deposits(country_name),
            "iot_industry_data": exogenous_industry_data[country_name],
        }
        for country_name in exog_countries
    }

    return exogenous_data
