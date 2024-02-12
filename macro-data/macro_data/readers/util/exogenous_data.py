import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.util.industry_extraction import compile_exogenous_industry_data


def create_all_exogenous_data(
    readers: DataReaders,
    country_names: list[Country],
    year_min: int = 2010,
    year_max: int = 2019,
    proxy_countries: dict[Country, Country] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Create exogenous data for each country in the given list of country names.

    This data includes:
        - log inflation
        - sectoral growth
        - unemployment rate
        - house price index
        - vacancy rate
        - total firm deposits and debt
        - industry data from the input-output tables

    Args:
        readers (DataReaders): An instance of the DataReaders class that provides access to various data sources.
        country_names (list[str]): A list of country names for which exogenous data needs to be created.
        year_min (int, optional): The minimum year for which exogenous data should be collected. Defaults to 2010.
        year_max (int, optional): The maximum year for which exogenous data should be collected. Defaults to 2019.
        proxy_countries (dict[str, str], optional): A dictionary of country names and their corresponding proxy EU
            countries. Defaults to None.

    Returns:
        dict[str, dict[str, pd.DataFrame]]: A dictionary containing exogenous data for each country, organized by country name and data type.
    """
    exogenous_industry_data = compile_exogenous_industry_data(readers, country_names, year_min, year_max)

    if proxy_countries is None:
        proxy_countries = {}

    for country in country_names:
        if country not in proxy_countries:
            proxy_countries[country] = country

    # get the set intersection of country_names and the keys of exogenous_industry_data
    exog_countries = list(set(country_names).intersection(exogenous_industry_data.keys()))
    # TODO this is a hack; sectoral growth and firm deposits and debt need to be readjusted
    exogenous_data = {
        country_name: {
            "log_inflation": readers.world_bank.get_log_inflation(country_name),
            "sectoral_growth": readers.eurostat.get_perc_sectoral_growth(proxy_countries[country_name]),
            "unemployment_rate": readers.oecd_econ.get_unemployment_rate(country_name),
            "house_price_index": readers.oecd_econ.get_house_price_index(country_name),
            "vacancy_rate": readers.oecd_econ.get_vacancy_rate(country_name),
            "total_firm_deposits_and_debt": readers.eurostat.get_total_industry_debt_and_deposits(
                proxy_countries[country_name]
            ),
            "iot_industry_data": exogenous_industry_data[country_name],
        }
        for country_name in exog_countries
    }

    return exogenous_data
