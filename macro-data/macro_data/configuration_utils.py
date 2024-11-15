from pathlib import Path
from typing import Optional

import yaml

from macro_data import DataConfiguration
from macro_data.configuration import CountryDataConfiguration
from macro_data.configuration.countries import Country

COUNTRY_CONF_PATH = Path(__file__).parent / "default_country_conf.yaml"


def read_country_conf() -> CountryDataConfiguration:
    with open(COUNTRY_CONF_PATH, "r") as file:
        country_conf_dict = yaml.safe_load(file)
        country_conf = CountryDataConfiguration(**country_conf_dict)
        country_conf.single_firm_per_industry = True
        return country_conf


def create_country_configurations(
    countries: list[str | Country],
    scale: dict[str | Country, int] | int,
    proxy_country_dict: Optional[dict[str | Country, Country | str]] = None,
) -> dict[Country, CountryDataConfiguration]:
    """
    Create a dictionary of country configurations.

    Args:
        countries (list[str | Country]): List of countries.
        proxy_country_dict (dict[str | Country, Country | str]): Dictionary of proxy countries.
        scale (dict[str | Country, int] | int): scale factor. If int, will be applied identically to all countries.
                                                Otherwise, a dictionary of scales for each country.

    Returns:
        dict[Country, CountryDataConfiguration]: Dictionary of country configurations.
    """
    country_configs = {}

    countries = [Country(country) if isinstance(country, str) else country for country in countries]

    if proxy_country_dict is None:
        # if there is a non-EU country, raise an error
        if any(not country.is_eu_country for country in countries):
            raise ValueError("Non-EU countries must have a proxy country. Please provide a proxy country dictionary.")

    if isinstance(scale, int):
        scale = {country: scale for country in countries}
    for country in countries:
        if country.is_eu_country:
            country_configs[country] = read_country_conf().copy(update={"scale": scale[country]})
        else:
            proxy_country = proxy_country_dict.get(country, None)
            if proxy_country is None:
                raise ValueError(f"{country} is not in EU: please set an EU proxy country.")
            if isinstance(proxy_country, str):
                proxy_country = Country(proxy_country)
            if not proxy_country.is_eu_country:
                raise ValueError(
                    f"{proxy_country} is not in EU, but was set as a proxy country for {country}."
                    f"Please set an EU country as a proxy country."
                )
            country_configs[country] = read_country_conf().copy(
                update={"eu_proxy_country": proxy_country, "scale": scale[country]}
            )
    return country_configs


def default_data_configuration(
    countries: list[str | Country],
    proxy_country_dict: Optional[dict[str | Country, Country | str]] = None,
    year: int = 2014,
    aggregate_industries: bool = True,
    single_firm_per_industry: bool = True,
    scale: dict[str | Country, int] | int = 10_000,
    seed: Optional[int] = None,
) -> DataConfiguration:
    """
    Create a default data configuration.

    Args:
        countries (list[str | Country]): List of countries.
        proxy_country_dict (dict[str | Country, Country | str]): Dictionary of proxy countries.
        year (int): Initial year.
        aggregate_industries (bool): Whether to aggregate industries.
        single_firm_per_industry (bool): Whether to have a single firm per industry.
        scale (dict[str | Country, int] | int): Scale factor.
        seed (Optional[int]): Seed value.

    Returns:
        DataConfiguration: The default data configuration.
    """
    country_configurations = create_country_configurations(countries, scale, proxy_country_dict)
    return DataConfiguration(
        year=year,
        country_configs=country_configurations,
        aggregate_industries=aggregate_industries,
        single_firm_per_industry=single_firm_per_industry,
        seed=seed,
    )
