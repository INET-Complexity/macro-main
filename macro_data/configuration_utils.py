"""
This module provides utility functions for creating and managing country configurations
in the macroeconomic model. It handles the creation of configuration objects for both
EU and non-EU countries, with support for proxy country relationships and scaling factors.

The module provides three main functions:
- read_country_conf: Reads default country configuration from YAML
- create_country_configurations: Creates configurations for multiple countries
- default_data_configuration: Creates a complete data configuration with defaults

Key features:
- Support for both EU and non-EU countries
- Proxy country mechanism for non-EU countries
- Flexible scaling of synthetic agents
- Industry aggregation options
- Configuration validation

Example:
    ```python
    from macro_data.configuration_utils import default_data_configuration

    # Create configuration for France
    fra_config = default_data_configuration(
        countries=["FRA"],
        year=2023,
        aggregate_industries=False
    )

    # Create configuration for USA using France as proxy
    multi_config = default_data_configuration(
        countries=["FRA", "USA"],
        proxy_country_dict={"USA": "FRA"},
        scale={"FRA": 10000, "USA": 20000}
    )
    ```
"""

from pathlib import Path
from typing import Optional

import yaml

from macro_data import DataConfiguration
from macro_data.configuration import CountryDataConfiguration
from macro_data.configuration.countries import Country

COUNTRY_CONF_PATH = Path(__file__).parent / "default_country_conf.yaml"


def read_country_conf() -> CountryDataConfiguration:
    """
    Read the default country configuration from YAML file.

    This function reads the default configuration settings from a YAML file
    and creates a CountryDataConfiguration object. The configuration includes
    settings for:
    - Firms (production, pricing, inventory)
    - Banks (lending, interest rates)
    - Central bank (monetary policy)
    - Government entities

    The function also sets single_firm_per_industry to True by default.

    Returns:
        CountryDataConfiguration: Default configuration for a country

    Raises:
        FileNotFoundError: If default_country_conf.yaml is not found
        yaml.YAMLError: If YAML file is malformed
    """
    with open(COUNTRY_CONF_PATH, "r") as file:
        country_conf_dict = yaml.safe_load(file)
        country_conf = CountryDataConfiguration(**country_conf_dict)
        country_conf.single_firm_per_industry = True
        return country_conf


def create_country_configurations(
    countries: list[str | Country],
    scale: dict[str | Country, int] | int,
    proxy_country_dict: Optional[dict[str | Country, Country | str]] = None,
    use_compustat: bool = False,
) -> dict[Country, CountryDataConfiguration]:
    """
    Create a dictionary of country configurations with appropriate settings and relationships.

    This function creates configuration objects for multiple countries, handling both EU
    and non-EU countries. For non-EU countries, it requires a proxy EU country to be
    specified. The function supports flexible scaling of synthetic agents either through
    a uniform scale factor or country-specific scaling.

    Args:
        countries (list[str | Country]): List of countries to configure. Can be either
            string country codes or Country enum values.
        scale (dict[str | Country, int] | int): Scale factor for synthetic agents.
            If an integer, applies the same scale to all countries.
            If a dictionary, specifies country-specific scales.
        proxy_country_dict (Optional[dict[str | Country, Country | str]]): Maps non-EU
            countries to their EU proxy countries. Required if any non-EU country is
            included in the countries list.
        use_compustat (bool): Whether to use Compustat data for firms and banks.
            If False, uses default constructors.

    Returns:
        dict[Country, CountryDataConfiguration]: Dictionary mapping countries to their
            configuration objects.

    Raises:
        ValueError: If a non-EU country is included without a proxy, or if a non-EU
            country is specified as a proxy.

    Example:
        ```python
        # Single EU country
        configs = create_country_configurations(
            countries=["FRA"],
            scale=10000
        )

        # Multiple countries with proxy
        configs = create_country_configurations(
            countries=["FRA", "USA"],
            scale={"FRA": 10000, "USA": 20000},
            proxy_country_dict={"USA": "FRA"}
        )
        ```
    """
    country_configs: dict[Country, CountryDataConfiguration] = {}

    countries = [Country(country) if isinstance(country, str) else country for country in countries]

    if proxy_country_dict is None:
        # if there is a non-EU country, raise an error
        if any(not country.is_eu_country for country in countries):
            raise ValueError("Non-EU countries must have a proxy country. Please provide a proxy country dictionary.")

    if isinstance(scale, int):
        scale = {country: scale for country in countries}
    for country in countries:
        if country.is_eu_country:
            country_configs[country] = read_country_conf().model_copy(update={"scale": scale[country]})
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
            country_configs[country] = read_country_conf().model_copy(
                update={"eu_proxy_country": proxy_country, "scale": scale[country]}
            )
        if not use_compustat:
            country_configs[country].firms_configuration.constructor = "Default"
            country_configs[country].banks_configuration.constructor = "Default"
    return country_configs


def default_data_configuration(
    countries: list[str | Country],
    proxy_country_dict: Optional[dict[str | Country, Country | str]] = None,
    year: int = 2014,
    aggregate_industries: bool = True,
    single_firm_per_industry: bool = True,
    scale: dict[str | Country, int] | int = 10_000,
    seed: Optional[int] = None,
    use_disagg_can_2014_reader: bool = False,
) -> DataConfiguration:
    """
    Create a complete data configuration with sensible defaults for model initialization.

    This function serves as the primary entry point for creating model configurations.
    It combines country-specific configurations with global settings to create a
    comprehensive configuration object suitable for initializing the economic model.

    The function supports both EU and non-EU countries, with special handling for
    Canada's energy sector disaggregation. It provides options for industry
    aggregation, firm structure, and synthetic agent scaling.

    Args:
        countries (list[str | Country]): List of countries to include in the model.
            Can be either string country codes or Country enum values.
        proxy_country_dict (Optional[dict[str | Country, Country | str]]): Maps non-EU
            countries to their EU proxy countries. Required if any non-EU country is
            included.
        year (int): Base year for data and model initialization. Defaults to 2014.
        aggregate_industries (bool): Whether to use aggregated industry categories.
            If True, uses broader industry classifications.
            If False, uses detailed industry breakdowns.
        single_firm_per_industry (bool): Whether to use one representative firm per
            industry. Simplifies model but reduces heterogeneity.
        scale (dict[str | Country, int] | int): Scale factor for synthetic agents.
            If an integer, applies the same scale to all countries.
            If a dictionary, specifies country-specific scales.
            Defaults to 10,000 agents per synthetic agent.
        seed (Optional[int]): Random seed for reproducibility. If None, uses
            system time.
        use_disagg_can_2014_reader (bool): Whether to use Canada's disaggregated
            energy sector reader. Only applicable when modeling Canada alone.

    Returns:
        DataConfiguration: Complete configuration object ready for model initialization.

    Raises:
        ValueError: If attempting to use Canada's disaggregated reader with other
            countries, or if non-EU countries lack proxy specifications.

    Example:
        ```python
        # Simple single-country configuration
        config = default_data_configuration(
            countries=["FRA"],
            year=2023,
            aggregate_industries=False
        )

        # Multi-country configuration with proxy
        config = default_data_configuration(
            countries=["FRA", "USA", "CAN"],
            proxy_country_dict={
                "USA": "FRA",
                "CAN": "FRA"
            },
            scale={
                "FRA": 10000,
                "USA": 20000,
                "CAN": 15000
            },
            aggregate_industries=True
        )

        # Canada with disaggregated energy sector
        config = default_data_configuration(
            countries=["CAN"],
            use_disagg_can_2014_reader=True
        )
        ```
    """
    # if we use the disaggregated reader for Canada, we can only have CAN in the list of countries, and
    # it can't be empty
    if use_disagg_can_2014_reader:
        if countries != ["CAN"]:
            raise ValueError("If using the disaggregated reader for Canada, only CAN can be in the list of countries.")

    country_configurations = create_country_configurations(countries, scale, proxy_country_dict)
    return DataConfiguration(
        year=year,
        country_configs=country_configurations,
        aggregate_industries=aggregate_industries,
        single_firm_per_industry=single_firm_per_industry,
        seed=seed,
        can_disaggregation=use_disagg_can_2014_reader,
    )
