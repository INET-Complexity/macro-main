"""
WARNING: This module references a deprecated configuration format.

This module provides utilities for processing and manipulating model configuration data.
It handles the parsing and transformation of configuration files, with special support
for country-specific configurations and interest rate settings.

The module provides three main functions:
- split_country_configs: Splits combined country configurations into individual ones
- process_config: Processes a configuration file or dictionary, handling country splits
- initial_interest_rates: Extracts initial interest rates from configuration for a country

Example:
    ```python
    from macro_data.configuration.process_config import process_config, initial_interest_rates

    # Load and process configuration
    config = process_config("path/to/config.yaml")

    # Get interest rates for a specific country
    france_rates = initial_interest_rates(config, "FRA")
    ```
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def split_country_configs(country_config: dict) -> dict[str, Any]:
    """
    Split combined country configurations into individual country configurations.

    This function takes a configuration dictionary that may contain combined country
    keys (e.g., "FRA&DEU") and splits them into individual country entries while
    maintaining the same configuration values.

    Args:
        country_config (dict): Configuration dictionary with potentially combined country keys

    Returns:
        dict[str, Any]: Configuration dictionary with split country entries

    Example:
        ```python
        config = {"FRA&DEU": {"param": 1}}
        split = split_country_configs(config)
        # Result: {"FRA": {"param": 1}, "DEU": {"param": 1}}
        ```
    """
    new_config = {}
    for key, value in country_config.items():
        # Split the key by '&' and assign the same value to each country code
        countries = key.split("&")
        for country in countries:
            new_config[country] = value
    return new_config


def process_config(config_path: str | Path | dict) -> dict[str, Any]:
    """
    Process a configuration file or dictionary, handling country splits and initialization.

    This function reads a configuration from a YAML file or dictionary and processes it by:
    1. Splitting combined country configurations (e.g., "FRA&DEU")
    2. Validating country codes against the model's supported countries
    3. Creating a clean configuration structure with model and initialization data

    Args:
        config_path (str | Path | dict): Path to configuration file or configuration dictionary

    Returns:
        dict[str, Any]: Processed configuration dictionary with structure:
            {
                "model": {...},  # Model-wide settings
                "init": {...}   # Country-specific initialization data
            }

    Example:
        ```python
        # From file
        config = process_config("config.yaml")

        # From dictionary
        config = process_config({
            "model": {"country_names": {"value": ["FRA", "DEU"]}},
            "init": {"FRA&DEU": {"param": 1}}
        })
        ```
    """
    # if path is str make it a Path
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if isinstance(config_path, Path):
        config = yaml.safe_load(open(config_path, "r"))
    else:
        config = config_path
    config = {
        "model": config["model"].copy(),
        "init": config["init"].copy(),
    }

    # Handle countries
    keys = list(config["init"].keys())
    for key in keys:
        if "&" in key:
            for c in key.split("&"):
                if c in config["model"]["country_names"]["value"]:
                    config["init"][c] = deepcopy(config["init"][key])
    for key in keys:
        if "&" in key:
            del config["init"][key]

    return config


def initial_interest_rates(config: dict[str, Any], country: str) -> dict[str, float]:
    """
    Extract initial interest rates for different loan types for a specific country.

    This function retrieves the configured initial interest rate markups for various
    types of loans (consumption loans, mortgages, overdrafts) for a given country
    from the configuration.

    Args:
        config (dict[str, Any]): The processed configuration dictionary
        country (str): The country code to get interest rates for

    Returns:
        dict[str, float]: Dictionary containing initial interest rate markups:
            {
                "bank_markup_interest_rate_household_consumption_loans": float,
                "bank_markup_interest_rate_mortgages": float,
                "bank_markup_interest_rate_overdraft_household": float
            }

    Example:
        ```python
        config = process_config("config.yaml")
        france_rates = initial_interest_rates(config, "FRA")
        mortgage_markup = france_rates["bank_markup_interest_rate_mortgages"]
        ```
    """
    banks_dict = config["init"][country]["banks"]["parameters"]
    bank_markup_interest_rate_household_consumption_loans = banks_dict[
        "initial_markup_interest_rate_household_consumption_loans"
    ]["value"]

    bank_markup_interest_rate_mortgages = banks_dict["initial_markup_mortgage_interest_rate"]["value"]
    bank_markup_interest_rate_overdraft_household = banks_dict["initial_markup_interest_rate_overdraft_households"][
        "value"
    ]

    return {
        "bank_markup_interest_rate_household_consumption_loans": bank_markup_interest_rate_household_consumption_loans,
        "bank_markup_interest_rate_mortgages": bank_markup_interest_rate_mortgages,
        "bank_markup_interest_rate_overdraft_household": bank_markup_interest_rate_overdraft_household,
    }
