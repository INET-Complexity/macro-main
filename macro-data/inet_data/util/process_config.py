from typing import Any, Union
from pathlib import Path
import yaml
from copy import deepcopy


def process_config(config_path: str | Path | dict) -> dict[str, Any]:
    """
    Process the configuration file (yaml) or dictionary.
    Read and separate country pairs as "FRA&DEU" into "FRA" and "DEU" in the init section of the config file.

    Args:
        config_path (Union[Path, dict]): The path to the configuration file or the configuration dictionary.

    Returns:
        dict[str, Any]: The processed configuration dictionary.
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
    Calculate the initial interest rates for different types of loans based on the given configuration and country. (This is just a
    wrapper for configuration data).

    Args:
        config (dict[str, Any]): The configuration dictionary.
        country (str): The country for which the interest rates are calculated.

    Returns:
        dict[str, float]: A dictionary containing the initial interest rates for different types of loans.
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
