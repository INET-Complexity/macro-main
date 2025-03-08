"""Utility module for checking and validating processed model data.

WARNING: This module references a deprecated way of storing preprocessed data.

This module provides functionality to check if previously processed data exists that matches
a given model configuration. It helps avoid reprocessing data when identical configurations
have already been processed, improving efficiency in data preparation workflows.

Key Features:
    - Configuration validation and normalization
    - Deep comparison of model configurations
    - Handling of country-specific configurations
    - Validation of agent parameters and functions
    - HDF5 data file checking

The module is particularly useful in scenarios where model configurations are reused or
when verifying the existence of compatible processed data before initiating new processing tasks.
"""

import ast
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd


def check_existing_processed_data(config: dict, data_path: Path) -> Optional[str]:
    """Check if processed data matching the given configuration already exists.

    WARNING: This function references a deprecated way of storing preprocessed data.

    This function searches through existing processed data files to find a match for the
    provided configuration. It performs deep comparison of model parameters, initialization
    settings, and function configurations.

    Args:
        config (dict): The configuration dictionary containing model and initialization settings.
                      Must have 'model' and 'init' keys at the top level.
        data_path (Path): Path to the directory containing the processed_data subdirectory.

    Returns:
        Optional[str]: The filename of matching processed data if found, None otherwise.

    The function performs the following checks:
    1. Validates basic config structure (must have 'model' and 'init' keys)
    2. Normalizes country configurations by expanding '&' separated country names
    3. For each existing processed data file:
        - Verifies file existence and structure
        - Compares model configuration values
        - Validates country-specific initialization settings
        - Checks agent parameters and function configurations
        - Verifies function parameters match exactly

    Example:
        >>> config = {
        ...     'model': {'country_names': {'value': ['USA', 'EU']}},
        ...     'init': {'USA': {'households': {'parameters': {...}}}}
        ... }
        >>> data_path = Path('/path/to/data')
        >>> result = check_existing_processed_data(config, data_path)
        >>> if result:
        ...     print(f"Found matching data in {result}")
        ... else:
        ...     print("No matching data found")
    """
    # Handle the config
    if "model" not in config.keys() or "init" not in config.keys():
        return None
    keys = list(config["init"].keys())
    for key in keys:
        if "&" in key:
            for c in key.split("&"):
                if c in config["model"]["country_names"]["value"]:
                    config["init"][c] = deepcopy(config["init"][key])
    for key in keys:
        if "&" in key:
            del config["init"][key]

    # Check existing files
    curr_path = data_path / "processed_data"
    for filename in os.listdir(curr_path):
        is_matching = True

        # Check if the file exists
        if not os.path.exists(curr_path / str(filename) / "data.h5"):
            continue

        # Check model config
        config_model = pd.DataFrame(pd.read_hdf(curr_path / str(filename) / "data.h5", "config_model")).values[0][0]
        config_model = ast.literal_eval(config_model)
        for key in config_model:
            if key in config["model"].keys():
                if str(config_model[key]["value"]) != str(config["model"][key]["value"]):
                    is_matching = False
            else:
                is_matching = False
                break

        # Check model init
        config_init = pd.DataFrame(pd.read_hdf(curr_path / str(filename) / "data.h5", "config_init")).values[0][0]
        config_init = ast.literal_eval(config_init)
        for country_name in config_init.keys():
            if country_name not in config["init"].keys():
                is_matching = False
                break
            for agent in config_init[country_name].keys():
                if agent not in config["init"][country_name].keys():
                    is_matching = False
                    break

                # Parameters
                if "parameters" in config_init[country_name][agent].keys():
                    for parameter in config_init[country_name][agent]["parameters"].keys():
                        if parameter not in config["init"][country_name][agent]["parameters"].keys():
                            is_matching = False
                            break
                        if str(config_init[country_name][agent]["parameters"][parameter]["value"]) != str(
                            config["init"][country_name][agent]["parameters"][parameter]["value"]
                        ):
                            is_matching = False
                            break

                # Functions
                if "functions" in config_init[country_name][agent].keys():
                    for function in config_init[country_name][agent]["functions"].keys():
                        if function not in config["init"][country_name][agent]["functions"].keys():
                            is_matching = False
                            break

                        # Name
                        if "name" in config_init[country_name][agent]["functions"][function].keys():
                            if str(config_init[country_name][agent]["functions"][function]["name"]["value"]) != str(
                                config["init"][country_name][agent]["functions"][function]["name"]["value"]
                            ):
                                is_matching = False
                                break

                        # Parameters
                        for parameter in config_init[country_name][agent]["functions"][function]["parameters"].keys():
                            if (
                                parameter
                                not in config["init"][country_name][agent]["functions"][function]["parameters"].keys()
                            ):
                                is_matching = False
                                break
                            if str(
                                config_init[country_name][agent]["functions"][function]["parameters"][parameter][
                                    "value"
                                ]
                            ) != str(
                                config["init"][country_name][agent]["functions"][function]["parameters"][parameter][
                                    "value"
                                ]
                            ):
                                is_matching = False
                                break

        if is_matching:
            return str(filename)

    return None
