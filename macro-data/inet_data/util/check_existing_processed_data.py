import os
import ast
import pandas as pd

from pathlib import Path
from copy import deepcopy

from typing import Optional


def check_existing_processed_data(config: dict, data_path: Path) -> Optional[str]:
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
        if not os.path.exists(curr_path / str(filename) / "inet_data.h5"):
            continue

        # Check model config
        config_model = pd.DataFrame(pd.read_hdf(curr_path / str(filename) / "inet_data.h5", "config_model")).values[0][
            0
        ]
        config_model = ast.literal_eval(config_model)
        for key in config_model:
            if key in config["model"].keys():
                if str(config_model[key]["value"]) != str(config["model"][key]["value"]):
                    is_matching = False
            else:
                is_matching = False
                break

        # Check model init
        config_init = pd.DataFrame(pd.read_hdf(curr_path / str(filename) / "inet_data.h5", "config_init")).values[0][0]
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
