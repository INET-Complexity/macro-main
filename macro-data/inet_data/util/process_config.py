from typing import Any, Union
from pathlib import Path
import yaml
from copy import deepcopy


def process_config(config_path: Path | dict) -> dict[str, Any]:
    """
    Process the configuration file (yaml) or dictionary.
    Read and separate country pairs as "FRA&DEU" into "FRA" and "DEU" in the init section of the config file.

    Args:
        config_path (Union[Path, dict]): The path to the configuration file or the configuration dictionary.

    Returns:
        dict[str, Any]: The processed configuration dictionary.
    """
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
