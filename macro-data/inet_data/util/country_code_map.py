import os
from pathlib import Path

import pandas as pd


# TODO This should not be hardcoded
def get_map_long_to_short(data_path: Path) -> dict[str, str]:
    p = data_path / "notation" / "wikipedia-iso-country-codes.csv"
    if not os.path.isfile(p):
        raise FileNotFoundError("Failed to find the country ISO code file.")
    iso_data = pd.read_csv(p)
    return dict(zip(iso_data["Alpha-3 code"], iso_data["Alpha-2 code"]))


# TODO same
def get_map_name_to_short(data_path: Path) -> dict[str, str]:
    p = data_path / "notation" / "wikipedia-iso-country-codes.csv"
    if not os.path.isfile(p):
        raise FileNotFoundError("Failed to find the country ISO code file.")
    iso_data = pd.read_csv(p)
    return dict(zip(iso_data["Alpha-3 code"], iso_data["English short name lower case"]))
