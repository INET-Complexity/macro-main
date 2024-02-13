from enum import StrEnum
from pathlib import Path
import yaml


# get this file's directory path
THIS_FILE_PATH = Path(__file__).parent.resolve()

with open(THIS_FILE_PATH / "3_codes.yaml", "r") as f:
    country_codes = yaml.safe_load(f)

with open(THIS_FILE_PATH / "country_names.yaml", "r") as f:
    country_names = yaml.safe_load(f)

EU_COUNTRIES = [
    "AUT",
    "BEL",
    "CZE",
    "DNK",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LUX",
    "NLD",
    "POL",
    "PRT",
    "SVK",
    "ESP",
    "SWE",
    "EST",
    "LVA",
    "SVN",
    "LTU",
    "HRV",
    "CYP",
    "MLT",
    "ROU",
    "BGR",
]


class Country(StrEnum):
    """
    Represents a country with its corresponding code.
    """

    FRANCE = "FRA"
    GERMANY = "DEU"
    ITALY = "ITA"
    UNITED_KINGDOM = "GBR"
    AUSTRIA = "AUT"

    UNITED_STATES = "USA"
    CANADA = "CAN"
    JAPAN = "JPN"

    REST_OF_WORLD = "ROW"

    def __str__(self):
        return country_names[self.value]

    def to_two_letter_code(self):
        return country_codes[self.value]

    @property
    def is_eu_country(self):
        return self.value in EU_COUNTRIES
