from enum import StrEnum
import yaml

with open("3_codes.yaml", "r") as f:
    country_codes = yaml.safe_load(f)

with open("country_names.yaml", "r") as f:
    country_names = yaml.safe_load(f)


class Country(StrEnum):
    """
    Represents a country with its corresponding code.
    """

    FRANCE = "FRA"
    GERMANY = "DEU"
    ITALY = "ITA"
    UNITED_KINGDOM = "GBR"
    AUSTRIA = "AUT"

    def __str__(self):
        return country_names[self.value]

    def to_two_letter_code(self):
        return country_codes[self.value]
