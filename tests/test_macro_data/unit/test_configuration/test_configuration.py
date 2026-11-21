import pytest
import yaml
from pydantic import ValidationError

from macro_data.configuration import DataConfiguration, split_country_configs


def test_read_config(gen_data_config_path):
    with open(gen_data_config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["country_configs"] = split_country_configs(config_dict["country_configs"])
    config_object = DataConfiguration(**config_dict)

    assert set(config_object.countries) == {"DEU", "FRA", "GBR"}

    # replace the key "GBR" with "XXX" in config_dict["country_configs"] dict
    config_dict["country_configs"]["XXX"] = config_dict["country_configs"]["GBR"]
    del config_dict["country_configs"]["GBR"]

    # check that creating the Configuration object with the modified config_dict raises a KeyError
    with pytest.raises(ValidationError):
        DataConfiguration(**config_dict)
