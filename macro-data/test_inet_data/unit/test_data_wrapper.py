import yaml

from inet_data import DataWrapper
from pathlib import Path
import tempfile

from inet_data.configuration import DataConfiguration

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_file = tmp / "creator.pkl"
            creator.save(tmp_file)
            new_creator = DataWrapper.init_from_pickle(tmp_file)

        assert creator.synthetic_countries.keys() == {"FRA"}

        assert new_creator.synthetic_countries.keys() == {"FRA"}

        new_creator.synthetic_countries["FRA"].reset_firm_function_dependent(
            capital_inputs_utilisation_rate=0.1,
            initial_inventory_to_input_fraction=0.1,
            intermediate_inputs_utilisation_rate=0.2,
            zero_initial_debt=False,
            zero_initial_deposits=False,
        )

        assert True
