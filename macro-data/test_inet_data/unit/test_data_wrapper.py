import yaml

from inet_data import DataWrapper
from pathlib import Path
import tempfile

from inet_data.configuration import Configuration

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = Configuration(**config_dict)
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.default_init(
            configuration=configuration,
            raw_data_path=raw_data_path,
            create_exogenous_industry_data=False,
            single_hfcs_survey=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_file = tmp / "creator.pkl"
            creator.save(tmp_file)
            new_creator = DataWrapper.init_from_pickle(tmp_file)

        assert creator.synthetic_countries.keys() == {"FRA"}

        assert new_creator.synthetic_countries.keys() == {"FRA"}
