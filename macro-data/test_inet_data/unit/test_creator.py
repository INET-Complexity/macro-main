from inet_data import Creator
from pathlib import Path
import tempfile

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self, configuration):
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = Creator.default_init(
            configuration=configuration,
            raw_data_path=raw_data_path,
            create_exogenous_industry_data=False,
            testing=True,
        )
        assert creator.synthetic_firms.keys() == {"FRA"}
