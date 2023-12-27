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

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_file = tmp / "creator.pkl"
            creator.save(tmp_file)
            new_creator = Creator.init_from_pickle(tmp_file)

        assert creator.synthetic_firms.keys() == {"FRA"}

        assert new_creator.synthetic_firms.keys() == {"FRA"}
