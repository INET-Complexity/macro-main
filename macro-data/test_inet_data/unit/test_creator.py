from inet_data import Creator
from pathlib import Path
import tempfile

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self, configuration):
        config_path = TEST_PATH / "unit" / "default_unit_test.yaml"
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        with tempfile.TemporaryDirectory() as temp_dir:
            creator = Creator.default_init(
                configuration=configuration,
                raw_data_path=raw_data_path,
                processed_data_path=Path(temp_dir),
                create_exogenous_industry_data=False,
                testing=True,
            )
        assert True
