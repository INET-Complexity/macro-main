from inet_data import Creator
from pathlib import Path
import tempfile

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self):
        config_path = TEST_PATH / "unit" / "default_unit_test.yaml"
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        with tempfile.TemporaryDirectory() as temp_dir:
            Creator(
                config_path=config_path,
                raw_data_path=raw_data_path,
                processed_data_path=Path(temp_dir) / "sample_processed_data",
                force_download=False,
                create_exogenous_industry_data=True,
                random_seed=0,
                testing=True,
            ).create(save_output=True)
        assert True
