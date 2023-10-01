import yaml
import logging
from pathlib import Path

from inet_data.creator import Creator
from inet_data.util.check_existing_processed_data import check_existing_processed_data
from inet_data.util.create_code import create_code


def run_data(
    config_file_path: Path | str,
    data_dir_path: Path | str,
    force_redo: bool = False,
    seed: int = 0,
) -> None:
    config = yaml.safe_load(open(config_file_path, "r"))
    # if type of data_path is str, convert it to Path
    if type(data_dir_path) == str:
        data_dir_path = Path(data_dir_path)
    # if type of config_path is str, convert it to Path
    if type(config_file_path) == str:
        config_file_path = Path(config_file_path)
    if not force_redo:
        processed_data_code = check_existing_processed_data(
            config=config,
            data_path=data_dir_path,
        )
    else:
        processed_data_code = None
    if processed_data_code is None:
        processed_data_code = create_code()
        Creator(
            config_path=config_file_path,
            raw_data_path=data_dir_path / "raw_data",
            processed_data_path=data_dir_path / "processed_data" / processed_data_code / "data.h5",
            force_download=False,
            create_exogenous_industry_data=True,
            random_seed=seed,
        ).create(save_output=True)
    logging.info(f"Processed data {processed_data_code}")
