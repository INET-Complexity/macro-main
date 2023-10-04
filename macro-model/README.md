[![Build and test package](https://github.com/INET-Complexity/inet-macro-model/actions/workflows/ci.yml/badge.svg)](https://github.com/INET-Complexity/inet-macro-model/actions/workflows/ci.yml)

# INET's MacroModel - Model

## Install the package 
In a python environment with Python >=3.10, run

```
pip install https://github.com/INET-Complexity/inet-macro-model.git
```

## Run model 

Before running the model, make sure to create the data using the [inet-macro-data](https://github.com/INET-Complexity/inet-macro-data) package. __You should not run the model in a folder where you have cloned this repository. The goal of this is to work as a package.__
Assuming your working directory is structured as follows,

```
.
├── configs
│   └── default.yaml
└── data
    ├── processed_data
    └── raw_data
```
after the data has been created and processed into the `processed_data` directory, you can run the model and write logs to a file `logs.log` as follows

```python
import yaml
from pathlib import Path
from inet_macromodel import Runner
from inet_macromodel import check_existing_processed_data
import logging

# define format for logs 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logs.log",
)


def run_model(
    config_filename: Path | str, data_dir: Path | str, output_dir: Path | str
) -> None:
    # if type of data_path is str, convert it to Path
    if type(data_dir) == str:
        data_dir = Path(data_dir)
    # if type of config_path is str, convert it to Path
    if type(config_filename) == str:
        config_filename = Path(config_filename)
    # if type of output_dir is str, convert it to Path
    if type(output_dir) == str:
        output_dir = Path(output_dir)

    config = yaml.safe_load(open(config_filename, "r"))
    processed_data_code = check_existing_processed_data(
        config=config,
        data_path=data_dir,
    )
    if processed_data_code is None:
        raise ValueError("No processed data found")

    Runner(
        config_path=config_filename,
        processed_data_path=data_dir
        / "processed_data"
        / processed_data_code
        / "data.h5",
        output_path=output_dir,
    ).run(random_seed=0)


if __name__ == "__main__":
    run_model(
        config_filename="./configs/default.yaml",
        data_dir="./data",
        output_dir="./output",
    )
```


This will save the run data into an `output` directory, writing into `./output/data.h5`. If you want to have less verbose logs, you can change the logging level to `logging.INFO` or `logging.WARNING`.
