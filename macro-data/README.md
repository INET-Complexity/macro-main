[![Build and test package](https://github.com/INET-Complexity/inet-macro-data/actions/workflows/ci.yml/badge.svg)](https://github.com/INET-Complexity/inet-macro-data/actions/workflows/ci.yml)

# INET's MacroModel - Data

## Install the package 
In a python environment with Python >=3.10, run

```
pip install https://github.com/INET-Complexity/inet-macro-data.git
```
If this doesn't work, it may be because git is not set up properly locally.

Try instead cloning the repository,
```
git clone https://github.com/INET-Complexity/inet-macro-data.git
```
and then installing the package from the local repository,
```
pip install ./inet-macro-data
```

## Run data 

__You should not run the model in a folder where you have cloned this repository. The goal of this is to work as a package. Run this in a folder created elsewhere in your computer specifically for a project.__

Assuming your working directory is structured as follows

```
.
├── configs
│   └── default.yaml
└── data
    ├── processed_data
    └── raw_data
```


To run the data, simply run the following

```python
from macro_data import DataConfiguration, DataWrapper
import yaml

# path to your yaml config file
config_file_path = "./configs/default.yaml"

raw_data_path = "./data/raw_data"

with open(config_file_path, "r") as f:
    config = DataConfiguration(**yaml.safe_load(f))

data = DataWrapper.from_config(config, raw_data_path=raw_data_path)

# save the agents to a pickle 

data.save("./agents.pkl")
```

This assumes that you have the raw data downloaded with the correct directory structure, which must be the same
as the structure of `./test_inet_data/unit/sample_raw_data`. The downloads need to be handled by a specific downloader 
depending on the data structure chosen by the user.

This will pickle the `DataWrapper` object, containing the synthetic countries and all the necessary data to
run a simulation, so that it can be loaded by the `inet-macro-model` package. 