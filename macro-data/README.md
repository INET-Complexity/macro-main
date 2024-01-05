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
from inet_data import run_data 

#path to your yaml config file
config_file_path = "./configs/default.yaml" 
#path to your data directory
data_dir_path = "./data"

run_data(config_file_path, data_dir_path)
```

This will then download and process the data into the required directories. A default yaml configuration file is available in this repository.

Note that to have more control over the direct data generation, you can modify the call to the creator, as follows

```python
from inet_data import DataWrapper, create_code

processed_data_code = create_code()
DataWrapper(
    config_path=config_file_path,
    raw_data_path=data_dir_path / "raw_data",
    processed_data_path=data_dir_path
                        / "processed_data"
                        / processed_data_code
                        / "data.h5",
    force_download=False,
    create_exogenous_industry_data=True,
    random_seed=0,
).create(save_output=True)
```


This is useful in particular if you want to set a seed for the initialisation data.
