[![Build and test package](https://github.com/INET-Complexity/inet-macro-data/actions/workflows/ci.yml/badge.svg)](https://github.com/INET-Complexity/inet-macro-data/actions/workflows/ci.yml)

# INET's MacroModel - Data

## Clone the github repository 

Clone the github repository using
```
git clone git@github.com:INET-Complexity/inet-macro-data.git
```


## Install the package 
Create a python environment for the macromodel  with Python >=3.10 and run, in the folder where you have cloned the repository and that contains `requirements.txt`,

```
pip install .
```

## Run data 

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
from inet_data import Creator, create_code

processed_data_code = create_code()
Creator(
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
