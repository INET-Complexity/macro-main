# Macromodel

## Introduction

This repository contains the code to run the macromodel.
It consists of two packages, `macro-data` and `macromodel`.
The first package takes care of the data processing and the second package runs the model.


## Install the package

Clone the repository and install the package by running the following command 
in a python environment with Python >=3.10 and in the root directory of the repository:

```bash
pip install .
```

If you want to install the package in editable mode, run

```bash
pip install -e ./ --config-settings editable_mode=strict
```

## Run data preprocessing

To run the model, you first need to preprocess the data. This is done by creating
a `DataWrapper` object, which you can then use directly or save to a pickle file.

You can use the template below:

```python
from macro_data.configuration_utils import default_data_configuration
from macro_data import DataWrapper

DATA_PATH = "path/to/raw/data" # replace with the path to the raw data

data_config = default_data_configuration(countries=["FRA"]) # replace with the desired countries

# the data_config can be modified

creator = DataWrapper.from_config(
        configuration=data_config, raw_data_path=DATA_PATH, single_hfcs_survey=False
    )

creator.save("path/to/save/data.pkl") # replace with the path to save the data
```


## Run the model

The model can be run using a datawrapper. In this case, we will do as if we have a `data.pkl` file already created.
Note that the code above can be used to create the data file, but you would need to adapt it to preprocess data for 
Canada and the USA in addition to France (by also specifying a proxy country for these countries), using eg

```python
fra_can_usa_configuration = default_data_configuration(
    countries=["FRA", "CAN", "USA"],
    proxy_country_dict={"CAN": "FRA", "USA": "FRA"},
)
```

The model can be run using the following code (in this example we are running the model for France, Canada and the USA, and this requires also the data for these countries to be present in the `data.pkl` file):

```python
from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

from pathlib import Path

# load the data

data = DataWrapper.init_from_pickle("./data.pkl")

# configurations for France, Canda and USA
country_configurations = {
    "FRA": CountryConfiguration(),
    "CAN": CountryConfiguration(),
    "USA": CountryConfiguration(),
}

configuration = SimulationConfiguration(
    country_configurations=country_configurations, t_max=20
)

# instantiate the model
model = Simulation.from_datawrapper(
    datawrapper=data, simulation_configuration=configuration
)

# run the model
model.run()
model.save(save_dir=Path("./output/"), file_name="can_usa_fra_run.h5")
```


This will save the run data into an `output` directory, writing into `./output/can_usa_fra_run.h5`. If you want to have less verbose logs, you can use a logger to change the logging level to `logging.INFO` or `logging.WARNING`.


This simulation runs the model using default country configurations. You can modify them directly from the `CountryConfiguration`  object. 