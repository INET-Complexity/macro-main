[![Build and test package](https://github.com/INET-Complexity/inet-macro-model/actions/workflows/ci.yml/badge.svg)](https://github.com/INET-Complexity/inet-macro-model/actions/workflows/ci.yml)

# MacroModel - Model

## Install the package 

To install this package, you need first to have the macro data package installed. You can find this package in [this repository](https://github.com/macro-cosm/macrocosm-macro-data). 

In a python environment with Python >=3.11, run

```
pip install https://github.com/macro-cosm/macrocosm-macro-data.git
```
and then install the macro model package. 

```
pip install https://github.com/macro-cosm/macrocosm-macro-model.git
```


If this doesn't work, it may be because git is not set up properly locally.

Try instead cloning the repositories,
```
git clone https://github.com/macro-cosm/macrocosm-macro-data.git
```
and then installing the package from the local repository,
```
pip install ./macrocosm-macro-data
```
and proceding in the same way for the model package.

## Run model 

Before running the model, make sure to create the data using the [inet-macro-data](https://github.com/INET-Complexity/inet-macro-data) package. __You should not run the model in a folder where you have cloned this repository. The goal of this is to work as a package.__


The starting point is to have generated a `data.pkl` file using the macro-data package. Refer to the readme of the macro-data package for more information on how to generate this file.

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