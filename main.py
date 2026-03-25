import argparse
from pathlib import Path

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

# arguments = argparse.ArgumentParser()

# arguments.add_argument("-sd", "--save_dir", type=str, dest="save_dir",
#                         help="Set the save directory for Macro output", default="./output/", required=True)
# arguments.add_argument("-f", "--file_name", type=str, dest="file_name",
#                         help="Set the file name for Macro output", default="multi_country_simulation.h5", required=True)
# args = arguments.parse_args()

# save_dir, file_name = args.save_dir, args.file_name
name_pricesetter = "ExoEnergyExogenousPriceSetter"  # "DefaultPriceSetter" #"EnergyExogenousPriceSetter" #"CIMSEnergyExogenousPriceSetter"
name_pricesetter_ROW = "CIMSEnergyExogenousROWPriceSetter"  # "EnergyExogenousROWPriceSetter" #"InflationRoWPriceSetter" #"CIMSEnergyExogenousROWPriceSetter"

# Configure data preprocessing
data_config = default_data_configuration(
    countries=["FRA", "CAN", "USA"],
    proxy_country_dict={"CAN": "FRA", "USA": "FRA"},  # Use France as proxy for non-EU countries
)

# Create DataWrapper instance
creator = DataWrapper.from_config(
    configuration=data_config,
    raw_data_path="raw_data",  # "path/to/raw/data",
    single_hfcs_survey=True,  # Use single survey for household finance data
)

# Save processed data
creator.save("./data.pkl")

# Load preprocessed data
data = DataWrapper.init_from_pickle("./data.pkl")

# Configure country-specific parameters
country_configurations = {"FRA": CountryConfiguration(), "CAN": CountryConfiguration(), "USA": CountryConfiguration()}
for country in country_configurations:
    country_configurations[country].firms.functions.prices.name = name_pricesetter

# Create simulation configuration
configuration = SimulationConfiguration(country_configurations=country_configurations, t_max=20)  # Number of time steps

# Initialize simulation
model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)

# Run simulation and save results
model.run()
# model.save(save_dir=Path(save_dir), file_name=file_name)

# model =
