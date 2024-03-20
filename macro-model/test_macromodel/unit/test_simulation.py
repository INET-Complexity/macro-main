import tempfile
from pathlib import Path

import pytest

from macromodel.configurations import SimulationConfiguration, CountryConfiguration
from macromodel.simulation import Simulation, check_compatibility
from macro_data.configuration.countries import Country as CountryName


def test_simulation(datawrapper):
    """Test the simulation."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    for _ in range(5):
        simulation.iterate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        simulation.save(save_dir=tmp, file_name="simulation.pkl")
        simulation.shallow_hdf_save(save_dir=tmp, file_name="simulation.h5")
        dicts = simulation.shallow_df_dict()
        assert "FRA" in dicts

    france = simulation.countries[CountryName("FRA")]

    shallow_output = france.shallow_output()

    gross_output = shallow_output["Gross Output"]

    france_datawrapper = datawrapper.synthetic_countries[CountryName("FRA")]
    france_datawrapper_firms = france_datawrapper.firms

    firm_data = france_datawrapper_firms.firm_data
    firms_output_lcu = firm_data.groupby("Industry").apply(lambda x: (x["Production"] * x["Price"]).sum())

    assert gross_output.loc[0] == pytest.approx(firms_output_lcu.sum(), rel=1e-4)

    assert True


def test_check_compatibility(datawrapper):
    """Test the compatibility check."""
    france = CountryName("FRA")
    country_data_configuration = datawrapper.configuration.country_configs[france]
    country_sim_configuration = CountryConfiguration()

    country_sim_configuration.firms.parameters.capital_inputs_utilisation_rate = 0.1
    country_sim_configuration.firms.parameters.intermediate_inputs_utilisation_rate = 0.1

    assert not check_compatibility(country_data_configuration, country_sim_configuration)


def test_reset_params(datawrapper):
    """Test the reset params."""
    country_sim_configuration = CountryConfiguration()

    country_sim_configuration.firms.reset_params["capital_inputs_utilisation_rate"] = 0.1
    country_sim_configuration.firms.reset_params["intermediate_inputs_utilisation_rate"] = 0.1

    sim_configuration = SimulationConfiguration(country_configurations={"FRA": country_sim_configuration})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    for _ in range(5):
        simulation.iterate()
    assert True
