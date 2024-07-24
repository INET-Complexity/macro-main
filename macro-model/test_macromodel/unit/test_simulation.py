import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from macromodel.configurations import SimulationConfiguration, CountryConfiguration
from macromodel.simulation import Simulation, check_compatibility
from macro_data.configuration.countries import Country as CountryName


@pytest.mark.parametrize("seed", [0, 100, 150, 200, 145])
def test_simulation(datawrapper, seed):
    """Test the simulation."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = seed

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    households = simulation.countries["FRA"].households
    individuals = simulation.countries["FRA"].individuals

    n_individuals = individuals.n_individuals
    households_lengths = [len(corr_ind) for corr_ind in households.states["corr_individuals"]]
    assert n_individuals == sum(households_lengths)
    # no empty households
    assert all(households_lengths)

    for _ in range(10):
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


def test_random_seed(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation.iterate()

    gdp1 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    simulation_bis = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation_bis.iterate()

    gdp_bis = np.stack(simulation_bis.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert gdp1 == pytest.approx(gdp_bis, rel=1e-2)


def test_reset(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation.iterate()

    gdp1 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    simulation.reset()

    assert len(simulation.countries["FRA"].firms.ts.historic("price")) == 1

    for i in range(3):
        simulation.iterate()

    gdp2 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert gdp1 == pytest.approx(gdp2, rel=1e-2)


def test_change_config(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation.iterate()

    gdp1 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()
    new_configuration = deepcopy(simulation.configuration)

    # first just change seed
    new_configuration.seed = 1

    simulation.reset(new_configuration)

    for i in range(3):
        simulation.iterate()

    gdp2 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert np.sum(gdp1 - gdp2) != 0

    # reset seed again, check that changing params  change the output

    new_configuration.seed = 0

    # edit France config
    new_configuration.country_configurations["FRA"].firms.parameters.capital_inputs_utilisation_rate = 0.5

    # edit France config
    new_configuration.country_configurations["FRA"].firms.parameters.capital_inputs_utilisation_rate = 0.5

    original_param = new_configuration.country_configurations["FRA"].firms.functions.prices.parameters[
        "price_setting_speed_gf"
    ]

    new_configuration.country_configurations["FRA"].firms.functions.prices.parameters["price_setting_speed_gf"] = (
        1 - original_param
    )

    simulation.reset(new_configuration)

    assert len(simulation.countries["FRA"].firms.ts.historic("price")) == 1

    for i in range(3):
        simulation.iterate()

    gdp3 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert np.sum(gdp1 - gdp3) != 0


def test_reset_row_params(datawrapper):
    """Test the reset params."""
    country_sim_configuration = CountryConfiguration()

    sim_configuration = SimulationConfiguration(country_configurations={"FRA": country_sim_configuration})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    for _ in range(5):
        simulation.iterate()

    values = [0.0, 1.0]

    for x in values:
        new_row_conf = deepcopy(sim_configuration.row_configuration)
        new_row_conf.functions.exports.parameters["consistency"] = x
        sim_configuration.row_configuration = new_row_conf

        simulation.reset(sim_configuration)
        row = simulation.rest_of_the_world
        func = row.functions["exports"]

        param = func.consistency

        assert param == x
        simulation.iterate()


def test_reset_firm_params(datawrapper):
    """Test the reset params."""
    country_sim_configuration = CountryConfiguration()

    def redo_configuration(
        country_conf: CountryConfiguration,
        target_inputs_capital_: float,
    ):
        new_country_conf_ = deepcopy(country_conf)
        new_country_conf_.firms.functions.target_production.parameters[
            "intermediate_inputs_target_considers_capital_inputs"
        ] = target_inputs_capital_
        return new_country_conf_

    country_sim_configuration.firms.reset_params["capital_inputs_utilisation_rate"] = 0.1
    country_sim_configuration.firms.reset_params["intermediate_inputs_utilisation_rate"] = 0.1

    sim_configuration = SimulationConfiguration(country_configurations={"FRA": country_sim_configuration})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    for _ in range(5):
        simulation.iterate()

    values = np.linspace(0, 1, 10)

    for x in values:
        new_country_conf = redo_configuration(country_sim_configuration, x)
        sim_configuration.country_configurations["FRA"] = new_country_conf

        simulation.reset(sim_configuration)
        firms = simulation.countries["FRA"].firms
        func = firms.functions["target_production"]

        param = func.intermediate_inputs_target_considers_capital_inputs

        assert param == x
        simulation.iterate()
