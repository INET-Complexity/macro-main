import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from macro_data.configuration.countries import Country as CountryName
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation, check_compatibility


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
        simulation.save(save_dir=tmp, file_name="simulation_long.h5")
        simulation.shallow_hdf_save(save_dir=tmp, file_name="simulation_shallow.h5")
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


@pytest.mark.parametrize("seed", [0, 100])
def test_all_industries(allind_datawrapper, seed):
    n_industries = allind_datawrapper.n_industries
    configuration = SimulationConfiguration(
        country_configurations={"FRA": CountryConfiguration.n_industry_default(n_industries=n_industries)}
    )

    configuration.seed = seed

    simulation = Simulation.from_datawrapper(datawrapper=allind_datawrapper, simulation_configuration=configuration)

    for _ in range(3):
        simulation.iterate()

    assert True


def test_canadian_disagg(can_disagg_datawrapper):
    n_industries = can_disagg_datawrapper.n_industries
    firms_bundled_industries = ["B05a", "B05b", "B05c", "C19"]
    industries = can_disagg_datawrapper.industries
    firms_energy_bundle = [list(industries).index(ind) for ind in firms_bundled_industries]

    firms_substitution_bundles = [firms_energy_bundle]

    # Household energy bundle with only B05a and C19 for testing
    household_bundled_industries = ["B05a", "C19"]
    household_energy_bundle = [list(industries).index(ind) for ind in household_bundled_industries]
    household_substitution_bundles = [household_energy_bundle]

    configuration = SimulationConfiguration(
        country_configurations={
            "CAN": CountryConfiguration.n_industry_default(
                n_industries=n_industries,
                firms_bundles=firms_substitution_bundles,
                household_bundles=household_substitution_bundles,
            )
        }
    )

    assert configuration.country_configurations["CAN"].firms.functions.production.name == "BundledLeontief"
    assert (
        configuration.country_configurations["CAN"].households.functions.consumption.name == "CESHouseholdConsumption"
    )

    assert configuration.country_configurations["CAN"].firms.functions.production.name == "BundledLeontief"

    configuration.seed = 0
    simulation = Simulation.from_datawrapper(datawrapper=can_disagg_datawrapper, simulation_configuration=configuration)

    for _ in range(3):
        simulation.iterate()

    shallow_output = simulation.countries["CAN"].shallow_output()

    keys = [
        "Firm Input Emissions",
        "Firm Capital Emissions",
        "Household Consumption Emissions",
        "Household Investment Emissions",
        "Government Emissions",
    ]

    for key in keys:
        assert np.all(shallow_output[key] > 0)

    assert True


def test_can_provincial(can_provincial_datawrapper):
    n_industries = can_provincial_datawrapper.n_industries

    all_provs = can_provincial_datawrapper.synthetic_countries.keys()

    configuration = SimulationConfiguration(
        country_configurations={
            province: CountryConfiguration.n_industry_default(n_industries=n_industries) for province in all_provs
        }
    )

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(
        datawrapper=can_provincial_datawrapper, simulation_configuration=configuration
    )

    for _ in range(3):
        simulation.iterate()

    shallow_output = simulation.countries["CAN_AB"].shallow_output()

    assert True


def test_tfp_growth_with_investment(datawrapper):
    """Test that TFP growth mechanism works with productivity investment.

    Creates two simulations with identical seeds:
    1. Control: No TFP growth (all parameters set to zero/disabled)
    2. Treatment: TFP growth enabled with high investment effectiveness and low hurdle rate

    Verifies that firms in the treatment simulation have higher TFP after several periods.
    """
    # Base configuration for control (no TFP growth)
    config_no_growth = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    config_no_growth.seed = 0  # Fixed seed for reproducibility

    # Disable TFP growth in control
    config_no_growth.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.0
    config_no_growth.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.0

    # Configuration for treatment (with TFP growth)
    config_with_growth = deepcopy(config_no_growth)

    # Enable TFP growth with favorable parameters
    config_with_growth.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001  # 0.1% base growth
    config_with_growth.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.5  # High elasticity

    # Set productivity investment planner parameters
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_investment_planner.name = (
        "SimpleProductivityInvestmentPlanner"
    )
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_investment_planner.parameters = {
        "hurdle_rate": 1e-5,  # Very low hurdle rate (almost no discounting)
        "investment_effectiveness": 0.5,  # High effectiveness
        "investment_elasticity": 0.5,  # Match the TFP elasticity
        "max_investment_fraction": 0.2,  # Allow up to 20% of available cash
    }

    # Also configure the productivity growth function
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
        "investment_effectiveness": 0.5,  # High effectiveness for growth calculation
    }

    # Create simulations
    sim_no_growth = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=config_no_growth)

    sim_with_growth = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=config_with_growth)

    # Get initial TFP values (should be identical)
    initial_tfp_no_growth = sim_no_growth.countries["FRA"].firms.states["tfp_multiplier"].copy()
    initial_tfp_with_growth = sim_with_growth.countries["FRA"].firms.states["tfp_multiplier"].copy()

    # Verify initial TFP values are the same (both should be 1.0)
    np.testing.assert_array_almost_equal(initial_tfp_no_growth, initial_tfp_with_growth)
    np.testing.assert_array_almost_equal(initial_tfp_no_growth, np.ones_like(initial_tfp_no_growth))

    # Run both simulations for several periods
    n_periods = 10
    for _ in range(n_periods):
        sim_no_growth.iterate()
        sim_with_growth.iterate()

    # Get final TFP values
    final_tfp_no_growth = sim_no_growth.countries["FRA"].firms.states["tfp_multiplier"]
    final_tfp_with_growth = sim_with_growth.countries["FRA"].firms.states["tfp_multiplier"]

    # Verify that TFP in the growth simulation is higher
    # Control should remain at 1.0 (no growth)
    np.testing.assert_array_almost_equal(final_tfp_no_growth, np.ones_like(final_tfp_no_growth))

    # Treatment should have TFP > 1.0 for at least most firms
    assert np.mean(final_tfp_with_growth) > 1.0, "Average TFP should be greater than 1.0 with growth enabled"
    assert np.sum(final_tfp_with_growth > 1.0) > len(final_tfp_with_growth) * 0.8, "Most firms should have TFP > 1.0"

    # Verify all firms in treatment have at least as much TFP as control
    assert np.all(final_tfp_with_growth >= final_tfp_no_growth), "All firms should have TFP >= control"

    # Check that productivity investment is actually happening
    if len(sim_with_growth.countries["FRA"].firms.ts.executed_productivity_investment) > 0:
        total_investment = sum(
            inv.sum() for inv in sim_with_growth.countries["FRA"].firms.ts.executed_productivity_investment
        )
        assert (
            total_investment > 0
        ), f"There should be positive productivity investment, first 5 elements: {total_investment[:5]}"


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


def test_longrun(datawrapper):
    """Test the longrun."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()}, t_max=200)

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    simulation.run()

    assert True


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


def test_alternative_labour(datawrapper):
    """Test the alternative labour."""
    country_sim_configuration = CountryConfiguration()

    country_sim_configuration.labour_market.functions.clearing.parameters["firing_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["hiring_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["individuals_quitting"] = True
    # random_firing_probability
    country_sim_configuration.labour_market.functions.clearing.parameters["random_firing_probability"] = 0.02

    sim_configuration = SimulationConfiguration(
        country_configurations={"FRA": country_sim_configuration}, seed=0, t_max=5
    )

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    simulation.run()

    assert True


def test_large_firing_rate(allind_datawrapper):
    country_sim_configuration = CountryConfiguration.n_industry_default(n_industries=allind_datawrapper.n_industries)

    country_sim_configuration.labour_market.functions.clearing.parameters["firing_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["hiring_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["individuals_quitting"] = True
    # random_firing_probability
    country_sim_configuration.labour_market.functions.clearing.parameters["random_firing_probability"] = 0.99

    sim_configuration = SimulationConfiguration(
        country_configurations={"FRA": country_sim_configuration}, seed=0, t_max=5
    )

    simulation = Simulation.from_datawrapper(datawrapper=allind_datawrapper, simulation_configuration=sim_configuration)

    simulation.run()

    assert True


@pytest.mark.parametrize(
    "tfp_growth_type", ["NoOpTFPGrowth", "SimpleTFPGrowth", "StochasticTFPGrowth", "SectoralTFPGrowth"]
)
@pytest.mark.parametrize("seed", [0, 100])
def test_simulation_with_tfp_growth(datawrapper, seed, tfp_growth_type):
    """Test the simulation with different TFP growth configurations."""
    # Create base configuration
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    # Modify the TFP growth configuration
    configuration.country_configurations["FRA"].firms.functions.productivity_growth.name = tfp_growth_type

    # Set parameters based on TFP growth type
    if tfp_growth_type == "NoOpTFPGrowth":
        # No parameters needed for NoOp
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {}
    elif tfp_growth_type == "SimpleTFPGrowth":
        # Parameters for simple TFP growth
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.1
        }
        # Also set the base growth rate in parameters
        configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001  # 0.1% per period
        configuration.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.3
    elif tfp_growth_type == "StochasticTFPGrowth":
        # Parameters for stochastic TFP growth
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.1,
            "shock_std": 0.005,  # 0.5% standard deviation for shocks
        }
        configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001
        configuration.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.3
    elif tfp_growth_type == "SectoralTFPGrowth":
        # Parameters for sectoral TFP growth
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.1,
            "sector_base_growth": {},  # Could specify sector-specific rates here
            "sector_effectiveness": {},  # Could specify sector-specific effectiveness here
        }
        configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001
        configuration.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.3

    configuration.seed = seed

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    # Check that TFP multiplier is initialized
    firms = simulation.countries["FRA"].firms
    assert "tfp_multiplier" in firms.states
    assert np.all(firms.states["tfp_multiplier"] == 1.0)  # Should start at 1.0

    # Run simulation for several iterations
    for _ in range(5):
        simulation.iterate()

    # Check TFP behavior based on type
    final_tfp = firms.states["tfp_multiplier"]

    if tfp_growth_type == "NoOpTFPGrowth":
        # TFP should remain at 1.0 (no growth)
        assert np.allclose(final_tfp, 1.0), f"NoOpTFPGrowth should keep TFP at 1.0, got {final_tfp}"
    else:
        # For other types, TFP might change (though with small growth rates, changes could be minimal)
        # We mainly check that the simulation runs without errors
        assert np.all(final_tfp > 0), f"TFP should be positive, got {final_tfp}"
        assert np.all(np.isfinite(final_tfp)), f"TFP should be finite, got {final_tfp}"

    assert True
