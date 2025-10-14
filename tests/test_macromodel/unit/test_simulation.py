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
        "n_firms": config_with_growth.country_configurations["FRA"].firms.n_firms,
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


def test_tfp_only_investment_allocation(datawrapper, seed=42):
    """Test that investment allocation to TFP-only works correctly.

    Configure investment to go 100% to TFP, 0% to technical coefficients.
    Verify TFP multiplier improves while technical coefficient multipliers stay at 1.0.
    """
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = seed

    # Configure for TFP-only investment allocation
    firms_config = configuration.country_configurations["FRA"].firms
    firms_config.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    firms_config.functions.productivity_investment_planner.parameters.update(
        {
            "tfp_investment_share": 1.0,  # 100% to TFP
            "max_investment_fraction": 0.2,  # High investment to see effects
            "investment_effectiveness": 0.3,  # High effectiveness
        }
    )
    firms_config.functions.productivity_growth.name = "SimpleTFPGrowth"
    firms_config.functions.technical_coefficients_growth.name = "NoOpTechnicalGrowth"  # Disable technical growth

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms

    # Store initial values
    initial_tfp = firms.states["tfp_multiplier"].copy()
    initial_intermediate_tech = firms.states["intermediate_tech_multipliers"].copy()
    initial_capital_tech = firms.states["capital_tech_multipliers"].copy()

    # Run simulation for several iterations to allow investment effects
    for _ in range(10):
        simulation.iterate()

    # Check results
    final_tfp = firms.states["tfp_multiplier"]
    final_intermediate_tech = firms.states["intermediate_tech_multipliers"]
    final_capital_tech = firms.states["capital_tech_multipliers"]

    # TFP should have improved (at least some firms should have TFP > 1.0)
    assert np.any(final_tfp > initial_tfp), "TFP should improve with TFP-only investment"
    assert np.all(final_tfp >= 1.0), "TFP multipliers should be >= 1.0"

    # Technical coefficients should remain at 1.0 (no technical investment)
    assert np.allclose(
        final_intermediate_tech, initial_intermediate_tech
    ), "Intermediate tech multipliers should not change with TFP-only investment"
    assert np.allclose(
        final_capital_tech, initial_capital_tech
    ), "Capital tech multipliers should not change with TFP-only investment"
    assert np.allclose(final_intermediate_tech, 1.0), "Intermediate tech multipliers should stay at 1.0"
    assert np.allclose(final_capital_tech, 1.0), "Capital tech multipliers should stay at 1.0"


def test_technical_only_investment_allocation(datawrapper, seed=42):
    """Test that investment allocation to technical coefficients-only works correctly.

    Configure investment to go 0% to TFP, 100% to technical coefficients.
    Verify technical coefficient multipliers improve while TFP stays at 1.0.
    """
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = seed

    # Configure for technical-only investment allocation
    firms_config = configuration.country_configurations["FRA"].firms
    firms_config.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    firms_config.functions.productivity_investment_planner.parameters.update(
        {
            "tfp_investment_share": 0.0,  # 0% to TFP, 100% to technical
            "max_investment_fraction": 0.2,  # High investment to see effects
            "technical_investment_effectiveness": 0.3,  # High effectiveness
            "technical_diminishing_returns": 0.1,  # Low diminishing returns for faster growth
        }
    )
    firms_config.functions.productivity_growth.name = "NoOpTFPGrowth"  # Disable TFP growth
    firms_config.functions.technical_coefficients_growth.name = "SimpleTechnicalGrowth"
    firms_config.functions.technical_coefficients_growth.parameters.update(
        {
            "investment_effectiveness": 0.3,
            "diminishing_returns_factor": 0.1,
        }
    )

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms

    # Store initial values
    initial_tfp = firms.states["tfp_multiplier"].copy()
    initial_intermediate_tech = firms.states["intermediate_tech_multipliers"].copy()
    initial_capital_tech = firms.states["capital_tech_multipliers"].copy()

    # Store base coefficients for comparison
    base_intermediate_coeffs = firms.base_intermediate_inputs_productivity_matrix
    base_capital_coeffs = firms.base_capital_inputs_productivity_matrix

    # Run simulation for several iterations to allow investment effects
    for _ in range(10):
        simulation.iterate()

    # Check results
    final_tfp = firms.states["tfp_multiplier"]
    final_intermediate_tech = firms.states["intermediate_tech_multipliers"]
    final_capital_tech = firms.states["capital_tech_multipliers"]

    # TFP should remain at 1.0 (no TFP investment)
    assert np.allclose(final_tfp, initial_tfp), "TFP should not change with technical-only investment"
    assert np.allclose(final_tfp, 1.0), "TFP multipliers should stay at 1.0"

    # Technical coefficients should have improved
    # At least some multipliers should be > 1.0, and all should be >= 1.0
    intermediate_improved = np.any(final_intermediate_tech > initial_intermediate_tech)
    capital_improved = np.any(final_capital_tech > initial_capital_tech)

    # At least one type should have improved
    assert (
        intermediate_improved or capital_improved
    ), "At least some technical multipliers should improve with technical-only investment"

    # All multipliers should be >= 1.0 (productivity improvements)
    assert np.all(final_intermediate_tech >= 1.0), "Intermediate tech multipliers should be >= 1.0"
    assert np.all(final_capital_tech >= 1.0), "Capital tech multipliers should be >= 1.0"

    # Check effective coefficients vs base coefficients
    # Effective coefficients should be >= base coefficients (due to multipliers >= 1.0)
    effective_intermediate = firms.get_effective_intermediate_coefficients()
    effective_capital = firms.get_effective_capital_coefficients()

    # Get base coefficients for each firm's industry
    firm_industries = firms.states["Industry"]
    base_intermediate_for_firms = base_intermediate_coeffs[:, firm_industries].T
    base_capital_for_firms = base_capital_coeffs[:, firm_industries].T

    # Check that effective >= base (element-wise)
    intermediate_comparison = effective_intermediate >= base_intermediate_for_firms
    capital_comparison = effective_capital >= base_capital_for_firms

    # All should be >= base, and at least some should be > base
    assert np.all(intermediate_comparison), "All effective intermediate coeffs should be >= base"
    assert np.all(capital_comparison), "All effective capital coeffs should be >= base"

    # At least some should be strictly greater (using your suggested check)
    intermediate_some_better = (effective_intermediate > base_intermediate_for_firms).sum() > 0
    capital_some_better = (effective_capital > base_capital_for_firms).sum() > 0

    assert (
        intermediate_some_better or capital_some_better
    ), "At least some effective coefficients should be strictly better than base coefficients"


def test_prehooks(datawrapper):
    """Test that pre-hooks execute correctly before each iteration."""
    # Track hook calls
    hook_calls = []

    def test_hook(simulation: Simulation, year: int, month: int) -> None:
        """Simple test hook that records when it's called."""
        hook_calls.append((year, month))

    # Create simulation
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    # Register the test hook
    simulation.prehooks.append(test_hook)

    # Run for 3 iterations
    for _ in range(3):
        simulation.iterate()

    # Verify hook was called correct number of times
    assert len(hook_calls) == 3, f"Expected 3 hook calls, got {len(hook_calls)}"

    # Verify each call had year and month
    for year, month in hook_calls:
        assert isinstance(year, int), "Year should be an integer"
        assert isinstance(month, int), "Month should be an integer"
        assert 1 <= month <= 12, f"Month should be between 1 and 12, got {month}"


def test_heterogeneous_investment_effectiveness(datawrapper):
    """Test simulation with heterogeneous investment effectiveness across firms.

    This tests that we can set different investment effectiveness parameters
    for different firms and that the simulation runs correctly.
    """
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = 0

    # Get n_firms from the configuration
    n_firms = configuration.country_configurations["FRA"].firms.n_firms

    # Set up productivity investment with heterogeneous parameters
    # Create a list of investment effectiveness values that vary across firms
    # Low effectiveness for first third, medium for middle third, high for last third
    investment_effectiveness_list = []
    for i in range(n_firms):
        if i < n_firms // 3:
            investment_effectiveness_list.append(0.05)  # Low
        elif i < 2 * n_firms // 3:
            investment_effectiveness_list.append(0.10)  # Medium
        else:
            investment_effectiveness_list.append(0.15)  # High

    # Configure productivity investment planner with heterogeneous parameters
    configuration.country_configurations["FRA"].firms.functions.productivity_investment_planner.name = (
        "SimpleProductivityInvestmentPlanner"
    )
    configuration.country_configurations["FRA"].firms.functions.productivity_investment_planner.parameters = {
        "n_firms": n_firms,
        "hurdle_rate": 0.10,  # Uniform
        "investment_effectiveness": investment_effectiveness_list,  # Heterogeneous
        "investment_elasticity": 0.3,  # Uniform
        "max_investment_fraction": 0.15,
        "investment_propensity": 0.5,
    }

    # Enable TFP growth to see the effects
    configuration.country_configurations["FRA"].firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
        "investment_effectiveness": 0.1
    }
    configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    # Verify heterogeneous parameters were set correctly
    planner = simulation.countries["FRA"].firms.functions["productivity_investment_planner"]
    assert isinstance(planner.investment_effectiveness, np.ndarray)
    assert len(planner.investment_effectiveness) == n_firms
    # Check that we have the three different values
    unique_values = np.unique(planner.investment_effectiveness)
    assert len(unique_values) == 3
    assert np.allclose(sorted(unique_values), [0.05, 0.10, 0.15])

    # Store initial TFP
    initial_tfp = simulation.countries["FRA"].firms.states["tfp_multiplier"].copy()

    # Run simulation for several periods
    for _ in range(10):
        simulation.iterate()

    # Get final TFP
    final_tfp = simulation.countries["FRA"].firms.states["tfp_multiplier"]

    # Verify simulation ran successfully
    assert np.all(final_tfp >= initial_tfp), "TFP should not decrease"
    assert np.all(np.isfinite(final_tfp)), "TFP should be finite"

    # Check that at least some firms invested (if there was investment opportunity)
    if len(simulation.countries["FRA"].firms.ts.executed_productivity_investment) > 0:
        total_investment_history = simulation.countries["FRA"].firms.ts.executed_productivity_investment
        total_invested = sum(inv.sum() for inv in total_investment_history)
        # Just verify no errors - investment amount depends on profitability
        assert total_invested >= 0
