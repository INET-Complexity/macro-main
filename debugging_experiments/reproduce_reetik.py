"""
Script to reproduce and investigate the TFP growth household income problem.

This script compares two simulations:
1. With TFP growth enabled
2. Without TFP growth (control)

The issue: Household income drops when TFP growth is enabled.
"""

from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import macro_data
from macromodel.configurations import SimulationConfiguration, CountryConfiguration
from macromodel.simulation import Simulation
from macromodel.debug import TFPLaborLog, capture_tfp_labor_snapshot


# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
PKL_PATH = SCRIPT_DIR / "data.pkl"


def load_data():
    """Load the macro data from pickle file."""
    return macro_data.DataWrapper.init_from_pickle(PKL_PATH)


def setup_substitution_bundles(data):
    """
    Set up energy substitution bundles for firms and households.

    Returns:
        tuple: (firms_substitution_bundles, households_substitution_bundles)
    """
    bundled_industries = ["B05a", "B05b", "B05c", "C19"]
    energy_bundle = [list(data.industries).index(ind) for ind in bundled_industries]

    firms_substitution_bundles = [energy_bundle]
    households_substitution_bundles = [energy_bundle]

    return firms_substitution_bundles, households_substitution_bundles


def create_base_configuration(data, firms_bundles, household_bundles, t_max=40, seed=1):
    """
    Create the base simulation configuration with common parameters.

    Args:
        data: DataWrapper with economic data
        firms_bundles: Substitution bundles for firms
        household_bundles: Substitution bundles for households
        t_max: Maximum simulation time steps
        seed: Random seed

    Returns:
        SimulationConfiguration
    """
    configuration = SimulationConfiguration(
        country_configurations={
            "CAN": CountryConfiguration.n_industry_default(
                n_industries=data.n_industries,
                firms_bundles=firms_bundles,
                household_bundles=household_bundles,
            )
        },
        t_max=t_max,
        seed=seed,
    )

    # Set common parameters for all countries
    for country in configuration.country_configurations:
        cfg = configuration.country_configurations[country]

        # Firm parameters
        cfg.firms.functions.target_intermediate_inputs.parameters["beta"] = 2.5
        cfg.firms.functions.prices.parameters["price_setting_speed_gf"] = 0.4
        cfg.firms.functions.prices.parameters["price_setting_speed_dp"] = 0.48
        cfg.firms.functions.prices.parameters["price_setting_speed_cp"] = 0.45

        # Household parameters
        cfg.households.functions.consumption.parameters["elasticity_of_substitution"] = 2.5

        # Labor market parameters
        cfg.labour_market.functions.clearing.parameters["random_firing_probability"] = 10 ** (-2.5)
        cfg.labour_market.functions.clearing.parameters["individuals_quitting_temperature"] = 10 ** (-2.5)

    return configuration


def configure_tfp_growth(configuration, data, enable_growth=True):
    """
    Configure TFP growth settings for a simulation.

    Args:
        configuration: SimulationConfiguration to modify
        data: DataWrapper with industry information
        enable_growth: If True, enable TFP growth; if False, disable it

    Returns:
        Modified configuration
    """
    cfg = configuration.country_configurations["CAN"].firms
    industries = list(data.industries)

    if enable_growth:
        # Enable TFP growth
        cfg.functions.productivity_growth.name = "SimpleTFPGrowth"
        cfg.parameters.tfp_base_growth_rate = 0.002
        cfg.parameters.tfp_investment_elasticity = 0.5
        cfg.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.5,
        }

        # Set productivity investment planner
        cfg.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
        cfg.functions.productivity_investment_planner.parameters = {
            "n_firms": len(industries),
            "hurdle_rate": 0.0,
            "investment_effectiveness": 0.2,
            "investment_elasticity": 0.5,
            "max_investment_fraction": 0.2,
            "technical_diminishing_returns": 0.07,
        }
    else:
        # Disable TFP growth
        cfg.functions.productivity_growth.name = "SimpleTFPGrowth"
        cfg.parameters.tfp_base_growth_rate = 0.0
        cfg.parameters.tfp_investment_elasticity = 0.0
        cfg.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.0,
        }

        # Disable productivity investment
        cfg.functions.productivity_investment_planner.name = "NoProductivityInvestmentPlanner"
        cfg.functions.productivity_investment_planner.parameters = {
            "n_firms": len(industries),
            "hurdle_rate": 0.0,
            "investment_effectiveness": 0.0,
            "investment_elasticity": 0.0,
            "max_investment_fraction": 0.0,
            "technical_diminishing_returns": 0.0,
        }

    return configuration


def run_comparison_simulations(data, t_max=40, seed=1, enable_logging=False):
    """
    Run both simulations (with and without TFP growth) and return them.

    Args:
        data: DataWrapper with economic data
        t_max: Maximum simulation time steps
        seed: Random seed
        enable_logging: If True, enable detailed TFP/labor logging (H1)

    Returns:
        tuple: (simulation_with_growth, simulation_without_growth, log_with_growth, log_without_growth)
        If enable_logging=False, logs will be None
    """
    # Setup substitution bundles
    firms_bundles, household_bundles = setup_substitution_bundles(data)

    # Create base configuration
    base_config = create_base_configuration(data, firms_bundles, household_bundles, t_max, seed)

    # Create configuration with TFP growth
    config_with_growth = deepcopy(base_config)
    configure_tfp_growth(config_with_growth, data, enable_growth=True)

    # Create configuration without TFP growth
    config_no_growth = deepcopy(base_config)
    configure_tfp_growth(config_no_growth, data, enable_growth=False)

    # Create simulations
    print("Creating simulation with TFP growth...")
    simulation_with_growth = Simulation.from_datawrapper(
        datawrapper=data,
        simulation_configuration=config_with_growth
    )

    print("Creating simulation without TFP growth...")
    simulation_no_growth = Simulation.from_datawrapper(
        datawrapper=data,
        simulation_configuration=config_no_growth
    )

    # Setup logging if enabled
    log_with_growth = None
    log_without_growth = None

    if enable_logging:
        print("Enabling H1 (Labor Substitution) logging...")
        log_with_growth = TFPLaborLog()
        log_without_growth = TFPLaborLog()

        # Register posthooks to capture snapshots
        def log_with_growth_hook(sim, t, year, month):
            snapshot = capture_tfp_labor_snapshot(sim, t)
            log_with_growth.add_snapshot(snapshot)

        def log_without_growth_hook(sim, t, year, month):
            snapshot = capture_tfp_labor_snapshot(sim, t)
            log_without_growth.add_snapshot(snapshot)

        simulation_with_growth.posthooks.append(log_with_growth_hook)
        simulation_no_growth.posthooks.append(log_without_growth_hook)

    # Run simulations
    print("Running simulation with TFP growth...")
    simulation_with_growth.run()

    print("Running simulation without TFP growth...")
    simulation_no_growth.run()

    return simulation_with_growth, simulation_no_growth, log_with_growth, log_without_growth


def plot_household_income_comparison(sim_with_growth, sim_no_growth):
    """
    Plot household income comparison between simulations.

    Args:
        sim_with_growth: Simulation with TFP growth enabled
        sim_no_growth: Simulation without TFP growth
    """
    income_growth = sim_with_growth.countries["CAN"].households.ts.get_aggregate("income")
    income_no_growth = sim_no_growth.countries["CAN"].households.ts.get_aggregate("income")

    plt.figure(figsize=(10, 6))
    plt.plot(income_growth, label="With TFP Growth", linewidth=2)
    plt.plot(income_no_growth, label="No Growth", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Income")
    plt.title("Household Income Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main(enable_logging=True):
    """Main execution function.

    Args:
        enable_logging: If True, enable H1 (Labor Substitution) logging and save CSV logs
    """
    # Load data
    print("Loading data...")
    data = load_data()

    # Run comparison simulations with optional logging
    sim_with_growth, sim_no_growth, log_with_growth, log_without_growth = run_comparison_simulations(
        data, t_max=40, seed=1, enable_logging=enable_logging
    )

    # Plot results
    plot_household_income_comparison(sim_with_growth, sim_no_growth)

    print("\nSimulation complete!")
    print(f"Final income with TFP growth: {sim_with_growth.countries['CAN'].households.ts.get_aggregate('income')[-1]:.2f}")
    print(f"Final income without TFP growth: {sim_no_growth.countries['CAN'].households.ts.get_aggregate('income')[-1]:.2f}")

    # Save logs if logging was enabled
    if enable_logging and log_with_growth and log_without_growth:
        print("\nSaving H1 (Labor Substitution) logs...")
        df_with_growth = pd.DataFrame(log_with_growth.to_dict_list())
        df_without_growth = pd.DataFrame(log_without_growth.to_dict_list())

        log_with_path = SCRIPT_DIR / "h1_logs_with_tfp.csv"
        log_without_path = SCRIPT_DIR / "h1_logs_without_tfp.csv"

        df_with_growth.to_csv(log_with_path, index=False)
        df_without_growth.to_csv(log_without_path, index=False)

        print("Logs saved to:")
        print(f"  - {log_with_path}")
        print(f"  - {log_without_path}")

        # Print key diagnostics
        print("\n" + "=" * 60)
        print("H1: LABOR SUBSTITUTION DIAGNOSTICS")
        print("=" * 60)

        # Final values
        final_tfp_with = df_with_growth['avg_tfp_multiplier'].iloc[-1]
        final_tfp_without = df_without_growth['avg_tfp_multiplier'].iloc[-1]
        final_employment_with = df_with_growth['total_employment'].iloc[-1]
        final_employment_without = df_without_growth['total_employment'].iloc[-1]
        final_unemployment_with = df_with_growth['unemployment_rate'].iloc[-1]
        final_unemployment_without = df_without_growth['unemployment_rate'].iloc[-1]

        print(f"\nFinal TFP multiplier:")
        print(f"  With TFP growth: {final_tfp_with:.4f}")
        print(f"  Without TFP growth: {final_tfp_without:.4f}")
        print(f"  Difference: {final_tfp_with - final_tfp_without:.4f}")

        print(f"\nFinal employment:")
        print(f"  With TFP growth: {final_employment_with}")
        print(f"  Without TFP growth: {final_employment_without}")
        print(f"  Difference: {final_employment_with - final_employment_without}")

        print(f"\nFinal unemployment rate:")
        print(f"  With TFP growth: {final_unemployment_with * 100:.2f}%")
        print(f"  Without TFP growth: {final_unemployment_without * 100:.2f}%")
        print(f"  Difference: {(final_unemployment_with - final_unemployment_without) * 100:.2f} percentage points")

        # Check if H1 is supported
        if final_unemployment_with > final_unemployment_without + 0.01:  # 1% threshold
            print("\n✓ H1 SUPPORTED: Unemployment is higher with TFP growth")
        else:
            print("\n✗ H1 NOT SUPPORTED: Unemployment is not higher with TFP growth")


if __name__ == "__main__":
    main()
