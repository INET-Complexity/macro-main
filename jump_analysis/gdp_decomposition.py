# %%
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import yaml

from macro_data import DataWrapper
from macromodel.configurations import SimulationConfiguration
from macromodel.simulation import Simulation


def plot_gdp_decomposition(
    simulation: Simulation, country_code: str = "FRA", normalize: bool = True, show_verification: bool = True
):
    """
    Plot GDP output decomposition for a given country from a simulation.

    Args:
        simulation: A completed Simulation object
        country_code (str): Country code to analyze (default: "FRA")
        normalize (bool): If True, normalize all components to their initial values (default: True)
        show_verification (bool): If True, print verification and summary tables (default: True)

    Returns:
        dict: Dictionary containing all the GDP components as arrays
    """
    # Extract GDP components for the specified country
    economy = simulation.countries[country_code].economy

    # Get GDP output and its components over time
    gdp_output = economy.ts.get_aggregate("gdp_output")
    total_output = economy.ts.get_aggregate("total_output")
    total_intermediate_consumption = economy.ts.get_aggregate("total_intermediate_consumption")
    total_taxes_on_production = economy.ts.get_aggregate("total_taxes_on_production")
    total_taxes_on_products = economy.ts.get_aggregate("total_taxes_less_subsidies_on_products")
    total_real_rent_paid = economy.ts.get_aggregate("total_real_rent_paid")
    total_imp_rent_paid = economy.ts.get_aggregate("total_imp_rent_paid")

    # Calculate gross value added (output - intermediate consumption - production taxes)
    gross_value_added = total_output - total_intermediate_consumption - total_taxes_on_production

    # Store components in dictionary for easy access
    components = {
        "gdp_output": gdp_output,
        "total_output": total_output,
        "total_intermediate_consumption": total_intermediate_consumption,
        "total_taxes_on_production": total_taxes_on_production,
        "total_taxes_on_products": total_taxes_on_products,
        "total_real_rent_paid": total_real_rent_paid,
        "total_imp_rent_paid": total_imp_rent_paid,
        "gross_value_added": gross_value_added,
    }

    # Normalize components if requested
    if normalize:
        normalized_components = {}
        for key, values in components.items():
            if len(values) > 0 and values[0] != 0:
                normalized_components[key] = values / values[0]
            else:
                normalized_components[key] = values
        plot_components = normalized_components
        y_label = "Normalized Value (t=0 = 1.0)"
        title_suffix = " (Normalized)"
    else:
        plot_components = components
        y_label = "Value"
        title_suffix = ""

    # Create the decomposition plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Time axis
    time = np.arange(len(gdp_output))

    # Top plot: Main GDP components
    ax1.plot(time, plot_components["gdp_output"], "k-", linewidth=2, label="GDP Output (Total)")
    ax1.plot(time, plot_components["total_output"], "b-", label="Total Output")

    # For intermediate consumption, show as negative in the formula but plot absolute value
    if normalize:
        ax1.plot(time, plot_components["total_intermediate_consumption"], "r--", label="Intermediate Consumption")
    else:
        ax1.plot(time, -plot_components["total_intermediate_consumption"], "r--", label="- Intermediate Consumption")

    ax1.plot(time, plot_components["gross_value_added"], "g-", label="Gross Value Added")

    ax1.set_xlabel("Time")
    ax1.set_ylabel(y_label)
    ax1.set_title(f"GDP Output Decomposition - Main Components{title_suffix}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Adjustments and smaller components
    ax2.plot(time, plot_components["total_taxes_on_products"], "orange", label="Taxes on Products")

    if normalize:
        ax2.plot(
            time, plot_components["total_taxes_on_production"], "purple", linestyle="--", label="Taxes on Production"
        )
    else:
        ax2.plot(
            time, -plot_components["total_taxes_on_production"], "purple", linestyle="--", label="- Taxes on Production"
        )

    ax2.plot(time, plot_components["total_real_rent_paid"], "brown", label="Real Rent Paid")
    ax2.plot(time, plot_components["total_imp_rent_paid"], "pink", label="Imputed Rent")

    # Show the sum of adjustments
    if normalize:
        # For normalized case, this is trickier - show actual adjustment values normalized by GDP[0]
        adjustments = (
            total_taxes_on_products - total_taxes_on_production + total_real_rent_paid + total_imp_rent_paid
        ) / gdp_output[0]
    else:
        adjustments = total_taxes_on_products - total_taxes_on_production + total_real_rent_paid + total_imp_rent_paid
    ax2.plot(time, adjustments, "k-", linewidth=2, label="Total Adjustments")

    ax2.set_xlabel("Time")
    ax2.set_ylabel(y_label)
    ax2.set_title(f"GDP Output Decomposition - Adjustments{title_suffix}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if show_verification:
        # Verification: Check that components sum to GDP
        print(f"GDP Decomposition Verification for {country_code}:")
        print(
            "GDP Output formula: total_output - intermediate_consumption - taxes_on_production + taxes_on_products + rent_paid + rent_imputed"
        )
        print()

        for t in range(min(len(gdp_output), 3)):  # Show first 3 timesteps
            calculated_gdp = (
                total_output[t]
                - total_intermediate_consumption[t]
                - total_taxes_on_production[t]
                + total_taxes_on_products[t]
                + total_real_rent_paid[t]
                + total_imp_rent_paid[t]
            )

            print(f"Time {t}:")
            print(f"  GDP Output (recorded): {gdp_output[t]:.2f}")
            print(f"  GDP Output (calculated): {calculated_gdp:.2f}")
            print(f"  Difference: {abs(gdp_output[t] - calculated_gdp):.6f}")
            print()

        # Jump analysis table: Changes from t=0 to t=1
        if len(gdp_output) > 1:
            print(f"GDP Jump Analysis from t=0 to t=1 for {country_code}:")
            print(f"{'Component':<30} {'Relative Change (%)':<18} {'GDP Points':<12}")
            print("-" * 65)

            # Helper function to calculate relative change and GDP contribution
            def calc_jump_metrics(component, component_name):
                if len(component) > 1:
                    if component[0] != 0:
                        rel_change = ((component[1] / component[0]) - 1) * 100
                    else:
                        rel_change = 0.0 if component[1] == 0 else float("inf")

                    gdp_contribution = (component[1] - component[0]) / gdp_output[0] * 100
                    return rel_change, gdp_contribution
                else:
                    return 0.0, 0.0

            # Calculate metrics for each component
            components_data = [
                (total_output, "Total Output", "+"),
                (total_intermediate_consumption, "Intermediate Consumption", "-"),
                (total_taxes_on_production, "Taxes on Production", "-"),
                (total_taxes_on_products, "Taxes on Products", "+"),
                (total_real_rent_paid, "Real Rent Paid", "+"),
                (total_imp_rent_paid, "Imputed Rent", "+"),
            ]

            jump_analysis = []
            for component, name, sign in components_data:
                rel_change, gdp_points = calc_jump_metrics(component, name)
                # Adjust GDP contribution sign based on component role in GDP formula
                if sign == "-":
                    gdp_points = -gdp_points

                jump_analysis.append((name, rel_change, gdp_points, abs(gdp_points)))
                print(f"{sign + ' ' + name:<30} {rel_change:<18.2f} {gdp_points:<12.3f}")

            # Calculate total GDP jump
            gdp_rel_change, _ = calc_jump_metrics(gdp_output, "GDP Output")
            gdp_jump_points = (gdp_output[1] - gdp_output[0]) / gdp_output[0] * 100

            print("-" * 65)
            print(f"{'GDP Output (Total)':<30} {gdp_rel_change:<18.2f} {gdp_jump_points:<12.3f}")
            print()

            # Sort by absolute GDP contribution to identify biggest contributors
            jump_analysis.sort(key=lambda x: x[3], reverse=True)
            print("Components ranked by absolute GDP point contribution:")
            print(f"{'Rank':<5} {'Component':<25} {'GDP Points':<12}")
            print("-" * 45)
            for i, (name, rel_change, gdp_points, abs_points) in enumerate(jump_analysis, 1):
                print(f"{i:<5} {name:<25} {gdp_points:<12.3f}")
        else:
            print("Not enough time steps for jump analysis (need at least 2 time steps)")
            print(f"Current length: {len(gdp_output)}")

    return components


# %%

# Example usage
# Load data and configuration (same as reproduce_jump.py)
with open("data.pkl", "rb") as f:
    data_wrapper = pkl.load(f)

data = DataWrapper.init_from_pickle("data.pkl")

with open("configuration.yaml", "r") as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

configuration = SimulationConfiguration(**configuration)
configuration.t_max = 5

# Run single simulation
model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)
model.run()

# Plot GDP decomposition with normalization (default)
components = plot_gdp_decomposition(model, country_code="FRA", normalize=True)

# Plot GDP decomposition without normalization
# components_abs = plot_gdp_decomposition(model, country_code="FRA", normalize=False)

# %%

rent = model.countries["FRA"].economy.ts.get_aggregate("total_real_rent_paid")
# %%
