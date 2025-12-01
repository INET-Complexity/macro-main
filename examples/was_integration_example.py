"""
Example script demonstrating how to use WAS (Wealth and Assets Survey) data 
with the macro_data package.

This script shows how to:
1. Initialize WAS data readers
2. Create synthetic populations from WAS data
3. Generate synthetic countries using WAS data
4. Access WAS-specific economic indicators

Note: This example assumes WAS data files are available in the specified paths.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from macro_data import DataConfiguration, Country, CountryDataConfiguration
from macro_data.configuration import (
    FirmsDataConfiguration,
    BanksDataConfiguration,
    CentralBankDataConfiguration,
)
from macro_data.data_wrapper import DataWrapper
from macro_data.processing.synthetic_population.was_synthetic_population import (
    SyntheticWASPopulation,
)
from macro_data.readers.population_data.was_reader import WASReader


def main():
    """Main function demonstrating WAS integration."""
    
    print("=== WAS Integration Example ===")
    print("This example demonstrates how to use WAS data with the macro_data package.")
    print()
    
    # Example 1: Initialize WAS Reader
    print("1. Initializing WAS Reader...")
    
    # Mock exchange rates (in a real scenario, this would come from actual data)
    class MockExchangeRates:
        def from_eur_to_lcu(self, country, year):
            return 1.0  # GBP to GBP conversion
    
    exchange_rates = MockExchangeRates()
    
    # Create sample WAS data (in a real scenario, this would be loaded from files)
    sample_individuals_df = pd.DataFrame({
        'ID': range(1, 101),
        'HID': np.repeat(range(1, 51), 2),  # 50 households, 2 individuals each
        'Age': np.random.randint(18, 80, 100),
        'Gender': np.random.choice([1, 2], 100),
        'Employee Income': np.random.normal(30000, 15000, 100),
        'Labour Status': np.random.choice([1, 2, 3], 100),
        'Education': np.random.choice([1, 2, 3, 4], 100),
    })
    
    sample_households_df = pd.DataFrame({
        'HID': range(1, 51),
        'Income': np.random.normal(60000, 25000, 50),
        'Value of the Main Residence': np.random.normal(250000, 100000, 50),
        'Wealth in Deposits': np.random.normal(15000, 10000, 50),
        'Formal Financial Assets': np.random.normal(25000, 15000, 50),
        'Outstanding Balance of HMR Mortgages': np.random.normal(120000, 80000, 50),
        'Type': np.random.choice([6, 7, 8, 9, 10, 11, 12], 50),
    })
    
    # Initialize WAS reader
    was_reader = WASReader(
        country_name_short="GB",
        individuals_df=sample_individuals_df,
        households_df=sample_households_df,
    )
    
    print(f"   - Loaded {len(was_reader.individuals_df)} individuals")
    print(f"   - Loaded {len(was_reader.households_df)} households")
    print()
    
    # Example 2: Create Synthetic WAS Population
    print("2. Creating Synthetic WAS Population...")
    
    # Mock industry data
    industries = ["A01", "B05", "C10", "D35", "E36"]
    industry_data = {
        "industry_vectors": pd.DataFrame({
            "Household Consumption in LCU": np.random.rand(len(industries)) * 1000000,
            "Household Capital Inputs in LCU": np.random.rand(len(industries)) * 500000,
            "Number of Firms by Industry": np.random.randint(10, 100, len(industries)),
        }, index=industries)
    }
    
    # Mock exogenous data
    class MockExogenousData:
        def __init__(self):
            self.unemployment_rate = 0.05
            self.participation_rate = 0.75
    
    exogenous_data = MockExogenousData()
    
    # Create synthetic WAS population
    was_population = SyntheticWASPopulation(
        country_name="United Kingdom",
        country_name_short="GB",
        scale=1000,
        year=2022,
        industries=industries,
        individual_data=sample_individuals_df,
        household_data=sample_households_df,
        social_housing_rent=500.0,
        coefficient_fa_income=0.05,
        consumption_weights=np.random.rand(len(industries)),
        consumption_weights_by_income=np.random.rand(5, len(industries)),
        investment=np.random.rand(len(industries)),
    )
    
    print(f"   - Created synthetic population with {was_population.number_of_households} households")
    print(f"   - Industries: {was_population.industries}")
    print()
    
    # Example 3: Compute Wealth and Income
    print("3. Computing Wealth and Income...")
    
    # Compute household wealth
    was_population.compute_household_wealth()
    was_population.compute_household_income(total_social_transfers=1000000)
    
    # Display wealth statistics
    wealth_stats = was_population.household_data["Wealth"].describe()
    print(f"   - Average household wealth: £{wealth_stats['mean']:,.0f}")
    print(f"   - Median household wealth: £{wealth_stats['50%']:,.0f}")
    print(f"   - Wealth range: £{wealth_stats['min']:,.0f} - £{wealth_stats['max']:,.0f}")
    
    # Display income statistics
    income_stats = was_population.household_data["Income"].describe()
    print(f"   - Average household income: £{income_stats['mean']:,.0f}")
    print(f"   - Median household income: £{income_stats['50%']:,.0f}")
    print()
    
    # Example 4: Set Saving and Investment Rates
    print("4. Setting Saving and Investment Rates...")
    
    was_population.set_household_saving_rates()
    was_population.set_household_investment_rates(capital_formation_taxrate=0.1)
    
    saving_stats = was_population.household_data["Saving Rate"].describe()
    investment_stats = was_population.household_data["Investment Rate"].describe()
    
    print(f"   - Average saving rate: {saving_stats['mean']:.1%}")
    print(f"   - Average investment rate: {investment_stats['mean']:.1%}")
    print()
    
    # Example 5: Normalize Consumption and Investment
    print("5. Normalizing Consumption and Investment...")
    
    # Mock IOT data
    iot_consumption = pd.Series(np.random.rand(len(industries)) * 1000000, index=industries)
    iot_investment = pd.Series(np.random.rand(len(industries)) * 500000, index=industries)
    
    was_population.normalise_household_consumption(iot_consumption, vat=0.2)
    was_population.normalise_household_investment(tau_cf=0.1, iot_hh_investment=iot_investment)
    
    print("   - Consumption and investment normalized to match IOT data")
    print()
    
    # Example 6: Display WAS-Specific Variables
    print("6. WAS-Specific Variables:")
    
    was_specific_vars = [
        "Value of Household Vehicles",
        "Value of Household Valuables", 
        "Value of Self-Employment Businesses",
        "Formal Financial Assets",
        "Voluntary Pension",
        "Outstanding Balance of Credit Card Debt",
    ]
    
    for var in was_specific_vars:
        if var in was_population.household_data.columns:
            value = was_population.household_data[var].mean()
            print(f"   - {var}: £{value:,.0f}")
        else:
            print(f"   - {var}: Not available in sample data")
    
    print()
    
    # Example 7: Restrict to Essential Columns
    print("7. Restricting to Essential Columns...")
    
    was_population.restrict()
    print(f"   - Restricted to {len(was_population.household_data.columns)} essential columns")
    print(f"   - Columns: {list(was_population.household_data.columns)}")
    print()
    
    print("=== WAS Integration Example Complete ===")
    print("The WAS data has been successfully integrated and processed!")
    print()
    print("Key Features Demonstrated:")
    print("- WAS data loading and processing")
    print("- Synthetic population generation from WAS data")
    print("- Wealth and income computation using WAS-specific variables")
    print("- Saving and investment rate calculations")
    print("- Consumption and investment normalization")
    print("- WAS-specific variable handling")


if __name__ == "__main__":
    main()

