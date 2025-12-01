"""
Standalone test script for SyntheticWASPopulation.

This script demonstrates two ways to test SyntheticWASPopulation:
1. Direct instantiation using __init__ (simpler, uses mock data)
2. Using from_readers (more realistic, requires actual data files)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.was_synthetic_population import (
    SyntheticWASPopulation,
)
from macro_data.readers import AGGREGATED_INDUSTRIES


def test_with_mock_data():
    """Test SyntheticWASPopulation using mock data (simpler approach)."""
    print("=== Testing with Mock Data ===")
    
    # Create sample individual data
    n_individuals = 1000
    n_households = 500
    
    individuals_df = pd.DataFrame({
        'Personal identifier': range(n_individuals),
        'HID': np.repeat(range(n_households), 2)[:n_individuals],
        'Age': np.random.randint(18, 80, n_individuals),
        'Gender': np.random.choice([1, 2], n_individuals),
        'Employee Income': np.random.normal(30000, 15000, n_individuals),
        'Self-Employment Income Total': np.random.normal(10000, 5000, n_individuals),
        'Labour Status': np.random.choice([1, 2, 3, 4], n_individuals),
        'Education': np.random.choice([1, 2, 3, 4], n_individuals),
        'Employment Industry': np.random.choice(AGGREGATED_INDUSTRIES[:5], n_individuals),
        'Activity Status': np.random.choice([1, 2, 3], n_individuals),
        'Income from Unemployment Benefits': np.random.normal(5000, 2000, n_individuals),
        'Income': np.random.normal(35000, 18000, n_individuals),
    })
    individuals_df.set_index('Personal identifier', inplace=True)
    
    # Create sample household data
    households_df = pd.DataFrame({
        'HID': range(n_households),
        'Household identifier': range(n_households),
        'Type': np.random.choice([6, 7, 8, 9, 10, 11, 12], n_households),
        'Tenure Status of the Main Residence': np.random.choice([1, 2, 3, 4], n_households),
        'Rent Paid': np.random.normal(800, 300, n_households),
        'Value of the Main Residence': np.random.normal(250000, 100000, n_households),
        'Value of other Properties': np.random.normal(50000, 30000, n_households),
        'Number of Properties other than Household Main Residence': np.random.randint(0, 3, n_households),
        'Rental Income from Real Estate': np.random.normal(5000, 3000, n_households),
        'Total value of savings accounts': np.random.normal(15000, 10000, n_households),
        'Total value of all formal financial assets': np.random.normal(25000, 15000, n_households),
        'Total value of individual pension wealth': np.random.normal(100000, 50000, n_households),
        'Total mortgage on main residence': np.random.normal(120000, 80000, n_households),
        'Total property debt excluding main residence': np.random.normal(20000, 15000, n_households),
        'Hhold total outstanding credit/store/charge card balance': np.random.normal(3000, 2000, n_households),
        'Employee Income': np.random.normal(60000, 25000, n_households),
        'Regular Social Transfers': np.random.normal(5000, 2000, n_households),
        'Income from Financial Assets': np.random.normal(2000, 1000, n_households),
        'Income': np.random.normal(65000, 28000, n_households),
    })
    households_df.set_index('HID', inplace=True)
    
    # Create consumption and investment weights
    n_industries = len(AGGREGATED_INDUSTRIES)
    consumption_weights = np.random.rand(n_industries)
    consumption_weights = consumption_weights / consumption_weights.sum()
    consumption_weights_by_income = np.random.rand(5, n_industries)
    consumption_weights_by_income = consumption_weights_by_income / consumption_weights_by_income.sum(axis=1, keepdims=True)
    investment_weights = np.random.rand(n_industries)
    investment_weights = investment_weights / investment_weights.sum()
    
    # Create SyntheticWASPopulation instance
    population = SyntheticWASPopulation(
        country_name="United Kingdom",
        country_name_short="GB",
        scale=1000,
        year=2014,
        industries=AGGREGATED_INDUSTRIES,
        individual_data=individuals_df,
        household_data=households_df,
        social_housing_rent=500.0,
        coefficient_fa_income=0.05,
        consumption_weights=consumption_weights,
        consumption_weights_by_income=consumption_weights_by_income,
        investment=investment_weights,
    )
    
    print(f"✓ Created population with {len(population.household_data)} households")
    print(f"✓ Created population with {len(population.individual_data)} individuals")
    
    # Test methods
    print("\nTesting methods...")
    population.compute_household_wealth()
    print("✓ compute_household_wealth() completed")
    
    population.compute_household_income(total_social_transfers=1000000)
    print("✓ compute_household_income() completed")
    
    population.set_household_saving_rates()
    print("✓ set_household_saving_rates() completed")
    
    population.set_household_investment_rates(capital_formation_taxrate=0.1)
    print("✓ set_household_investment_rates() completed")
    
    # Display some statistics
    print("\n=== Statistics ===")
    print(f"Average household wealth: £{population.household_data['Wealth'].mean():,.0f}")
    print(f"Average household income: £{population.household_data['Income'].mean():,.0f}")
    print(f"Average saving rate: {population.household_data['Saving Rate'].mean():.1%}")
    
    return population


def test_with_readers(raw_data_path: Path, year: int = 2014):
    """Test SyntheticWASPopulation using actual data readers (more realistic)."""
    print("\n=== Testing with Data Readers ===")
    print(f"Using data path: {raw_data_path}")
    print(f"Year: {year}")
    
    try:
        from macro_data.readers.default_readers import DataReaders
        from macro_data.readers.exogenous_data import ExogenousCountryData
        
        # Initialize data readers
        print("\nInitializing DataReaders...")
        readers = DataReaders.from_raw_data(
            raw_data_path=raw_data_path,
            country_names=[Country.UNITED_KINGDOM],
            simulation_year=year,
            scale_dict={Country.UNITED_KINGDOM: 10000},
            industries=AGGREGATED_INDUSTRIES,
            aggregate_industries=True,
        )
        print("✓ DataReaders initialized")
        
        # Check if WAS data is available
        if Country.UNITED_KINGDOM not in readers.was:
            print("⚠ Warning: No WAS data found. Check that WAS data files exist in:")
            print(f"  {raw_data_path / 'was' / 'stata'}")
            return None
        
        # Get industry data (simplified - in real usage this comes from ICIO)
        # Need to include all required columns for compile_national_accounts_data
        print("\nPreparing industry data...")
        n_industries = len(AGGREGATED_INDUSTRIES)
        base_values = np.random.rand(n_industries) * 1e6
        
        industry_vectors = pd.DataFrame({
            "Output in LCU": base_values * 10,
            "Household Consumption in LCU": base_values,
            "Household Capital Inputs in LCU": base_values * 0.5,
            "Firm Capital Inputs in LCU": base_values * 2,
            "Government Consumption in LCU": base_values * 0.3,
            "Intermediate Inputs Use in LCU": base_values * 5,
            "Labour Compensation in LCU": base_values * 3,
            "Taxes Less Subsidies in LCU": base_values * 0.1,
            "Exports in LCU": base_values * 0.8,
            "Imports in LCU": base_values * 0.6,
            "Number of Firms": np.random.randint(10, 100, n_industries),
        }, index=AGGREGATED_INDUSTRIES)
        
        industry_data = {
            "industry_vectors": industry_vectors,
        }
        
        # Get exogenous data
        print("Preparing exogenous data...")
        exogenous_data = ExogenousCountryData.from_data_readers(
            readers=readers,
            country_name=Country.UNITED_KINGDOM,
            industry_vectors=industry_vectors,
            year=year,
            quarter=1,
        )
        print("✓ Exogenous data prepared")
        
        # Calculate total unemployment benefits
        total_unemployment_benefits = readers.get_total_unemployment_benefits_lcu(
            Country.UNITED_KINGDOM, year
        )
        
        # Create synthetic population
        print("\nCreating SyntheticWASPopulation...")
        population = SyntheticWASPopulation.from_readers(
            readers=readers,
            country_name=Country.UNITED_KINGDOM,
            country_name_short="GB",
            scale=10000,
            year=year,
            quarter=1,
            industry_data=industry_data,
            industries=AGGREGATED_INDUSTRIES,
            total_unemployment_benefits=total_unemployment_benefits,
            exogenous_data=exogenous_data,
            rent_as_fraction_of_unemployment_rate=0.25,
            n_quantiles=5,
        )
        
        print(f"✓ Created population with {len(population.household_data)} households")
        print(f"✓ Created population with {len(population.individual_data)} individuals")
        
        # Test methods
        print("\nTesting methods...")
        population.compute_household_wealth()
        print("✓ compute_household_wealth() completed")
        
        population.compute_household_income(total_social_transfers=total_unemployment_benefits)
        print("✓ compute_household_income() completed")
        
        # Display statistics
        print("\n=== Statistics ===")
        print(f"Average household wealth: £{population.household_data['Wealth'].mean():,.0f}")
        print(f"Average household income: £{population.household_data['Income'].mean():,.0f}")
        
        return population
        
    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print("Some dependencies may not be available.")
        return None
    except FileNotFoundError as e:
        print(f"⚠ File not found: {e}")
        print("Make sure WAS data files are in the correct location.")
        return None
    except Exception as e:
        print(f"⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("SyntheticWASPopulation Standalone Test")
    print("=" * 60)
    print("\nNote: The existing unit test at")
    print("  tests/test_macro_data/unit/test_readers/test_population_data/test_was_reader.py")
    print("  does test with real data, but it looks in:")
    print("  tests/test_macro_data/unit/sample_raw_data/was")
    print("  (and skips if files aren't found)")
    print("\nThis script tests with your actual data location.")
    print("=" * 60)
    
    # Test 1: With mock data (always works)
    population_mock = test_with_mock_data()
    
    # Test 2: With actual data readers (requires data files)
    # Set the path to your actual data directory
    # The WAS data should be in: raw_data_path / "was" / "stata"
    raw_data_path = Path(__file__).parent / "inet-macro-dev" / "data" / "raw_data"
    
    if raw_data_path.exists():
        print(f"\nFound data path: {raw_data_path}")
        was_data_path = raw_data_path / "was" / "stata"
        if was_data_path.exists():
            print(f"Found WAS data directory: {was_data_path}")
            # Check for actual WAS files
            was_files = list(was_data_path.glob("was_round_*_person_eul_*.dta")) + \
                       list(was_data_path.glob("was_wave_*_person_eul_*.dta"))
            if was_files:
                print(f"Found {len(was_files)} WAS data file(s)")
                # Try to determine the year from available files
                # For now, use 2014 (most recent round 8)
                population_real = test_with_readers(raw_data_path, year=2014)
            else:
                print(f"⚠ No WAS .dta files found in {was_data_path}")
                print("  Expected files like: was_round_8_person_eul_*.dta")
        else:
            print(f"⚠ WAS data directory not found: {was_data_path}")
    else:
        print(f"\n⚠ Data path not found: {raw_data_path}")
        print("  Update the path in this script to point to your data directory")
        print("  Expected structure: raw_data_path / 'was' / 'stata' / 'was_round_*_*.dta'")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

