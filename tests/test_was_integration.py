"""
Tests for WAS (Wealth and Assets Survey) integration with the macro_data package.

This module tests the integration of WAS data with the synthetic population generation
and model initialization processes.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.was_synthetic_population import (
    SyntheticWASPopulation,
    RESTRICT_COLS,
    CONVERT_HH_COLS,
    CONVERT_IND_COLS,
)
from macro_data.readers.population_data.was_reader import WASReader


class TestWASIntegration:
    """Test class for WAS integration functionality."""

    def test_was_reader_initialization(self):
        """Test that WAS reader can be initialized correctly."""
        # Mock data
        mock_individuals_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'HID': [1, 1, 2],
            'Age': [25, 30, 45],
            'Gender': [1, 2, 1],
            'Employee Income': [30000, 40000, 50000],
        })
        
        mock_households_df = pd.DataFrame({
            'HID': [1, 2],
            'Income': [70000, 50000],
            'Value of the Main Residence': [200000, 150000],
            'Wealth in Deposits': [10000, 5000],
        })

        # Mock exchange rates
        mock_exchange_rates = Mock()
        mock_exchange_rates.from_eur_to_lcu.return_value = 1.0

        # Test WAS reader initialization
        was_reader = WASReader(
            country_name_short="GB",
            individuals_df=mock_individuals_df,
            households_df=mock_households_df,
        )

        assert was_reader.country_name_short == "GB"
        assert len(was_reader.individuals_df) == 3
        assert len(was_reader.households_df) == 2

    def test_was_synthetic_population_initialization(self):
        """Test that SyntheticWASPopulation can be initialized correctly."""
        # Mock data
        mock_individuals_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'HID': [1, 1, 2],
            'Age': [25, 30, 45],
            'Gender': [1, 2, 1],
            'Employee Income': [30000, 40000, 50000],
            'Labour Status': [1, 1, 1],
            'Education': [3, 4, 3],
        })
        
        mock_households_df = pd.DataFrame({
            'HID': [1, 2],
            'Income': [70000, 50000],
            'Value of the Main Residence': [200000, 150000],
            'Wealth in Deposits': [10000, 5000],
            'Type': [6, 6],  # Two adults younger than 65
        })

        # Test SyntheticWASPopulation initialization
        was_population = SyntheticWASPopulation(
            country_name="United Kingdom",
            country_name_short="GB",
            scale=1000,
            year=2022,
            industries=["A01", "B05", "C10"],
            individual_data=mock_individuals_df,
            household_data=mock_households_df,
            social_housing_rent=500.0,
            coefficient_fa_income=0.05,
            consumption_weights=np.array([0.3, 0.4, 0.3]),
            consumption_weights_by_income=np.array([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]),
            investment=np.array([0.2, 0.3, 0.5]),
        )

        assert was_population.country_name == "United Kingdom"
        assert was_population.country_name_short == "GB"
        assert was_population.scale == 1000
        assert was_population.year == 2022
        assert len(was_population.industries) == 3

    def test_was_column_lists(self):
        """Test that WAS-specific column lists are properly defined."""
        # Test that RESTRICT_COLS includes WAS-specific variables
        assert "Value of Household Vehicles" in RESTRICT_COLS
        assert "Value of Household Valuables" in RESTRICT_COLS
        assert "Value of Self-Employment Businesses" in RESTRICT_COLS
        assert "Formal Financial Assets" in RESTRICT_COLS
        assert "Voluntary Pension" in RESTRICT_COLS
        assert "Outstanding Balance of Credit Card Debt" in RESTRICT_COLS

        # Test that CONVERT_HH_COLS includes WAS-specific monetary variables
        assert "Employee Income" in CONVERT_HH_COLS
        assert "Employee Income Net" in CONVERT_HH_COLS
        assert "Self-Employment Income" in CONVERT_HH_COLS
        assert "Value of Household Vehicles" in CONVERT_HH_COLS
        assert "Formal Financial Assets" in CONVERT_HH_COLS

        # Test that CONVERT_IND_COLS includes WAS-specific individual variables
        assert "Employee Income" in CONVERT_IND_COLS
        assert "Employee Income Net" in CONVERT_IND_COLS
        assert "Self-Employment Income" in CONVERT_IND_COLS

    def test_was_wealth_computation(self):
        """Test WAS-specific wealth computation methods."""
        # Mock household data with WAS-specific variables
        mock_households_df = pd.DataFrame({
            'HID': [1, 2, 3],
            'Value of the Main Residence': [200000, 150000, 300000],
            'Value of other Properties': [50000, 0, 100000],
            'Value of Household Vehicles': [15000, 10000, 20000],
            'Value of Household Valuables': [5000, 3000, 8000],
            'Value of Self-Employment Businesses': [0, 25000, 0],
            'Wealth in Deposits': [10000, 5000, 15000],
            'Formal Financial Assets': [20000, 10000, 30000],
            'Other Assets': [5000, 2000, 8000],
            'Voluntary Pension': [30000, 15000, 40000],
            'Outstanding Balance of HMR Mortgages': [100000, 80000, 150000],
            'Other Property Mortgage': [10000, 5000, 15000],
            'Outstanding Balance of Credit Card Debt': [2000, 1000, 3000],
            'Outstanding Balance of other Non-Mortgage Loans': [5000, 2000, 8000],
        })

        # Create WAS population instance
        was_population = SyntheticWASPopulation(
            country_name="United Kingdom",
            country_name_short="GB",
            scale=1000,
            year=2022,
            industries=["A01", "B05", "C10"],
            individual_data=pd.DataFrame(),
            household_data=mock_households_df,
            social_housing_rent=500.0,
            coefficient_fa_income=0.05,
            consumption_weights=np.array([0.3, 0.4, 0.3]),
            consumption_weights_by_income=np.array([[0.3, 0.4, 0.3]]),
            investment=np.array([0.2, 0.3, 0.5]),
        )

        # Test wealth computation methods
        was_population.set_household_other_real_assets_wealth()
        was_population.set_household_total_real_assets()
        was_population.set_household_deposits()
        was_population.set_household_other_financial_assets()
        was_population.set_household_financial_assets()
        was_population.set_household_wealth()
        was_population.set_household_mortgage_debt()
        was_population.set_household_other_debt()
        was_population.set_household_debt()
        was_population.set_household_net_wealth()

        # Verify wealth calculations
        assert "Wealth Other Real Assets" in was_population.household_data.columns
        assert "Wealth in Real Assets" in was_population.household_data.columns
        assert "Wealth in Financial Assets" in was_population.household_data.columns
        assert "Wealth" in was_population.household_data.columns
        assert "Debt" in was_population.household_data.columns
        assert "Net Wealth" in was_population.household_data.columns

        # Check that wealth values are reasonable
        assert was_population.household_data["Wealth"].min() > 0
        assert was_population.household_data["Net Wealth"].min() < was_population.household_data["Wealth"].max()

    def test_was_income_computation(self):
        """Test WAS-specific income computation methods."""
        # Mock data
        mock_individuals_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'HID': [1, 1, 2],
            'Employee Income': [30000, 40000, 50000],
        })
        
        mock_households_df = pd.DataFrame({
            'HID': [1, 2],
            'Regular Social Transfers': [2000, 1500],
            'Income from Financial Assets': [1000, 500],
            'Rental Income from Real Estate': [0, 2000],
        })

        # Create WAS population instance
        was_population = SyntheticWASPopulation(
            country_name="United Kingdom",
            country_name_short="GB",
            scale=1000,
            year=2022,
            industries=["A01", "B05", "C10"],
            individual_data=mock_individuals_df,
            household_data=mock_households_df,
            social_housing_rent=500.0,
            coefficient_fa_income=0.05,
            consumption_weights=np.array([0.3, 0.4, 0.3]),
            consumption_weights_by_income=np.array([[0.3, 0.4, 0.3]]),
            investment=np.array([0.2, 0.3, 0.5]),
        )

        # Test income computation methods
        was_population.set_household_employee_income()
        was_population.set_household_income_from_financial_assets()
        was_population.set_household_income()

        # Verify income calculations
        assert "Employee Income" in was_population.household_data.columns
        assert "Income from Financial Assets" in was_population.household_data.columns
        assert "Income" in was_population.household_data.columns

        # Check that income values are reasonable
        assert was_population.household_data["Employee Income"].min() >= 0
        assert was_population.household_data["Income"].min() > 0

    def test_was_saving_and_investment_rates(self):
        """Test WAS-specific saving and investment rate calculations."""
        # Mock household data
        mock_households_df = pd.DataFrame({
            'HID': [1, 2, 3],
            'Income': [50000, 75000, 100000],
            'Wealth': [100000, 200000, 500000],
        })

        # Create WAS population instance
        was_population = SyntheticWASPopulation(
            country_name="United Kingdom",
            country_name_short="GB",
            scale=1000,
            year=2022,
            industries=["A01", "B05", "C10"],
            individual_data=pd.DataFrame(),
            household_data=mock_households_df,
            social_housing_rent=500.0,
            coefficient_fa_income=0.05,
            consumption_weights=np.array([0.3, 0.4, 0.3]),
            consumption_weights_by_income=np.array([[0.3, 0.4, 0.3]]),
            investment=np.array([0.2, 0.3, 0.5]),
        )

        # Test saving and investment rate calculations
        was_population.set_household_saving_rates()
        was_population.set_household_investment_rates(capital_formation_taxrate=0.1)

        # Verify saving and investment rates
        assert "Saving Rate" in was_population.household_data.columns
        assert "Investment Rate" in was_population.household_data.columns

        # Check that rates are within reasonable bounds
        assert was_population.household_data["Saving Rate"].min() >= 0
        assert was_population.household_data["Saving Rate"].max() <= 0.5
        assert was_population.household_data["Investment Rate"].min() >= 0
        assert was_population.household_data["Investment Rate"].max() <= 0.4

    def test_was_emissions_calculation(self):
        """Test WAS-specific emissions calculation."""
        # Mock household data
        mock_households_df = pd.DataFrame({
            'HID': [1, 2, 3],
            'Amount spent on Consumption of Goods and Services': [
                [1000, 2000, 1500],
                [1500, 2500, 2000],
                [2000, 3000, 2500]
            ],
            'Investment': [
                [500, 1000, 800],
                [800, 1200, 1000],
                [1000, 1500, 1200]
            ],
        })

        # Create WAS population instance
        was_population = SyntheticWASPopulation(
            country_name="United Kingdom",
            country_name_short="GB",
            scale=1000,
            year=2022,
            industries=["A01", "B05", "C10"],
            individual_data=pd.DataFrame(),
            household_data=mock_households_df,
            social_housing_rent=500.0,
            coefficient_fa_income=0.05,
            consumption_weights=np.array([0.3, 0.4, 0.3]),
            consumption_weights_by_income=np.array([[0.3, 0.4, 0.3]]),
            investment=np.array([0.2, 0.3, 0.5]),
        )

        # Test emissions calculation
        emission_factors = np.array([0.1, 0.2, 0.15])
        emitting_indices = np.array([0, 1, 2])
        was_population.add_emissions(emission_factors, emitting_indices, tau_cf=0.1)

        # Verify emissions calculation
        assert "Emissions" in was_population.household_data.columns
        assert was_population.total_emissions >= 0

    def test_was_restrict_method(self):
        """Test that the restrict method works with WAS-specific columns."""
        # Mock household data with all WAS columns
        mock_households_df = pd.DataFrame({
            col: np.random.rand(10) for col in RESTRICT_COLS
        })

        # Create WAS population instance
        was_population = SyntheticWASPopulation(
            country_name="United Kingdom",
            country_name_short="GB",
            scale=1000,
            year=2022,
            industries=["A01", "B05", "C10"],
            individual_data=pd.DataFrame(),
            household_data=mock_households_df,
            social_housing_rent=500.0,
            coefficient_fa_income=0.05,
            consumption_weights=np.array([0.3, 0.4, 0.3]),
            consumption_weights_by_income=np.array([[0.3, 0.4, 0.3]]),
            investment=np.array([0.2, 0.3, 0.5]),
        )

        # Test restrict method
        was_population.restrict()

        # Verify that only restricted columns remain
        assert set(was_population.household_data.columns) == set(RESTRICT_COLS)
        assert len(was_population.household_data.columns) == len(RESTRICT_COLS)


if __name__ == "__main__":
    pytest.main([__file__])
