"""Tests for synthetic housing market functionality.

This module tests the creation and validation of synthetic housing markets,
including:
1. Household-property matching
2. Tenure status distribution
3. Property value and rent assignments
4. Tenant-landlord relationships
"""

import pathlib
import numpy as np
import pytest

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)
from macro_data.processing.synthetic_matching.matching_households_with_houses import (
    set_housing_df,
    match_renters_to_properties,
)
from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
)
from macro_data.readers import AGGREGATED_INDUSTRIES

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticHousingMarket:
    """Test suite for synthetic housing market functionality."""

    def test__create(
        self,
        readers,
    ):
        ...
        
    def test_household_matching(
        self,
        readers,
        configuration,
        industry_data,
        exogenous_data,
    ):
        """Test that household matching produces valid results.
        
        This test verifies several key aspects of the housing market matching:
        1. Presence and distribution of different tenure types (owners, private renters)
        2. Pending more work on other tests
        
        Args:
            readers: Data readers for various data sources
            configuration: System configuration
            industry_data: Industry-specific data
            exogenous_data: External economic data
        """
        # Create synthetic population with French data
        france = Country("FRA")
        population = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=france,
            year=2014,
            scale=10000,
            country_name_short=france.to_two_letter_code(),
            industries=AGGREGATED_INDUSTRIES,
            industry_data=industry_data[france],
            rent_as_fraction_of_unemployment_rate=0.5,
            total_unemployment_benefits=1000.0,
            quarter=1,
            exogenous_data=exogenous_data,
        )

        # Debug prints for initial state
        print("\nInitial state:")
        print(f"Total households: {len(population.household_data)}")
        print(f"Number of renters: {np.sum(population.household_data['Tenure Status of the Main Residence'] == 3)}")
        print(f"Number of properties owned: {np.sum(population.household_data['Number of Properties other than Household Main Residence'])}")

        # Set up housing market parameters with realistic values
        rental_income_taxes = 0.2  # 20% tax on rental income
        social_housing_rent = 1000.0  # Standardized social housing rent
        total_imputed_rent = 50000.0  # Total imputed rent for owner-occupied properties

        # Create housing market and perform household-property matching
        housing_df = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=rental_income_taxes,
            social_housing_rent=social_housing_rent,
            total_imputed_rent=total_imputed_rent,
        )

        # Test 1: Verify presence of all tenure types
        # Tenure status: 1 = owner-occupied, 3 = private rental, -1 = social housing
        owners = population.household_data["Tenure Status of the Main Residence"] == 1
        renters = population.household_data["Tenure Status of the Main Residence"] == 3
        social_housing = population.household_data["Tenure Status of the Main Residence"] == -1

        assert owners.sum() > 0, "No owner-occupied households found"
        assert renters.sum() > 0, "No private renters found"
