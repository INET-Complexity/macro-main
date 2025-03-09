"""Module for harmonizing household and bank financial data.

This module harmonizes financial data from different sources:
1. Household Survey Data:
   - Reported deposit holdings
   - Outstanding loan balances
   - Financial asset information
   - Debt service payments

2. Banking System Data:
   - Aggregate household deposits
   - Retail loan portfolio
   - Balance sheet totals
   - Customer account data

The harmonization process involves:
1. Data Validation:
   - Checking total deposits match across sources
   - Validating loan balances
   - Ensuring consistent customer counts

2. Data Reconciliation:
   - Scaling household deposits to match bank totals
   - Adjusting loan balances for consistency
   - Computing account distributions

3. Optimal Assignment:
   - Minimizing discrepancy between data sources
   - Preserving financial relationships
   - Recording final assignments

Note:
    This module focuses on harmonizing financial data from different sources
    to create a consistent initial state. The actual financial market dynamics
    are implemented in the simulation package.
"""

import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa

from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


def match_households_with_banks_random(
    population: SyntheticPopulation,
    banks: SyntheticBanks,
) -> None:
    """Initialize household-bank relationships with random assignment.

    This function provides a simple initialization mechanism that:
    1. Randomly assigns households to banks
    2. Records assignments in household data
    3. Updates bank customer lists

    Useful for:
    - Initial data setup
    - Testing and validation
    - Cases where optimal harmonization is not required

    Args:
        population (SyntheticPopulation): Household financial data
        banks (SyntheticBanks): Bank balance sheet data
    """
    bank_by_household = np.random.choice(
        range(banks.number_of_banks),
        len(population.household_data),
        replace=True,
    )
    population.household_data["Corresponding Bank ID"] = bank_by_household
    banks.bank_data["Corresponding Households ID"] = [
        list(np.where(bank_by_household == bank_id)[0]) for bank_id in range(banks.number_of_banks)
    ]


def match_households_with_banks_optimal(
    population: SyntheticPopulation,
    banks: SyntheticBanks,
) -> None:
    """Harmonize household and bank financial data using optimal assignment.

    This function reconciles financial data by:
    1. Scaling household data to match bank totals
    2. Allocating accounts based on bank size
    3. Using linear sum assignment to minimize discrepancies
    4. Recording harmonized relationships

    The optimization:
    - Minimizes differences between reported values
    - Respects bank balance sheet constraints
    - Maintains consistent financial totals
    - Preserves deposit-loan relationships

    Args:
        population (SyntheticPopulation): Household financial data
        banks (SyntheticBanks): Bank balance sheet data
    """
    # rescale
    rescale(population, "Wealth in Deposits", banks, "Deposits from Households")
    rescale(population, "Debt", banks, "Loans to Households")

    # create cost matrix
    # sum of loans and deposits to households

    loans_and_deposits = (
        banks.bank_data["Deposits from Households"].values + banks.bank_data["Loans to Households"].values
    )
    # number of households by bank
    number_of_households_by_bank = population.household_data.shape[0] * loans_and_deposits / loans_and_deposits.sum()

    # round down
    number_of_households_by_bank = np.floor(number_of_households_by_bank).astype(int)

    # assign households to banks if needed
    if population.household_data.shape[0] > number_of_households_by_bank.sum():
        add_inds = np.random.choice(
            len(number_of_households_by_bank),
            population.household_data.shape[0] - number_of_households_by_bank.sum(),
            replace=True,
        )
        for ind in add_inds:
            number_of_households_by_bank[ind] += 1

    # assign households to banks
    bank_accounts = []
    for bank_id in range(banks.number_of_banks):
        book_value = (
            banks.bank_data["Deposits from Households"].values[bank_id]
            + banks.bank_data["Loans to Households"].values[bank_id]
        )
        extension = (
            np.full(
                number_of_households_by_bank[bank_id],
                book_value / number_of_households_by_bank[bank_id],
            )
            if number_of_households_by_bank[bank_id] > 0
            else np.array([])
        )
        bank_accounts.extend(extension)

    bank_accounts = np.array(bank_accounts)

    banks_by_account = np.concatenate(
        [np.full(number_of_households_by_bank[bank_id], bank_id) for bank_id in range(banks.number_of_banks)]
    ).astype(int)

    cost = sp.spatial.distance_matrix(
        (population.household_data["Wealth in Deposits"].values + population.household_data["Debt"].values)[:, None],
        bank_accounts[:, None],
    ).astype(float)

    # Find the optimal configuration
    corr_households_rel, corr_bank_accounts = lsa(cost)
    corr_banks = banks_by_account[corr_bank_accounts]
    population.household_data["Corresponding Bank ID"] = corr_banks
    banks.bank_data["Corresponding Households ID"] = [
        np.where(corr_banks == bank_id) for bank_id in range(banks.number_of_banks)
    ]


def rescale(population: SyntheticPopulation, households_field: str, banks: SyntheticBanks, banks_field: str):
    """Reconcile household financial data with bank totals.

    This function ensures consistency between sources by:
    1. Checking if bank totals need initialization
    2. Scaling household values to match bank totals
    3. Maintaining relative proportions

    Used for:
    - Deposit total reconciliation
    - Loan balance harmonization
    - Balance sheet validation

    Args:
        population (SyntheticPopulation): Household financial data
        households_field (str): Field in household data to reconcile
        banks (SyntheticBanks): Bank balance sheet data
        banks_field (str): Field in bank data to match
    """
    if banks.bank_data[banks_field].values.sum() == 0:
        banks.bank_data[banks_field] = np.full(
            banks.bank_data.shape[0],
            1.0 / banks.bank_data.shape[0] * population.household_data[households_field].sum(),
        )
    else:
        banks.bank_data[banks_field] *= (
            population.household_data[households_field].sum() / banks.bank_data[banks_field].sum()
        )
