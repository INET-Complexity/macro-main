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
    """
    Matches households with banks based on a random selection.

    Args:
        population (SyntheticPopulation): The synthetic population data.
        banks (SyntheticBanks): The synthetic banks data.

    Returns:
        None
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
    if banks.bank_data[banks_field].values.sum() == 0:
        banks.bank_data[banks_field] = np.full(
            banks.bank_data.shape[0],
            1.0 / banks.bank_data.shape[0] * population.household_data[households_field].sum(),
        )
    else:
        banks.bank_data[banks_field] *= (
            population.household_data[households_field].sum() / banks.bank_data[banks_field].sum()
        )
