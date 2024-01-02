import numpy as np

from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


def match_households_with_banks(
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
