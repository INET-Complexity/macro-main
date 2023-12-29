import numpy as np

from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


def match_households_with_banks(
    synthetic_population: SyntheticPopulation,
    synthetic_banks: SyntheticBanks,
) -> None:
    """
    Matches households with banks based on a random selection.

    Args:
        synthetic_population (SyntheticPopulation): The synthetic population data.
        synthetic_banks (SyntheticBanks): The synthetic banks data.

    Returns:
        None
    """
    bank_by_household = np.random.choice(
        range(synthetic_banks.number_of_banks),
        len(synthetic_population.household_data),
        replace=True,
    )
    synthetic_population.household_data["Corresponding Bank ID"] = bank_by_household
    synthetic_banks.bank_data["Corresponding Households ID"] = [
        list(np.where(bank_by_household == bank_id)[0]) for bank_id in range(synthetic_banks.number_of_banks)
    ]
