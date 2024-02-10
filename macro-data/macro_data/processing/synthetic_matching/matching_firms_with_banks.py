import numpy as np

from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms


def match_firms_with_banks(
    firms: SyntheticFirms,
    banks: SyntheticBanks,
) -> None:
    """
    Matches synthetic firms with synthetic banks based on random assignment.

    Args:
        firms (SyntheticFirms): The synthetic firms dataset.
        banks (SyntheticBanks): The synthetic banks dataset.

    Returns:
        None
    """
    bank_by_firm = np.random.choice(
        range(banks.number_of_banks),
        firms.number_of_firms,
        replace=True,
    )
    firms.firm_data["Corresponding Bank ID"] = bank_by_firm
    banks.bank_data["Corresponding Firms ID"] = [
        list(np.where(bank_by_firm == bank_id)[0]) for bank_id in range(banks.number_of_banks)
    ]
