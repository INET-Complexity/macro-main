import numpy as np

from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms


def match_firms_with_banks(
    synthetic_firms: SyntheticFirms,
    synthetic_banks: SyntheticBanks,
) -> None:
    """
    Matches synthetic firms with synthetic banks based on random assignment.

    Args:
        synthetic_firms (SyntheticFirms): The synthetic firms dataset.
        synthetic_banks (SyntheticBanks): The synthetic banks dataset.

    Returns:
        None
    """
    bank_by_firm = np.random.choice(
        range(synthetic_banks.number_of_banks),
        synthetic_firms.number_of_firms,
        replace=True,
    )
    synthetic_firms.firm_data["Corresponding Bank ID"] = bank_by_firm
    synthetic_banks.bank_data["Corresponding Firms ID"] = [
        list(np.where(bank_by_firm == bank_id)[0]) for bank_id in range(synthetic_banks.number_of_banks)
    ]
