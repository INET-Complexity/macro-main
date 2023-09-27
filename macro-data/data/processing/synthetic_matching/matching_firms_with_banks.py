import numpy as np

from data.processing.synthetic_banks.synthetic_banks import (
    SyntheticBanks,
)
from data.processing.synthetic_firms.synthetic_firms import (
    SyntheticFirms,
)


def match_firms_with_banks(
    synthetic_firms: SyntheticFirms,
    synthetic_banks: SyntheticBanks,
) -> None:
    bank_by_firm = np.random.choice(
        range(synthetic_banks.number_of_banks),
        synthetic_firms.number_of_firms,
        replace=True,
    )
    synthetic_firms.firm_data["Corresponding Bank ID"] = bank_by_firm
    synthetic_banks.bank_data["Corresponding Firms ID"] = [
        list(np.where(bank_by_firm == bank_id)[0]) for bank_id in range(synthetic_banks.number_of_banks)
    ]
