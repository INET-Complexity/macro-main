import numpy as np
import scipy as sp

from scipy.optimize import linear_sum_assignment as lsa

from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms


def match_firms_with_banks_random(
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


def match_firms_with_banks_optimal(
    firms: SyntheticFirms,
    banks: SyntheticBanks,
) -> None:
    # rescale
    rescale(firms, "Deposits", banks, "Deposits from Firms")
    rescale(firms, "Debt", banks, "Loans to Firms")

    # create cost matrix
    # sum of loans and deposits to firms
    loans_and_deposits = banks.bank_data["Deposits from Firms"].values + banks.bank_data["Loans to Firms"].values
    # number of firms by bank
    number_of_firms_by_bank = (
        firms.number_of_firms * loans_and_deposits / loans_and_deposits.sum()
        if loans_and_deposits.sum() > 0
        else np.full(banks.number_of_banks, firms.number_of_firms / banks.number_of_banks)
    )

    # round down
    number_of_firms_by_bank = np.floor(number_of_firms_by_bank).astype(int)

    # assign firms to banks if needed
    number_of_firms_by_bank_sum = number_of_firms_by_bank.sum()

    if firms.number_of_firms - number_of_firms_by_bank_sum > 0:
        add_inds = np.random.choice(
            len(number_of_firms_by_bank),
            firms.number_of_firms - number_of_firms_by_bank.sum(),
            replace=True,
        )
        for ind in add_inds:
            number_of_firms_by_bank[ind] += 1

    # assign firms to banks

    bank_accounts = []
    for bank_id, number_of_firms in enumerate(number_of_firms_by_bank):
        book_value = (
            banks.bank_data["Deposits from Firms"].values[bank_id] + banks.bank_data["Loans to Firms"].values[bank_id]
        )
        extension = np.full(number_of_firms, book_value / number_of_firms) if number_of_firms > 0 else []
        bank_accounts.extend(extension)

    bank_accounts = np.array(bank_accounts)

    bank_by_account = np.concatenate(
        [np.full(number_of_firms_by_bank[bank_id], bank_id) for bank_id in range(banks.number_of_banks)]
    ).astype(int)

    cost = sp.spatial.distance_matrix(
        (firms.firm_data["Deposits"].values + firms.firm_data["Debt"].values)[:, None],
        bank_accounts[:, None],
    )

    # Record the optimal configuration
    _, corr_bank_accounts = lsa(cost)
    corr_banks = bank_by_account[corr_bank_accounts]
    firms.firm_data["Corresponding Bank ID"] = corr_banks
    banks.bank_data["Corresponding Firms ID"] = [
        np.where(corr_banks == bank_id) for bank_id in range(banks.number_of_banks)
    ]


def rescale(firms: SyntheticFirms, firms_field: str, banks: SyntheticBanks, banks_field: str):
    if banks.bank_data[banks_field].values.sum() == 0:
        banks.bank_data[banks_field] = np.full(
            banks.bank_data.shape[0],
            1.0 / banks.bank_data.shape[0] * firms.firm_data[firms_field].values.sum(),
        )
    else:
        banks.bank_data[banks_field] *= (
            firms.firm_data[firms_field].values.sum() / banks.bank_data[banks_field].values.sum()
        )
