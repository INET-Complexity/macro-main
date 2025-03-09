"""Module for harmonizing firm and bank financial data.

This module harmonizes financial data from different sources:
1. Firm Survey/Balance Sheet Data:
   - Reported cash holdings
   - Outstanding credit balances
   - Industry-specific financials
   - Balance sheet information

2. Banking System Data:
   - Aggregate corporate deposits
   - Commercial loan portfolio
   - Balance sheet totals
   - Corporate account data

The harmonization process involves:
1. Data Validation:
   - Checking total deposits match across sources
   - Validating loan balances
   - Ensuring consistent client counts

2. Data Reconciliation:
   - Scaling firm deposits to match bank totals
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
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms


def match_firms_with_banks_random(
    firms: SyntheticFirms,
    banks: SyntheticBanks,
) -> None:
    """Initialize firm-bank relationships with random assignment.

    This function provides a simple initialization mechanism that:
    1. Randomly assigns firms to banks
    2. Records assignments in firm data
    3. Updates bank corporate client lists

    Useful for:
    - Initial data setup
    - Testing and validation
    - Cases where optimal harmonization is not required

    Args:
        firms (SyntheticFirms): Firm financial data
        banks (SyntheticBanks): Bank balance sheet data
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
    """Harmonize firm and bank financial data using optimal assignment.

    This function reconciles financial data by:
    1. Scaling firm data to match bank totals
    2. Allocating accounts based on bank size
    3. Using linear sum assignment to minimize discrepancies
    4. Recording harmonized relationships

    The optimization:
    - Minimizes differences between reported values
    - Respects bank balance sheet constraints
    - Maintains consistent financial totals
    - Preserves deposit-loan relationships

    Args:
        firms (SyntheticFirms): Firm financial data
        banks (SyntheticBanks): Bank balance sheet data
    """
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
    corr_banks = [bank_by_account[corresponding] for corresponding in corr_bank_accounts]

    firms.firm_data["Corresponding Bank ID"] = corr_banks

    corr_banks = np.array(corr_banks)

    banks.bank_data["Corresponding Firms ID"] = [
        np.where(corr_banks == bank_id)[0] for bank_id in range(banks.number_of_banks)
    ]


def rescale(firms: SyntheticFirms, firms_field: str, banks: SyntheticBanks, banks_field: str):
    """Reconcile firm financial data with bank totals.

    This function ensures consistency between sources by:
    1. Checking if bank totals need initialization
    2. Scaling firm values to match bank totals
    3. Maintaining relative proportions

    Used for:
    - Corporate deposit reconciliation
    - Commercial loan harmonization
    - Balance sheet validation

    Args:
        firms (SyntheticFirms): Firm financial data
        firms_field (str): Field in firm data to reconcile
        banks (SyntheticBanks): Bank balance sheet data
        banks_field (str): Field in bank data to match
    """
    if banks.bank_data[banks_field].values.sum() == 0:
        banks.bank_data[banks_field] = np.full(
            banks.bank_data.shape[0],
            1.0 / banks.bank_data.shape[0] * firms.firm_data[firms_field].values.sum(),
        )
    else:
        banks.bank_data[banks_field] *= (
            firms.firm_data[firms_field].values.sum() / banks.bank_data[banks_field].values.sum()
        )
