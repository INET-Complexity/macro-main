"""Module for preprocessing default credit market relationship data.

This module provides utility functions for preprocessing credit relationship data
between banks and borrowers. Key preprocessing includes:

1. Firm Loan Data:
   - Long-term loan preprocessing
   - Initial loan value calculations
   - Bank-firm relationship mapping

2. Household Loan Data:
   - Consumer loan preprocessing
   - Mortgage loan preprocessing
   - Bank-household relationship mapping

3. Parameter Processing:
   - Interest rate application
   - Maturity period setting
   - Initial state organization

Note:
    This module is NOT used for simulating credit market behavior. It only handles
    the preprocessing and organization of data that will later be used to initialize
    behavioral models in the simulation package.
"""

import pandas as pd

from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


def create_firm_loan_df(firms: SyntheticFirms, banks: SyntheticBanks, firm_loan_maturity: int = 60) -> pd.DataFrame:
    """Preprocess firm loan relationship data.

    This function organizes initial firm loan data by:
    1. Identifying firms with debt
    2. Matching with corresponding banks
    3. Setting initial loan parameters

    The preprocessed data includes:
    - Loan type (2 for firm loans)
    - Initial and current loan values
    - Bank-firm relationships
    - Interest rates
    - Maturity periods

    Args:
        firms (SyntheticFirms): Firm data container
        banks (SyntheticBanks): Bank data container
        firm_loan_maturity (int, optional): Initial maturity. Defaults to 60.

    Returns:
        pd.DataFrame: Preprocessed firm loan relationship data
    """
    selection = firms.firm_data["Debt"] > 0
    data_sel = firms.firm_data.loc[selection]

    loan_df = pd.DataFrame(index=data_sel.index)
    loan_df["loan_type"] = 2
    loan_df["loan_value_initial"] = data_sel["Debt"]
    loan_df["loan_value"] = data_sel["Debt"]
    loan_df["loan_bank_id"] = data_sel["Corresponding Bank ID"]
    loan_df["loan_recipient_id"] = data_sel.index

    loan_df = pd.merge(
        left=loan_df,
        right=banks.bank_data["Long-Term Interest Rates on Firm Loans"],
        left_on="loan_bank_id",
        right_index=True,
    )

    loan_df.rename(columns={"Long-Term Interest Rates on Firm Loans": "loan_interest_rate"}, inplace=True)
    loan_df["loan_maturity"] = firm_loan_maturity
    return loan_df


def create_household_loan_df(
    synthetic_population: SyntheticPopulation, synthetic_banks: SyntheticBanks, consumption_loan_maturity: int = 12
) -> pd.DataFrame:
    """Preprocess household consumer loan relationship data.

    This function organizes initial household loan data by:
    1. Identifying households with non-mortgage debt
    2. Matching with corresponding banks
    3. Setting initial loan parameters

    The preprocessed data includes:
    - Loan type (4 for consumer loans)
    - Initial and current loan values
    - Bank-household relationships
    - Interest rates
    - Maturity periods

    Args:
        synthetic_population (SyntheticPopulation): Population data container
        synthetic_banks (SyntheticBanks): Bank data container
        consumption_loan_maturity (int, optional): Initial maturity. Defaults to 12.

    Returns:
        pd.DataFrame: Preprocessed household loan relationship data
    """
    debt_col = "Outstanding Balance of other Non-Mortgage Loans"
    selection = synthetic_population.household_data[debt_col] > 0
    data_sel = synthetic_population.household_data.loc[selection]

    loan_df = pd.DataFrame(index=data_sel.index)
    loan_df["loan_type"] = 4
    loan_df["loan_value_initial"] = data_sel[debt_col]
    loan_df["loan_value"] = data_sel[debt_col]
    loan_df["loan_bank_id"] = data_sel["Corresponding Bank ID"]
    loan_df["loan_recipient_id"] = data_sel.index

    loan_df = pd.merge(
        left=loan_df,
        right=synthetic_banks.bank_data["Interest Rates on Household Consumption Loans"],
        left_on="loan_bank_id",
        right_index=True,
    )

    loan_df.rename(columns={"Interest Rates on Household Consumption Loans": "loan_interest_rate"}, inplace=True)
    loan_df["loan_maturity"] = consumption_loan_maturity
    return loan_df


def create_mortgage_loan_df(
    synthetic_population: SyntheticPopulation, synthetic_banks: SyntheticBanks, mortgage_loan_maturity: int = 120
) -> pd.DataFrame:
    """Preprocess household mortgage loan relationship data.

    This function organizes initial mortgage data by:
    1. Identifying households with mortgage debt
    2. Matching with corresponding banks
    3. Setting initial loan parameters

    The preprocessed data includes:
    - Loan type (5 for mortgages)
    - Initial and current loan values
    - Bank-household relationships
    - Interest rates
    - Maturity periods

    Args:
        synthetic_population (SyntheticPopulation): Population data container
        synthetic_banks (SyntheticBanks): Bank data container
        mortgage_loan_maturity (int, optional): Initial maturity. Defaults to 120.

    Returns:
        pd.DataFrame: Preprocessed mortgage relationship data
    """
    debt_columns = ["Outstanding Balance of HMR Mortgages", "Outstanding Balance of Mortgages on other Properties"]

    total_debt = synthetic_population.household_data[debt_columns].sum(axis=1)
    selection = total_debt > 0
    data_sel = synthetic_population.household_data.loc[selection]

    loan_df = pd.DataFrame(index=data_sel.index)
    loan_df["loan_type"] = 5
    loan_df["loan_value_initial"] = data_sel[debt_columns].sum(axis=1)
    loan_df["loan_value"] = data_sel[debt_columns].sum(axis=1)
    loan_df["loan_bank_id"] = data_sel["Corresponding Bank ID"]
    loan_df["loan_recipient_id"] = data_sel.index

    loan_df = pd.merge(
        left=loan_df,
        right=synthetic_banks.bank_data["Interest Rates on Mortgages"],
        left_on="loan_bank_id",
        right_index=True,
    )

    loan_df.rename(columns={"Interest Rates on Mortgages": "loan_interest_rate"}, inplace=True)
    loan_df["loan_maturity"] = mortgage_loan_maturity
    return loan_df
