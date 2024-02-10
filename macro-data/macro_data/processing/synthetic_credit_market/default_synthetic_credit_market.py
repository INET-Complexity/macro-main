import pandas as pd

from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_population.synthetic_population import SyntheticPopulation


def create_firm_loan_df(firms: SyntheticFirms, banks: SyntheticBanks, firm_loan_maturity: int = 60) -> pd.DataFrame:
    """
    Create a DataFrame of firm loans.
    Loans are created for all firms with debt, and are set with a loan status of 2 (firm loan).

    The corresponding loan value is set to the firm's debt, and the loan interest rate is set to the long-term
    loan of the corresponding bank.

    Loan maturity is set exogenously (default: 60 months).

    Args:
        firms (SyntheticFirms): Object containing firm data.
        banks (SyntheticBanks): Object containing bank data.
        firm_loan_maturity (int, optional): Loan maturity in months. Defaults to 60.

    Returns:
        pd.DataFrame: DataFrame containing firm loan information.
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
    """
    Create a DataFrame of household loans based on synthetic population data and synthetic bank data.

    Loans are created for all households with debt, and are set with a loan status of 4 (household loan).

    The corresponding loan value is set to the household's debt, and the loan interest rate is set to the consumption
    loan interest rate of the corresponding bank.

    Loan maturity is set exogenously (default: 12 months).

    Parameters:
        synthetic_population (SyntheticPopulation): Object containing synthetic population data.
        synthetic_banks (SyntheticBanks): Object containing synthetic bank data.
        consumption_loan_maturity (int, optional): Maturity period of the loans in months. Defaults to 12.

    Returns:
        pd.DataFrame: DataFrame containing household loan information.
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
    """
    Create a DataFrame of mortgage loan data based on the synthetic population and synthetic banks.

    Loans are created for all households with mortgage debt (including mortgages on their own property and mortgages on rental properties),
    and are set with a loan status of 5 (mortgage loan).

    The corresponding loan value is set to the household's mortgage debt, and the loan interest rate is set to the mortgage loan interest rate
    of the corresponding bank.

    Parameters:
        synthetic_population (SyntheticPopulation): The synthetic population data.
        synthetic_banks (SyntheticBanks): The synthetic banks data.
        mortgage_loan_maturity (int, optional): The maturity period of the mortgage loan in months. Defaults to 120.

    Returns:
        pd.DataFrame: The DataFrame containing the mortgage loan data.
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
