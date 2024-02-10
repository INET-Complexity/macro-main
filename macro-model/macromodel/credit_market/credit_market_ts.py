import numpy as np
import pandas as pd

from macromodel.credit_market.types_of_loans import LoanTypes
from macromodel.timeseries import TimeSeries


def create_credit_market_timeseries(loan_data: pd.DataFrame) -> TimeSeries:
    return TimeSeries(
        total_newly_loans_granted_firms_short_term=[np.nan],
        total_newly_loans_granted_firms_long_term=[np.nan],
        total_newly_loans_granted_households_payday=[np.nan],
        total_newly_loans_granted_households_consumption_expansion=[np.nan],
        total_newly_loans_granted_mortgages=[np.nan],
        #
        total_outstanding_loans_granted_firms_short_term=[
            loan_data.loc[
                loan_data["loan_type"] == LoanTypes.FIRM_SHORT_TERM_LOAN,
                "loan_value",
            ].sum()
        ],
        total_outstanding_loans_granted_firms_long_term=[
            loan_data.loc[
                loan_data["loan_type"] == LoanTypes.FIRM_LONG_TERM_LOAN,
                "loan_value",
            ].sum()
        ],
        total_outstanding_loans_granted_households_payday=[
            loan_data.loc[
                loan_data["loan_type"] == LoanTypes.HOUSEHOLD_PAYDAY_LOAN,
                "loan_value",
            ].sum()
        ],
        total_outstanding_loans_granted_households_consumption_expansion=[
            loan_data.loc[
                loan_data["loan_type"] == LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN,
                "loan_value",
            ].sum()
        ],
        total_outstanding_loans_granted_mortgages=[
            loan_data.loc[
                loan_data["loan_type"] == LoanTypes.MORTGAGE,
                "loan_value",
            ].sum()
        ],
    )
