import numpy as np
import pandas as pd

from macromodel.credit_market.types_of_loans import LoanTypes
from macromodel.timeseries import TimeSeries


def create_credit_market_timeseries(
    total_short_term_loans: float,
    total_long_term_loans: float,
    total_payday_loans: float,
    total_consumption_expansion_loans: float,
    total_mortgage_loans: float,
) -> TimeSeries:
    return TimeSeries(
        total_newly_loans_granted_firms_short_term=[np.nan],
        total_newly_loans_granted_firms_long_term=[np.nan],
        total_newly_loans_granted_households_consumption=[np.nan],
        total_newly_loans_granted_mortgages=[np.nan],
        #
        total_outstanding_loans_granted_firms_short_term=[total_short_term_loans],
        total_outstanding_loans_granted_firms_long_term=[total_long_term_loans],
        total_outstanding_loans_granted_households_consumption=[total_consumption_expansion_loans],
        total_outstanding_loans_granted_mortgages=[total_mortgage_loans],
    )
