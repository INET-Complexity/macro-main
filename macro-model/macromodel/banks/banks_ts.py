import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_banks_timeseries(bank_data: pd.DataFrame, long_term_ir: float, scale: int) -> TimeSeries:
    return TimeSeries(
        n_banks=len(bank_data),
        #
        equity=bank_data["Equity"].values,
        equity_histogram=get_histogram(bank_data["Equity"].values, scale),
        deposits=bank_data["Deposits"].values,
        deposits_histogram=get_histogram(bank_data["Deposits"].values, scale),
        profits=bank_data["Profits"].values,
        expected_profits=bank_data["Profits"].values,
        profits_histogram=get_histogram(bank_data["Profits"].values, scale),
        market_share=bank_data["Market Share"].values,
        market_share_histogram=get_histogram(bank_data["Market Share"].values, None),
        liability=bank_data["Liability"].values,
        liability_histogram=get_histogram(bank_data["Liability"].values, scale),
        #
        deposits_from_firms=bank_data["Deposits from Firms"].values,
        total_deposits_from_firms=[bank_data["Deposits from Firms"].values.sum()],
        deposits_from_households=bank_data["Deposits from Households"].values,
        total_deposits_from_households=[bank_data["Deposits from Households"].values.sum()],
        #
        short_term_loans_to_firms=np.zeros(len(bank_data)),
        total_short_term_loans_to_firms=[0.0],
        long_term_loans_to_firms=bank_data["Loans to Firms"].values,
        total_long_term_loans_to_firms=[bank_data["Loans to Firms"].values.sum()],
        payday_loans_to_households=np.zeros(len(bank_data)),
        total_payday_loans_to_households=[0.0],
        consumption_expansion_loans_to_households=bank_data["Consumption Loans to Households"].values,
        total_consumption_expansion_loans_to_households=[bank_data["Consumption Loans to Households"].values.sum()],
        mortgages_to_households=bank_data["Mortgages to Households"].values,
        total_mortgages_to_households=[bank_data["Mortgages to Households"].values.sum()],
        total_outstanding_loans=bank_data["Loans to Firms"].values
        + bank_data["Consumption Loans to Households"].values
        + bank_data["Mortgages to Households"].values,
        #
        interest_received_on_loans=bank_data["Interest received from Loans"].values,
        interest_received_on_deposits=bank_data["Interest received from Deposits"].values,
        interest_received=bank_data["Interest received from Loans"].values
        + bank_data["Interest received from Deposits"].values,
        #
        interest_rates_on_short_term_firm_loans=bank_data["Short-Term Interest Rates on Firm Loans"].values,
        average_interest_rates_on_short_term_firm_loans=[
            bank_data["Short-Term Interest Rates on Firm Loans"].values.mean()
        ],
        interest_rates_on_long_term_firm_loans=bank_data["Long-Term Interest Rates on Firm Loans"].values,
        average_interest_rates_on_long_term_firm_loans=[
            bank_data["Long-Term Interest Rates on Firm Loans"].values.mean()
        ],
        interest_rates_on_household_payday_loans=bank_data["Interest Rates on Household Payday Loans"].values,
        average_interest_rates_on_household_payday_loans=[
            bank_data["Interest Rates on Household Payday Loans"].values.mean()
        ],
        interest_rates_on_household_consumption_loans=bank_data["Interest Rates on Household Consumption Loans"].values,
        average_interest_rates_on_household_consumption_loans=[
            bank_data["Interest Rates on Household Consumption Loans"].values.mean()
        ],
        interest_rates_on_mortgages=bank_data["Interest Rates on Mortgages"].values,
        average_interest_rates_on_mortgages=[bank_data["Interest Rates on Mortgages"].values.mean()],
        #
        interest_rate_on_firm_deposits=bank_data["Interest Rates on Firm Deposits"].values,
        average_interest_rate_on_firm_deposits=[bank_data["Interest Rates on Firm Deposits"].values.mean()],
        overdraft_rate_on_firm_deposits=bank_data["Overdraft Rate on Firm Deposits"].values,
        average_overdraft_rate_on_firm_deposits=[bank_data["Overdraft Rate on Firm Deposits"].values.mean()],
        interest_rate_on_household_deposits=bank_data["Interest Rates on Household Deposits"].values,
        average_interest_rate_on_household_deposits=[bank_data["Interest Rates on Household Deposits"].values.mean()],
        overdraft_rate_on_household_deposits=bank_data["Overdraft Rate on Household Deposits"].values,
        average_overdraft_rate_on_household_deposits=[bank_data["Overdraft Rate on Household Deposits"].values.mean()],
        #
        interest_rate_on_government_debt=np.array([long_term_ir]),
        new_loans_fraction_firms=np.divide(
            bank_data["Loans to Firms"].values,
            bank_data["Loans to Firms"].values
            + bank_data["Consumption Loans to Households"].values
            + bank_data["Mortgages to Households"].values,
            out=np.zeros(bank_data["Loans to Firms"].values.shape),
            where=bank_data["Loans to Firms"].values
            + bank_data["Consumption Loans to Households"].values
            + bank_data["Mortgages to Households"].values
            != 0.0,
        ),
        new_loans_fraction_hh_cons=np.divide(
            bank_data["Consumption Loans to Households"].values,
            bank_data["Loans to Firms"].values
            + bank_data["Consumption Loans to Households"].values
            + bank_data["Mortgages to Households"].values,
            out=np.zeros(bank_data["Consumption Loans to Households"].values.shape),
            where=bank_data["Loans to Firms"].values
            + bank_data["Consumption Loans to Households"].values
            + bank_data["Mortgages to Households"].values
            != 0.0,
        ),
        new_loans_fraction_mortgages=np.divide(
            bank_data["Mortgages to Households"].values,
            bank_data["Loans to Firms"].values
            + bank_data["Consumption Loans to Households"].values
            + bank_data["Mortgages to Households"].values,
            out=np.zeros(bank_data["Mortgages to Households"].values.shape),
            where=bank_data["Loans to Firms"].values
            + bank_data["Consumption Loans to Households"].values
            + bank_data["Mortgages to Households"].values
            != 0.0,
        ),
        #
    )
