import numpy as np
import pandas as pd

from model.timeseries import TimeSeries
from model.util.get_histogram import get_histogram


def create_households_timeseries(
    data: pd.DataFrame,
    initial_industry_consumption: np.ndarray,
    n_industries: int,
    scale: int,
    vat: float,
) -> TimeSeries:
    return TimeSeries(
        n_households=len(data),
        #
        target_consumption_before_ce=np.full((len(data), n_industries), np.nan),
        target_consumption_ce=np.full(len(data), np.nan),
        target_consumption=np.full((len(data), n_industries), np.nan),
        amount_bought=np.full(len(data), np.nan),
        consumption=np.full(len(data), np.nan),
        total_consumption=[(1 + vat) * initial_industry_consumption.sum()],
        industry_consumption=initial_industry_consumption,
        investment_in_other_real_assets=np.full(len(data), np.nan),
        total_investment_in_other_real_assets=[0.0],
        #
        income=data["Income"].values,
        income_histogram=get_histogram(data["Income"].values, scale),
        income_employee=data["Employee Income"].values,
        total_income_employee=[data["Employee Income"].values.sum()],
        income_social_transfers=data["Regular Social Transfers"].values,
        total_income_social_transfers=[data["Regular Social Transfers"].values.sum()],
        income_rental=data["Rental Income from Real Estate"].values,
        total_income_rental=[data["Rental Income from Real Estate"].values.sum()],
        income_financial_assets=data["Income from Financial Assets"].values,
        total_income_financial_assets=[data["Income from Financial Assets"].values.sum()],
        #
        price_paid_for_property=np.full(len(data), np.nan),
        rent=data["Rent Paid"].values,
        rent_imputed=data["Rent Imputed"].values,
        max_price_willing_to_pay=np.full(len(data), np.nan),
        max_rent_willing_to_pay=np.full(len(data), np.nan),
        rent_div_income_histogram=get_histogram(data["Rent Paid"].values / data["Income"].values, None),
        #
        wealth=data["Wealth"].values,
        wealth_histogram=get_histogram(data["Wealth"].values, scale),
        wealth_real_assets=data["Wealth in Real Assets"].values,
        wealth_main_residence=data["Value of the Main Residence"].values,
        total_wealth_main_residence=[np.sum(data["Value of the Main Residence"].values)],
        wealth_other_properties=data["Value of other Properties"].values,
        total_wealth_other_properties=[np.sum(data["Value of other Properties"].values)],
        wealth_other_real_assets=data["Wealth Other Real Assets"].values,
        total_wealth_other_real_assets=[np.sum(data["Wealth Other Real Assets"].values)],
        wealth_deposits=data["Wealth in Deposits"].values,
        total_wealth_deposits=[np.sum(data["Wealth in Deposits"].values)],
        wealth_other_financial_assets=data["Wealth in Other Financial Assets"].values,
        total_wealth_other_financial_assets=[np.sum(data["Wealth in Other Financial Assets"].values)],
        wealth_financial_assets=data["Wealth in Financial Assets"].values,
        #
        payday_loan_debt=np.zeros(len(data)),
        total_payday_loan_debt=[0.0],
        consumption_expansion_loan_debt=data["Outstanding Balance of other Non-Mortgage Loans"].values,
        total_consumption_expansion_loan_debt=[np.sum(data["Outstanding Balance of other Non-Mortgage Loans"].values)],
        mortgage_debt=data["Outstanding Balance of HMR Mortgages"].values
        + data["Outstanding Balance of Mortgages on other Properties"].values,
        total_mortgage_debt=[
            np.sum(
                data["Outstanding Balance of HMR Mortgages"].values
                + data["Outstanding Balance of Mortgages on other Properties"].values
            )
        ],
        debt=data["Debt"].values,
        debt_histogram=get_histogram(data["Debt"].values, scale),
        #
        net_wealth=data["Net Wealth"].values,
        #
        target_payday_loans=np.full(len(data), np.nan),
        total_target_payday_loans=[0.0],
        received_payday_loans=np.full(len(data), np.nan),
        total_received_payday_loans=[0.0],
        target_consumption_expansion_loans=np.full(len(data), np.nan),
        total_target_consumption_expansion_loans=[0.0],
        received_consumption_expansion_loans=np.full(len(data), np.nan),
        total_received_consumption_expansion_loans=[0.0],
        target_mortgage=np.full(len(data), np.nan),
        total_target_mortgage=[0.0],
        received_mortgages=np.full(len(data), np.nan),
        total_received_mortgages=[0.0],
        #
        debt_installments=data["Debt Installments"].values,
        total_debt_installments=[data["Debt Installments"].values.sum()],
        #
        interest_paid_on_deposits=np.full(len(data), np.nan),
        interest_paid_on_loans=np.full(len(data), np.nan),
        interest_paid=np.full(len(data), np.nan),
    )
