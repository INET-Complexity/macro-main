import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_households_timeseries(
    data: pd.DataFrame,
    initial_consumption_by_industry: np.ndarray,
    initial_hh_investment: np.ndarray,
    initial_investment_by_industry: np.ndarray,
    initial_hh_consumption: np.ndarray,
    scale: int,
    vat: float,
    tau_cf: float,
) -> TimeSeries:
    n_industries = len(initial_consumption_by_industry)
    return TimeSeries(
        n_households=len(data),
        #
        target_consumption_loans=initial_hh_consumption,
        total_target_consumption_loans=[0.0],
        target_consumption=np.full((len(data), n_industries), np.nan),
        amount_bought=np.full(len(data), np.nan),
        consumption=np.full(len(data), np.nan),
        total_consumption=[(1 + vat) * initial_consumption_by_industry.sum()],
        total_consumption_before_vat=[initial_consumption_by_industry.sum()],
        industry_consumption=initial_consumption_by_industry,
        #
        target_investment=initial_hh_investment,
        investment=initial_hh_investment,
        total_investment=[(1 + tau_cf) * initial_hh_investment.sum()],
        total_investment_before_vat=[initial_hh_investment.sum()],
        industry_investment=initial_investment_by_industry,
        #
        income=data["Income"].values,
        income_histogram=get_histogram(data["Income"].values, scale),
        expected_income=data["Income"].values,
        income_employee=data["Employee Income"].values,
        total_income_employee=[data["Employee Income"].values.sum()],
        expected_income_employee=data["Employee Income"].values,
        income_social_transfers=data["Regular Social Transfers"].values,
        total_income_social_transfers=[data["Regular Social Transfers"].values.sum()],
        expected_income_social_transfers=data["Regular Social Transfers"].values,
        income_rental=data["Rental Income from Real Estate"].values,
        total_income_rental=[data["Rental Income from Real Estate"].values.sum()],
        income_financial_assets=data["Income from Financial Assets"].values,
        total_income_financial_assets=[data["Income from Financial Assets"].values.sum()],
        expected_income_financial_assets=data["Income from Financial Assets"].values,
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
        mortgage_debt=data["Outstanding Balance of HMR Mortgages"].values
        + data["Outstanding Balance of Mortgages on other Properties"].values,
        total_mortgage_debt=[
            np.sum(
                data["Outstanding Balance of HMR Mortgages"].values
                + data["Outstanding Balance of Mortgages on other Properties"].values
            )
        ],
        consumption_loan_debt=data["Outstanding Balance of other Non-Mortgage Loans"].values,
        received_consumption_loans=np.full(len(data), np.nan),
        total_consumption_loan_debt=[np.sum(data["Outstanding Balance of other Non-Mortgage Loans"].values)],
        debt=data["Debt"].values,
        total_received_consumption_loans=[0.0],
        debt_histogram=get_histogram(data["Debt"].values, scale),
        #
        net_wealth=data["Net Wealth"].values,
        #
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
