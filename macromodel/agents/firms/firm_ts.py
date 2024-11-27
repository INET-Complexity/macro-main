from collections import Counter

import numpy as np
import pandas as pd

from macromodel.agents.firms.utils import calculate_tail_exponent
from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


class FirmTimeSeries(TimeSeries):

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        intermediate_inputs_stock: np.ndarray,
        used_intermediate_inputs: np.ndarray,
        capital_inputs_stock: np.ndarray,
        used_capital_inputs: np.ndarray,
        initial_good_prices: np.ndarray,
        n_industries: int,
        calculate_hill_exponent: bool = False,
    ) -> "FirmTimeSeries":
        gross_operating_surplus_mixed_income = (
            data["Price"].values * (data["Production"].values + data["Inventory"].values)
            - data["Total Wages Paid"].values
            - np.matmul(used_intermediate_inputs, initial_good_prices)
            - data["Taxes paid on Production"].values
        )
        return cls(
            n_firms=data.shape[0],
            limiting_intermediate_inputs=np.full(data.shape[0], np.nan),
            limiting_capital_inputs=np.full(data.shape[0], np.nan),
            target_intermediate_inputs_production=np.full(data.shape[0], np.nan),
            target_capital_inputs_production=np.full(data.shape[0], np.nan),
            wage_tightness_markup=np.full(data.shape[0], np.nan),
            n_firms_by_industry=get_n_firms_by_industry(data, n_industries),
            number_of_employees=data["Number of Employees"].values.astype(int),
            number_of_employees_histogram=get_histogram(data["Number of Employees"].values.astype(int), None),
            #
            production=data["Production"].values,
            production_histogram=get_histogram(data["Production"].values, None),
            production_nominal=data["Price"].values * data["Production"].values,
            output_by_employee_histogram=get_histogram(
                data["Production"].values / data["Number of Employees"].values, None
            ),
            target_production=np.full(data.shape[0], np.nan),
            constrained_intermediate_inputs_target_production=np.full(len(data), np.nan),
            constrained_capital_inputs_target_production=np.full(len(data), np.nan),
            #
            price=data["Price"].values,
            price_offered=np.full(n_industries, 1.0),
            price_in_usd=data["Price in USD"].values,
            profits=data["Profits"].values,
            expected_profits=data["Profits"].values,
            total_wage=data["Total Wages Paid"].values,
            real_wage_per_capita=data["Total Wages Paid"].values / data["Number of Employees"].values,
            unit_costs=data["Unit Costs"].values,
            taxes_paid_on_production=data["Taxes paid on Production"].values,
            corporate_taxes_paid=data["Corporate Taxes Paid"].values,
            equity=data["Equity"].values,
            #
            estimated_demand=data["Demand"].values,
            demand=data["Demand"].values.copy(),
            #
            unconstrained_target_intermediate_inputs=np.full((data.shape[0], n_industries), np.nan),
            unconstrained_target_intermediate_inputs_costs=np.full(data.shape[0], np.nan),
            unconstrained_target_capital_inputs=np.full((data.shape[0], n_industries), np.nan),
            unconstrained_target_capital_inputs_costs=np.full(data.shape[0], np.nan),
            target_intermediate_inputs=used_intermediate_inputs,
            target_capital_inputs=used_capital_inputs,
            #
            inventory=data["Inventory"].values,
            inventory_nominal=data["Price"].values * data["Inventory"].values,
            total_inventory_change=np.zeros(data.shape[0]),
            #
            intermediate_inputs_stock=intermediate_inputs_stock,
            intermediate_inputs_stock_value=np.matmul(intermediate_inputs_stock, initial_good_prices),
            intermediate_inputs_stock_industry=intermediate_inputs_stock.sum(axis=0),
            used_intermediate_inputs=used_intermediate_inputs,
            used_intermediate_inputs_costs=np.matmul(used_intermediate_inputs, initial_good_prices),
            total_intermediate_inputs_bought_costs=np.matmul(used_intermediate_inputs, initial_good_prices),
            #
            capital_inputs_stock=capital_inputs_stock,
            capital_inputs_stock_value=np.matmul(capital_inputs_stock, initial_good_prices),
            capital_inputs_stock_industry=capital_inputs_stock.sum(axis=0),
            expected_capital_inputs_stock_value=np.matmul(capital_inputs_stock, initial_good_prices),
            used_capital_inputs=used_capital_inputs,
            used_capital_inputs_costs=np.matmul(used_capital_inputs, initial_good_prices),
            total_capital_inputs_bought_costs=np.matmul(used_capital_inputs, initial_good_prices),
            gross_fixed_capital_formation=(used_capital_inputs * initial_good_prices).sum(axis=0),
            #
            real_amount_bought_as_intermediate_inputs=np.full((data.shape[0], n_industries), np.nan),
            real_amount_bought_as_capital_goods=np.full((data.shape[0], n_industries), np.nan),
            total_sales=data["Price"].values * data["Production"].values - data["Taxes paid on Production"].values,
            #
            target_short_term_credit=np.zeros(data.shape[0]),
            total_target_short_term_credit=[0.0],
            target_long_term_credit=np.zeros(data.shape[0]),
            total_target_long_term_credit=[0.0],
            received_short_term_credit=np.full(data.shape[0], np.nan),
            total_received_short_term_credit=[0.0],
            received_long_term_credit=np.full(data.shape[0], np.nan),
            total_received_long_term_credit=[0.0],
            received_credit=np.full(data.shape[0], np.nan),
            #
            short_term_loan_debt=np.zeros(data.shape[0]),
            long_term_loan_debt=data["Debt"].values,
            debt=data["Debt"].values,
            deposits=data["Deposits"].values,
            debt_installments=data["Debt Installments"].values,
            total_debt_installments=[data["Debt Installments"].values.sum()],
            interest_paid_on_deposits=data["Interest paid on deposits"].values,
            interest_paid_on_loans=data["Interest paid on loans"].values,
            interest_paid=data["Interest paid"].values,
            #
            total_debt=[data["Debt"].sum()],
            total_deposits=[data["Deposits"].sum()],
            #
            estimated_growth_by_firm=np.full(data.shape[0], np.nan),
            labour_inputs=data["Labour Inputs"].values,
            labour_productivity=data["Labour Productivity"].values,
            labour_productivity_factor=np.ones(data.shape[0]),
            normalised_labour_inputs=data["Labour Inputs"].values,
            desired_labour_inputs=data["Labour Inputs"].values,
            labour_costs=np.full(data.shape[0], np.nan),
            #
            gross_operating_surplus_mixed_income=gross_operating_surplus_mixed_income,
            #
            total_hill_tail_estimate_production=[
                0.0 if not calculate_hill_exponent else calculate_tail_exponent(data["Production"].values.copy())
            ],
            total_hill_tail_estimate_number_of_employees=[
                0.0 if not calculate_hill_exponent else data["Number of Employees"].values.astype(int).copy()
            ],
            total_hill_tail_estimate_output_by_employee=[
                (
                    0.0
                    if not calculate_hill_exponent
                    else calculate_tail_exponent(data["Production"].values / data["Number of Employees"].values.copy())
                )
            ],
        )

    def reset_values(
        self,
        inventory: np.ndarray,
        initial_good_prices: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_stock: np.ndarray,
    ):
        self.dicts["inventory"] = [inventory]
        self.dicts["inventory_nominal"] = [self.current("price") * inventory]

        self.dicts["intermediate_inputs_stock"] = [intermediate_inputs_stock]
        self.dicts["intermediate_inputs_stock_value"] = [np.matmul(intermediate_inputs_stock, initial_good_prices)]
        self.dicts["intermediate_inputs_stock_industry"] = [intermediate_inputs_stock.sum(axis=0)]

        self.dicts["capital_inputs_stock"] = [capital_inputs_stock]
        self.dicts["capital_inputs_stock_value"] = [np.matmul(capital_inputs_stock, initial_good_prices)]
        self.dicts["capital_inputs_stock_industry"] = [capital_inputs_stock.sum(axis=0)]

        equity = (
            self.current("deposits")
            + self.current("price")
            * (
                self.current("intermediate_inputs_stock").sum(axis=1)
                + self.current("capital_inputs_stock").sum(axis=1)
                + inventory
            )
            - self.current("debt")
        )

        self.dicts["equity"] = [equity]

        gross_operating_surplus = (
            self.current("price") * (self.current("production") + inventory)
            - self.current("total_wage")
            - np.matmul(self.current("used_intermediate_inputs"), initial_good_prices)
            - self.current("taxes_paid_on_production")
        )

        self.dicts["gross_operating_surplus_mixed_income"] = [gross_operating_surplus]


def create_firms_timeseries(
    data: pd.DataFrame,
    intermediate_inputs_stock: np.ndarray,
    used_intermediate_inputs: np.ndarray,
    capital_inputs_stock: np.ndarray,
    used_capital_inputs: np.ndarray,
    initial_good_prices: np.ndarray,
    n_industries: int,
    calculate_hill_exponent: bool = False,
) -> TimeSeries:
    gross_operating_surplus_mixed_income = (
        data["Price"].values * (data["Production"].values + data["Inventory"].values)
        - data["Total Wages Paid"].values
        - np.matmul(used_intermediate_inputs, initial_good_prices)
        - data["Taxes paid on Production"].values
    )
    return TimeSeries(
        n_firms=data.shape[0],
        limiting_intermediate_inputs=np.full(data.shape[0], np.nan),
        limiting_capital_inputs=np.full(data.shape[0], np.nan),
        target_intermediate_inputs_production=np.full(data.shape[0], np.nan),
        target_capital_inputs_production=np.full(data.shape[0], np.nan),
        wage_tightness_markup=np.full(data.shape[0], np.nan),
        n_firms_by_industry=get_n_firms_by_industry(data, n_industries),
        number_of_employees=data["Number of Employees"].values.astype(int),
        number_of_employees_histogram=get_histogram(data["Number of Employees"].values.astype(int), None),
        #
        production=data["Production"].values,
        production_histogram=get_histogram(data["Production"].values, None),
        production_nominal=data["Price"].values * data["Production"].values,
        output_by_employee_histogram=get_histogram(
            data["Production"].values / data["Number of Employees"].values, None
        ),
        target_production=np.full(data.shape[0], np.nan),
        constrained_intermediate_inputs_target_production=np.full(len(data), np.nan),
        constrained_capital_inputs_target_production=np.full(len(data), np.nan),
        #
        price=data["Price"].values,
        price_offered=np.full(n_industries, 1.0),
        price_in_usd=data["Price in USD"].values,
        profits=data["Profits"].values,
        expected_profits=data["Profits"].values,
        total_wage=data["Total Wages Paid"].values,
        real_wage_per_capita=data["Total Wages Paid"].values / data["Number of Employees"].values,
        unit_costs=data["Unit Costs"].values,
        taxes_paid_on_production=data["Taxes paid on Production"].values,
        corporate_taxes_paid=data["Corporate Taxes Paid"].values,
        equity=data["Equity"].values,
        #
        estimated_demand=data["Demand"].values,
        demand=data["Demand"].values.copy(),
        #
        unconstrained_target_intermediate_inputs=np.full((data.shape[0], n_industries), np.nan),
        unconstrained_target_intermediate_inputs_costs=np.full(data.shape[0], np.nan),
        unconstrained_target_capital_inputs=np.full((data.shape[0], n_industries), np.nan),
        unconstrained_target_capital_inputs_costs=np.full(data.shape[0], np.nan),
        target_intermediate_inputs=used_intermediate_inputs,
        target_capital_inputs=used_capital_inputs,
        #
        inventory=data["Inventory"].values,
        inventory_nominal=data["Price"].values * data["Inventory"].values,
        total_inventory_change=np.zeros(data.shape[0]),
        #
        intermediate_inputs_stock=intermediate_inputs_stock,
        intermediate_inputs_stock_value=np.matmul(intermediate_inputs_stock, initial_good_prices),
        intermediate_inputs_stock_industry=intermediate_inputs_stock.sum(axis=0),
        used_intermediate_inputs=used_intermediate_inputs,
        used_intermediate_inputs_costs=np.matmul(used_intermediate_inputs, initial_good_prices),
        total_intermediate_inputs_bought_costs=np.matmul(used_intermediate_inputs, initial_good_prices),
        #
        capital_inputs_stock=capital_inputs_stock,
        capital_inputs_stock_value=np.matmul(capital_inputs_stock, initial_good_prices),
        capital_inputs_stock_industry=capital_inputs_stock.sum(axis=0),
        expected_capital_inputs_stock_value=np.matmul(capital_inputs_stock, initial_good_prices),
        used_capital_inputs=used_capital_inputs,
        used_capital_inputs_costs=np.matmul(used_capital_inputs, initial_good_prices),
        total_capital_inputs_bought_costs=np.matmul(used_capital_inputs, initial_good_prices),
        gross_fixed_capital_formation=(used_capital_inputs * initial_good_prices).sum(axis=0),
        #
        real_amount_bought_as_intermediate_inputs=np.full((data.shape[0], n_industries), np.nan),
        real_amount_bought_as_capital_goods=np.full((data.shape[0], n_industries), np.nan),
        total_sales=data["Price"].values * data["Production"].values - data["Taxes paid on Production"].values,
        #
        target_short_term_credit=np.zeros(data.shape[0]),
        total_target_short_term_credit=[0.0],
        target_long_term_credit=np.zeros(data.shape[0]),
        total_target_long_term_credit=[0.0],
        received_short_term_credit=np.full(data.shape[0], np.nan),
        total_received_short_term_credit=[0.0],
        received_long_term_credit=np.full(data.shape[0], np.nan),
        total_received_long_term_credit=[0.0],
        received_credit=np.full(data.shape[0], np.nan),
        #
        short_term_loan_debt=np.zeros(data.shape[0]),
        long_term_loan_debt=data["Debt"].values,
        debt=data["Debt"].values,
        deposits=data["Deposits"].values,
        debt_installments=data["Debt Installments"].values,
        total_debt_installments=[data["Debt Installments"].values.sum()],
        interest_paid_on_deposits=data["Interest paid on deposits"].values,
        interest_paid_on_loans=data["Interest paid on loans"].values,
        interest_paid=data["Interest paid"].values,
        #
        total_debt=[data["Debt"].sum()],
        total_deposits=[data["Deposits"].sum()],
        #
        estimated_growth_by_firm=np.full(data.shape[0], np.nan),
        labour_inputs=data["Labour Inputs"].values,
        labour_productivity=data["Labour Productivity"].values,
        labour_productivity_factor=np.ones(data.shape[0]),
        normalised_labour_inputs=data["Labour Inputs"].values,
        desired_labour_inputs=data["Labour Inputs"].values,
        labour_costs=np.full(data.shape[0], np.nan),
        #
        gross_operating_surplus_mixed_income=gross_operating_surplus_mixed_income,
        #
        total_hill_tail_estimate_production=[
            0.0 if not calculate_hill_exponent else calculate_tail_exponent(data["Production"].values.copy())
        ],
        total_hill_tail_estimate_number_of_employees=[
            0.0 if not calculate_hill_exponent else data["Number of Employees"].values.astype(int).copy()
        ],
        total_hill_tail_estimate_output_by_employee=[
            (
                0.0
                if not calculate_hill_exponent
                else calculate_tail_exponent(data["Production"].values / data["Number of Employees"].values.copy())
            )
        ],
    )


def get_n_firms_by_industry(data: pd.DataFrame, n_industries: int) -> np.ndarray:
    occ = Counter(data["Industry"])
    return np.array([occ[ind] for ind in range(n_industries)])
