from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from macromodel.agents.firms.utils import calculate_tail_exponent
from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


class FirmTimeSeries(TimeSeries):
    """Time series data container for tracking firm-level economic variables.

    This class extends TimeSeries to store and manage temporal data for firms in the economy.
    It tracks a wide range of economic variables including:

    Production & Sales:
    - production: Quantity produced by each firm
    - production_nominal: Value of production in nominal terms
    - production_histogram: Distribution of production quantities
    - target_production: Desired production levels
    - total_sales: Revenue from sales net of production taxes
    - demand: Realized demand for each firm's output
    - estimated_demand: Expected future demand
    - price: Output prices set by firms
    - price_offered: Average price offered by industry
    - price_in_usd: Prices converted to USD
    - unit_costs: Cost per unit of output

    Labor & Employment:
    - number_of_employees: Workers per firm
    - number_of_employees_histogram: Distribution of firm sizes
    - labour_inputs: Effective labor input (productivity-adjusted)
    - labour_productivity: Output per unit of labor
    - labour_productivity_factor: Productivity adjustment factor
    - normalised_labour_inputs: Labor input normalized by industry
    - desired_labour_inputs: Target employment level
    - labour_costs: Total labor cost
    - total_wage: Total wages paid
    - real_wage_per_capita: Real wages per worker
    - wage_tightness_markup: Wage adjustment for labor market conditions

    Inventory & Inputs:
    - inventory: Stock of finished goods
    - inventory_nominal: Value of inventory
    - total_inventory_change: Change in inventory value
    - intermediate_inputs_stock: Stock of intermediate inputs
    - intermediate_inputs_stock_value: Value of input stocks
    - intermediate_inputs_stock_industry: Input stocks by industry
    - used_intermediate_inputs: Inputs consumed in production
    - used_intermediate_inputs_costs: Cost of inputs used
    - capital_inputs_stock: Stock of capital goods
    - capital_inputs_stock_value: Value of capital stock
    - capital_inputs_stock_industry: Capital stock by industry
    - used_capital_inputs: Capital consumed in production
    - used_capital_inputs_costs: Cost of capital used

    Financial:
    - profits: Operating profits
    - expected_profits: Projected future profits
    - equity: Net worth (assets - liabilities)
    - deposits: Bank deposit balances
    - debt: Total outstanding debt
    - short_term_loan_debt: Short-term borrowing
    - long_term_loan_debt: Long-term borrowing
    - debt_installments: Loan repayments
    - interest_paid: Total interest expense
    - interest_paid_on_deposits: Interest on deposits
    - interest_paid_on_loans: Interest on borrowing
    - corporate_taxes_paid: Corporate income taxes paid
    - taxes_paid_on_production: Production taxes paid

    Credit Market:
    - target_short_term_credit: Desired short-term borrowing
    - target_long_term_credit: Desired long-term borrowing
    - received_short_term_credit: Actual short-term credit received
    - received_long_term_credit: Actual long-term credit received
    - received_credit: Total new credit received

    Planning & Targets:
    - limiting_intermediate_inputs: Input constraints on production
    - limiting_capital_inputs: Capital constraints on production
    - target_intermediate_inputs_production: Target input production
    - target_capital_inputs_production: Target capital production
    - estimated_growth_by_firm: Expected growth rates
    - unconstrained_target_intermediate_inputs: Desired input purchases
    - unconstrained_target_capital_inputs: Desired capital purchases
    - planned_productivity_investment: Investment amount planned for TFP improvement
    - executed_productivity_investment: Actual investment made in TFP improvement

    Emissions:
    - inputs_emissions: Emissions from input use
    - capital_emissions: Emissions from capital use
    - coal_inputs_emissions: Emissions from coal inputs
    - gas_inputs_emissions: Emissions from gas inputs
    - oil_inputs_emissions: Emissions from oil inputs
    - refined_products_inputs_emissions: Emissions from refined products
    - coal_capital_emissions: Emissions from coal capital
    - gas_capital_emissions: Emissions from gas capital
    - oil_capital_emissions: Emissions from oil capital
    - refined_products_capital_emissions: Emissions from refined products capital

    Statistical:
    - n_firms: Total number of firms
    - n_firms_by_industry: Firm count by industry
    - output_by_employee_histogram: Distribution of labor productivity
    - total_hill_tail_estimate_production: Power law tail estimate for production
    - total_hill_tail_estimate_number_of_employees: Power law tail for firm size
    - total_hill_tail_estimate_output_by_employee: Power law tail for productivity
    """

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
        inputs_emissions: Optional[np.ndarray] = None,
        capital_emissions: Optional[np.ndarray] = None,
        coal_inputs_emissions: Optional[np.ndarray] = None,
        gas_inputs_emissions: Optional[np.ndarray] = None,
        oil_inputs_emissions: Optional[np.ndarray] = None,
        refined_products_inputs_emissions: Optional[np.ndarray] = None,
        coal_capital_emissions: Optional[np.ndarray] = None,
        gas_capital_emissions: Optional[np.ndarray] = None,
        oil_capital_emissions: Optional[np.ndarray] = None,
        refined_products_capital_emissions: Optional[np.ndarray] = None,
    ) -> "FirmTimeSeries":
        """Create a FirmTimeSeries instance from initial data.

        Factory method that initializes a FirmTimeSeries with starting values for all tracked variables.
        Takes firm-level data and input/capital stocks as primary inputs.

        Args:
            data (pd.DataFrame): Initial firm data containing:
                - Price: Output prices
                - Production: Production quantities
                - Inventory: Stock of finished goods
                - Total Wages Paid: Wage payments
                - Taxes paid on Production: Production taxes
                - Number of Employees: Workforce size
                - Demand: Initial demand
                - Unit Costs: Production costs per unit
                - Debt: Outstanding loans
                - Deposits: Bank balances
                - Debt Installments: Loan payments
                - Interest paid on deposits: Deposit interest
                - Interest paid on loans: Loan interest
                - Interest paid: Total interest
                - Labour Inputs: Labor usage
                - Labour Productivity: Output per worker
            intermediate_inputs_stock (np.ndarray): Initial intermediate input inventories
            used_intermediate_inputs (np.ndarray): Initial intermediate input usage
            capital_inputs_stock (np.ndarray): Initial capital stock
            used_capital_inputs (np.ndarray): Initial capital input usage
            initial_good_prices (np.ndarray): Starting prices for goods
            n_industries (int): Number of industry sectors
            calculate_hill_exponent (bool, optional): Whether to calculate power law tails. Defaults to False.
            inputs_emissions (np.ndarray, optional): Initial input-related emissions. Defaults to None.
            capital_emissions (np.ndarray, optional): Initial capital-related emissions. Defaults to None.
            coal_inputs_emissions (np.ndarray, optional): Coal input emissions. Defaults to None.
            gas_inputs_emissions (np.ndarray, optional): Gas input emissions. Defaults to None.
            oil_inputs_emissions (np.ndarray, optional): Oil input emissions. Defaults to None.
            refined_products_inputs_emissions (np.ndarray, optional): Refined products emissions. Defaults to None.
            coal_capital_emissions (np.ndarray, optional): Coal capital emissions. Defaults to None.
            gas_capital_emissions (np.ndarray, optional): Gas capital emissions. Defaults to None.
            oil_capital_emissions (np.ndarray, optional): Oil capital emissions. Defaults to None.
            refined_products_capital_emissions (np.ndarray, optional): Refined products capital emissions. Defaults to None.

        Returns:
            FirmTimeSeries: Initialized time series container with all variables set to starting values
        """
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
            planned_productivity_investment=np.zeros(data.shape[0]),
            executed_productivity_investment=np.zeros(data.shape[0]),
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
            inputs_emissions=inputs_emissions,
            capital_emissions=capital_emissions,
            coal_inputs_emissions=coal_inputs_emissions,
            gas_inputs_emissions=gas_inputs_emissions,
            oil_inputs_emissions=oil_inputs_emissions,
            refined_products_inputs_emissions=refined_products_inputs_emissions,
            coal_capital_emissions=coal_capital_emissions,
            gas_capital_emissions=gas_capital_emissions,
            oil_capital_emissions=oil_capital_emissions,
            refined_products_capital_emissions=refined_products_capital_emissions,
        )

    def reset_values(
        self,
        inventory: np.ndarray,
        initial_good_prices: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_stock: np.ndarray,
    ):
        """Reset key time series variables to new values.

        Updates inventory, input stocks, and derived financial values to new states.
        Used when reconfiguring the model or starting a new simulation period.

        Args:
            inventory (np.ndarray): New inventory levels for each firm
            initial_good_prices (np.ndarray): New prices for goods
            intermediate_inputs_stock (np.ndarray): New intermediate input stocks
            capital_inputs_stock (np.ndarray): New capital input stocks

        The method updates:
        1. Inventory levels and values
        2. Intermediate input stocks and values
        3. Capital input stocks and values
        4. Equity values based on new asset positions
        5. Gross operating surplus based on new inventory and input values
        """
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
        planned_productivity_investment=np.zeros(data.shape[0]),
        executed_productivity_investment=np.zeros(data.shape[0]),
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
    """Count the number of firms in each industry.

    Args:
        data (pd.DataFrame): Firm data containing an 'Industry' column with industry indices
        n_industries (int): Total number of industries in the economy

    Returns:
        np.ndarray: Array of length n_industries containing the count of firms in each industry
    """
    occ = Counter(data["Industry"])
    return np.array([occ[ind] for ind in range(n_industries)])
