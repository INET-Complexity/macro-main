import numpy as np
import pandas as pd

from pathlib import Path
from mergedeep import merge

from inet_macromodel.agents.agent import Agent
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.goods_market.value_type import ValueType
from inet_macromodel.util.function_mapping import get_functions
from inet_macromodel.firms.firm_ts import create_firms_timeseries
from inet_macromodel.credit_market.credit_market import CreditMarket

from typing import Any, Callable


class Firms(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        n_transactors: int,
        functions: dict[str, Any],
        parameters: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, np.ndarray],
    ):
        super().__init__(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            n_transactors,
            n_transactors,
            functions,
            parameters,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.REAL,
                "Seller Value Type": ValueType.REAL,
                "Buyer Priority": 1,
                "Seller Priority": 1,
            },
        )

    @classmethod
    def from_data(
        cls,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        data: pd.DataFrame,
        corr_employees: pd.DataFrame,
        intermediate_inputs_stock: pd.DataFrame,
        used_intermediate_inputs: pd.DataFrame,
        capital_inputs_stock: pd.DataFrame,
        used_capital_inputs: pd.DataFrame,
        intermediate_inputs_productivity_matrix: pd.DataFrame,
        capital_inputs_productivity_matrix: pd.DataFrame,
        capital_inputs_depreciation_matrix: pd.DataFrame,
        industry_vectors: pd.DataFrame,
        goods_criticality_matrix: pd.DataFrame,
        calculate_hill_exponent: bool,
        config: dict[str, Any],
        init_config: dict[str, Any],
    ) -> "Firms":
        parameters, functions = {}, {}
        merge(parameters, config["parameters"], init_config["parameters"])
        merge(functions, config["functions"], init_config["functions"])

        # Get corresponding functions and parameters
        functions = get_functions(
            functions,
            loc="inet_macromodel.firms",
            func_dir=Path(__file__).parent / "func",
        )
        parameters["Intermediate Inputs Productivity Matrix"] = intermediate_inputs_productivity_matrix.values
        parameters["Capital Inputs Productivity Matrix"] = capital_inputs_productivity_matrix.values
        parameters["Capital Inputs Depreciation Matrix"] = capital_inputs_depreciation_matrix.values
        parameters["Industry Vectors"] = industry_vectors
        parameters["Goods Criticality Matrix"] = goods_criticality_matrix

        # Create the corresponding time series object
        ts = create_firms_timeseries(
            data=data,
            intermediate_inputs_stock=intermediate_inputs_stock.values,
            used_intermediate_inputs=used_intermediate_inputs.values,
            capital_inputs_stock=capital_inputs_stock.values,
            used_capital_inputs=used_capital_inputs.values,
            initial_good_prices=industry_vectors["Average Initial Price"].values,
            n_industries=n_industries,
            calculate_hill_exponent=calculate_hill_exponent,
        )

        # Additional states
        states: dict[str, Any] = {}
        for state_name in [
            "Industry",
            "Corresponding Bank ID",
        ]:
            if state_name not in data.columns:
                raise ValueError("Missing " + state_name + " from the data for initialising firms.")
            states[state_name] = data[state_name].values.astype(int)
        states["Employments"] = [corr_employees.values[i][0] for i in range(len(corr_employees.values))]
        states["is_insolvent"] = np.full(len(data), False)
        states["Excess Demand"] = np.zeros(len(data))

        return cls(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            ts.current("n_firms"),
            functions,
            parameters,
            ts,
            states,
        )

    def update_number_of_firms(self) -> None:
        self.ts.n_firms.append(self.ts.current("n_firms"))
        self.ts.n_firms_by_industry.append(self.ts.current("n_firms_by_industry"))

    def set_estimates(
        self,
        current_estimated_sectoral_growth: np.ndarray,
        previous_average_good_prices: np.ndarray,
    ) -> None:
        self.ts.estimated_growth_by_firm.append(
            self.compute_estimated_growth_by_firm(previous_average_good_prices=previous_average_good_prices)
        )
        self.ts.estimated_demand.append(
            self.compute_estimated_demand(current_estimated_sectoral_growth=current_estimated_sectoral_growth)
        )

    def compute_estimated_growth_by_firm(self, previous_average_good_prices: np.ndarray) -> np.ndarray:
        if len(self.ts.historic("inventory")) == 1:
            prev_supply = self.ts.current("production") + self.ts.current("inventory")
        else:
            prev_supply = self.ts.current("production") + self.ts.prev("inventory")
        return self.functions["growth_estimator"].compute_growth(
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=prev_supply,
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
        )

    def compute_estimated_demand(self, current_estimated_sectoral_growth: np.ndarray) -> np.ndarray:
        return self.functions["demand_estimator"].compute_estimated_demand(
            previous_demand=self.ts.current("demand"),
            estimated_sectoral_growth=current_estimated_sectoral_growth,
            estimated_growth_by_firm=self.ts.current("estimated_growth_by_firm"),
            firm_industry=self.states["Industry"],
        )

    def set_targets(self, sectoral_sentiment: np.ndarray) -> None:
        self.ts.unconstrained_target_production.append(self.compute_unconstrained_target_production())
        self.ts.constrained_target_production.append(self.compute_constrained_target_production())
        self.ts.target_production.append(self.compute_target_production(sectoral_sentiment=sectoral_sentiment))
        self.ts.desired_labour_inputs.append(self.compute_desired_labour_inputs())

    def compute_unconstrained_target_production(self) -> np.ndarray:
        return self.functions["target_production"].compute_unconstrained_target_production(
            current_estimated_demand=self.ts.current("estimated_demand"),
            initial_inventory=self.ts.initial("inventory"),
            previous_inventory=self.ts.current("inventory"),
            initial_production=self.ts.initial("production"),
            previous_production=self.ts.current("production"),
        )

    def compute_constrained_target_production(self) -> np.ndarray:
        current_limiting_stock = self.functions["production"].compute_limiting_stock(
            intermediate_inputs_productivity_matrix=self.parameters["Intermediate Inputs Productivity Matrix"][
                :, self.states["Industry"]
            ].T,
            intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            capital_inputs_productivity_matrix=self.parameters["Capital Inputs Productivity Matrix"][
                :, self.states["Industry"]
            ].T,
            capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            intermediate_inputs_utilisation_rate=self.parameters["intermediate_inputs_utilisation_rate"]["value"],
            capital_inputs_utilisation_rate=self.parameters["capital_inputs_utilisation_rate"]["value"],
            goods_criticality_matrix=self.parameters["Goods Criticality Matrix"].values[:, self.states["Industry"]].T,
        )

        return self.functions["target_production"].compute_constrained_target_production(
            current_unconstrained_target_production=self.ts.current("unconstrained_target_production"),
            current_limiting_stock=current_limiting_stock,
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_firm_production=self.ts.current("production"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

    def compute_target_production(self, sectoral_sentiment: np.ndarray) -> np.ndarray:
        return self.ts.current("constrained_target_production") + sectoral_sentiment[self.states["Industry"]] * (
            self.ts.current("unconstrained_target_production") - self.ts.current("constrained_target_production")
        )

    def compute_desired_labour_inputs(self) -> np.ndarray:
        return self.functions["desired_labour"].compute_desired_labour(
            current_desired_production=self.ts.current("target_production"),
        )

    @staticmethod
    def compute_labour_inputs(corresponding_firm: np.ndarray, current_labour_inputs: np.ndarray) -> np.ndarray:
        current_labour_inputs_without_unemployed = current_labour_inputs.copy()
        current_labour_inputs_without_unemployed[corresponding_firm == -1] = 0.0
        corresponding_firm_with_0 = corresponding_firm.copy()
        corresponding_firm_with_0[corresponding_firm_with_0 == -1] = 0
        return np.bincount(corresponding_firm_with_0, weights=current_labour_inputs_without_unemployed)

    @staticmethod
    def compute_n_employees(corresponding_firm: np.ndarray) -> np.ndarray:
        corresponding_firm_with_0 = corresponding_firm.copy()
        corresponding_firm_with_0[corresponding_firm_with_0 == -1] = 0
        return np.bincount(corresponding_firm_with_0, weights=corresponding_firm != -1)

    def compute_production(self) -> np.ndarray:
        return self.functions["production"].compute_production(
            desired_production=self.ts.current("constrained_target_production"),
            current_labour_inputs=self.ts.current("labour_inputs"),
            intermediate_inputs_productivity_matrix=self.parameters["Intermediate Inputs Productivity Matrix"][
                :, self.states["Industry"]
            ].T,
            intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            capital_inputs_productivity_matrix=self.parameters["Capital Inputs Productivity Matrix"][
                :, self.states["Industry"]
            ].T,
            capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            intermediate_inputs_utilisation_rate=self.parameters["intermediate_inputs_utilisation_rate"]["value"],
            capital_inputs_utilisation_rate=self.parameters["capital_inputs_utilisation_rate"]["value"],
            goods_criticality_matrix=self.parameters["Goods Criticality Matrix"].values[:, self.states["Industry"]].T,
        )

    def compute_total_sales(self) -> np.ndarray:
        return self.ts.current("price") * self.ts.current("production") - self.ts.current("taxes_paid_on_production")

    def compute_offered_wage_function(
        self,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        unemployment_benefits_by_individual: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        return self.functions["offered_wage_setter"].get_offered_wage_given_labour_inputs_function(
            firm_employments=self.states["Employments"],
            current_individual_labour_inputs=current_individual_labour_inputs,
            previous_employee_income=previous_employee_income,
            historic_desired_labour_inputs=self.ts.historic("desired_labour_inputs"),
            historic_realised_labour_inputs=self.ts.historic("labour_inputs"),
            unemployment_benefits_by_individual=unemployment_benefits_by_individual,
        )

    def set_wages(
        self,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
    ) -> np.ndarray:
        return self.functions["wage_setter"].set_wages(
            firm_employments=self.states["Employments"],
            current_individual_labour_inputs=current_individual_labour_inputs,
            previous_employee_income=previous_employee_income,
            historic_desired_labour_inputs=self.ts.historic("desired_labour_inputs"),
            historic_realised_labour_inputs=self.ts.historic("labour_inputs"),
        )

    @staticmethod
    def compute_total_wages_paid(
        corresponding_firm: np.ndarray,
        individual_wages: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> np.ndarray:
        individual_wages_without_unemployed = individual_wages.copy()
        individual_wages_without_unemployed[corresponding_firm == -1] = 0.0
        corresponding_firm_with_0 = corresponding_firm.copy()
        corresponding_firm_with_0[corresponding_firm_with_0 == -1] = 0
        return (
            (1.0 + employer_social_insurance_tax)
            / (1 - employee_social_insurance_tax - income_taxes * (1 - employee_social_insurance_tax))
            * np.bincount(corresponding_firm_with_0, weights=individual_wages_without_unemployed)
        )

    def compute_price(
        self,
        current_estimated_ppi_inflation: np.ndarray,
        previous_average_good_prices: np.ndarray,
    ) -> np.ndarray:
        if len(self.ts.historic("price")) == 1:
            return self.ts.current("price")
        return self.functions["prices"].compute_price(
            prev_prices=self.ts.current("price"),
            current_estimated_ppi_inflation=current_estimated_ppi_inflation,
            excess_demand=self.states["Excess Demand"],
            inventories=self.ts.current("inventory"),
            production=self.ts.current("production"),
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=self.ts.prev("production") + self.ts.prev("inventory"),
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
            curr_unit_costs=self.ts.current("unit_costs"),
            prev_unit_costs=self.ts.prev("unit_costs"),
        )

    def compute_unconstrained_demand_for_intermediate_inputs(self) -> np.ndarray:
        return self.functions["target_intermediate_inputs"].compute_unconstrained_target_intermediate_inputs(
            current_target_production=self.ts.current("target_production"),
            intermediate_inputs_productivity_matrix=self.parameters["Intermediate Inputs Productivity Matrix"][
                :, self.states["Industry"]
            ].T,
            prev_intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            initial_intermediate_inputs_stock=self.ts.initial("intermediate_inputs_stock"),
            prev_production=self.ts.current("production"),
            initial_production=self.ts.initial("production"),
        )

    def compute_unconstrained_demand_for_intermediate_inputs_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        return np.matmul(
            self.ts.current("unconstrained_target_intermediate_inputs"),
            current_good_prices,
        )

    def compute_unconstrained_demand_for_capital_inputs(self) -> np.ndarray:
        return self.functions["target_capital_inputs"].compute_unconstrained_target_capital_inputs(
            current_target_production=self.ts.current("target_production"),
            capital_inputs_depreciation_matrix=self.parameters["Capital Inputs Depreciation Matrix"][
                :, self.states["Industry"]
            ].T,
            prev_capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            initial_capital_inputs_stock=self.ts.initial("capital_inputs_stock"),
            prev_production=self.ts.current("production"),
            initial_production=self.ts.initial("production"),
        )

    def compute_unconstrained_demand_for_capital_inputs_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        return np.matmul(
            self.ts.current("unconstrained_target_capital_inputs"),
            current_good_prices,
        )

    def compute_target_credit(self) -> None:
        estimated_change_in_deposits = (
            self.ts.current("price") * self.ts.current("constrained_target_production")
            - self.ts.current("total_wage")
            - self.ts.current("labour_costs")
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("corporate_taxes_paid")
            - self.ts.current("interest_paid")
            - self.ts.current("debt_installments")
        )
        estimated_deposits = self.ts.current("deposits") + estimated_change_in_deposits
        target_short_term_credit, target_long_term_credit = self.functions["target_credit"].compute_target_credit(
            estimated_deposits=estimated_deposits,
            unconstrained_target_intermediate_inputs_costs=self.ts.current(
                "unconstrained_target_intermediate_inputs_costs"
            ),
            unconstrained_target_capital_inputs_costs=self.ts.current("unconstrained_target_capital_inputs_costs"),
        )
        self.ts.target_short_term_credit.append(target_short_term_credit)
        self.ts.total_target_short_term_credit.append([target_short_term_credit.sum()])
        self.ts.target_long_term_credit.append(target_long_term_credit)
        self.ts.total_target_long_term_credit.append([target_long_term_credit.sum()])

    def compute_debt(self) -> np.ndarray:
        return self.ts.current("short_term_loan_debt") + self.ts.current("long_term_loan_debt")

    def compute_interest_paid_on_deposits(
        self,
        bank_interest_rate_on_firm_deposits: np.ndarray,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        return -(
            bank_interest_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.maximum(0.0, self.ts.current("deposits"))
            + bank_overdraft_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.minimum(0.0, self.ts.current("deposits"))
        )

    def compute_interest_paid(self) -> np.ndarray:
        return self.ts.current("interest_paid_on_loans") + self.ts.current("interest_paid_on_deposits")

    def compute_offered_price(self) -> np.ndarray:
        return np.bincount(
            self.states["Industry"],
            weights=self.ts.current("price_in_usd") * (self.ts.current("production") + self.ts.current("inventory")),
        ) / np.bincount(self.states["Industry"], weights=self.ts.current("production") + self.ts.current("inventory"))

    def prepare_buying_goods(self) -> None:
        # Target intermediate inputs
        self.ts.target_intermediate_inputs.append(
            self.functions["target_intermediate_inputs"].compute_target_intermediate_inputs(
                unconstrained_target_intermediate_inputs=self.ts.current("unconstrained_target_intermediate_inputs"),
                target_short_term_credit=self.ts.current("target_short_term_credit"),
                received_short_term_credit=self.ts.current("received_short_term_credit"),
            )
        )

        # Target capital inputs
        self.ts.target_capital_inputs.append(
            self.functions["target_capital_inputs"].compute_target_capital_inputs(
                unconstrained_target_capital_inputs=self.ts.current("unconstrained_target_capital_inputs"),
                target_long_term_credit=self.ts.current("target_long_term_credit"),
                received_long_term_credit=self.ts.current("received_long_term_credit"),
            )
        )

        # Setting total real amount of goods to buy
        self.set_goods_to_buy(self.ts.current("target_intermediate_inputs") + self.ts.current("target_capital_inputs"))

    def prepare_selling_goods(self) -> None:
        self.set_goods_to_sell(self.ts.current("production") + self.ts.current("inventory"))
        self.ts.price_in_usd.append(1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("price"))
        self.ts.price_offered.append(self.compute_offered_price())
        self.set_prices(self.ts.current("price_in_usd"))
        self.set_seller_industries(self.states["Industry"])

    def prepare_goods_market_clearing(self, exchange_rate_usd_to_lcu: float) -> None:
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.prepare_buying_goods()
        self.prepare_selling_goods()

    def distribute_bought_goods(self) -> None:
        (
            new_intermediate_inputs,
            new_capital_inputs,
        ) = self.functions["bought_goods_distributor"].distribute_bought_goods(
            desired_intermediate_inputs=self.ts.current("target_intermediate_inputs"),
            desired_investment=self.ts.current("target_capital_inputs"),
            buy_real=self.ts.current("real_amount_bought"),
        )
        self.ts.real_amount_bought_as_intermediate_inputs.append(new_intermediate_inputs)
        self.ts.real_amount_bought_as_capital_goods.append(new_capital_inputs)

    def compute_gross_fixed_capital_formation(self, current_good_prices: np.ndarray) -> np.ndarray:
        return (self.ts.current("real_amount_bought_as_capital_goods") * current_good_prices).sum(axis=0)

    def update_total_newly_bought_costs(self, current_good_prices: np.ndarray) -> None:
        amount_ii = (self.ts.current("real_amount_bought_as_intermediate_inputs") * current_good_prices).sum(axis=1)
        amount_cap = (self.ts.current("real_amount_bought_as_capital_goods") * current_good_prices).sum(axis=1)

        # Just take fractions
        self.ts.total_intermediate_inputs_bought_costs.append(
            self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            * np.divide(
                amount_ii,
                amount_ii + amount_cap,
                out=np.zeros_like(amount_ii),
                where=amount_ii + amount_cap != 0,
            )
        )
        self.ts.total_capital_inputs_bought_costs.append(
            self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            - self.ts.current("total_intermediate_inputs_bought_costs")
        )

    def compute_demand(self) -> np.ndarray:
        return self.functions["demand_for_goods"].compute_demand(
            sell_real=self.ts.current("real_amount_sold"),
            excess_demand=self.ts.current("real_excess_demand"),
        )

    def compute_nominal_production(self, current_good_prices: np.ndarray) -> np.ndarray:
        return current_good_prices[self.states["Industry"]] * self.ts.current("production")

    def compute_inventory(self) -> np.ndarray:
        new_inventories = (
            self.ts.current("inventory") + self.ts.current("production") - self.ts.current("real_amount_sold")
        )
        return np.maximum(
            0.0,
            (1 - np.array(self.parameters["depreciation_rates"]["value"])[self.states["Industry"]]) * new_inventories,
        )

    def compute_nominal_inventory(self, current_good_prices: np.ndarray) -> np.ndarray:
        return current_good_prices[self.states["Industry"]] * self.ts.current("inventory")

    def compute_used_intermediate_inputs(self):
        return self.functions["production"].compute_intermediate_inputs_used(
            realised_production=self.ts.current("production"),
            intermediate_inputs_productivity_matrix=self.parameters["Intermediate Inputs Productivity Matrix"][
                :, self.states["Industry"]
            ].T,
            intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            goods_criticality_matrix=self.parameters["Goods Criticality Matrix"].values[:, self.states["Industry"]].T,
        )

    def compute_used_intermediate_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        return (self.ts.current("used_intermediate_inputs") * current_good_prices).sum(axis=1)

    def compute_intermediate_inputs_stock(self) -> np.ndarray:
        return np.maximum(
            0.0,
            self.ts.current("intermediate_inputs_stock")
            - self.ts.current("used_intermediate_inputs")
            + self.ts.current("real_amount_bought_as_intermediate_inputs"),
        )

    def compute_intermediate_inputs_stock_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        return (self.ts.current("intermediate_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_used_capital_inputs(self):
        return self.functions["production"].compute_capital_inputs_used(
            realised_production=self.ts.current("production"),
            capital_inputs_depreciation_matrix=self.parameters["Capital Inputs Depreciation Matrix"][
                :, self.states["Industry"]
            ].T,
            capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            goods_criticality_matrix=self.parameters["Goods Criticality Matrix"].values[:, self.states["Industry"]].T,
        )

    def compute_used_capital_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        return (self.ts.current("used_capital_inputs") * current_good_prices).sum(axis=1)

    def compute_capital_inputs_stock(self) -> np.ndarray:
        # There might be a delay between buying new capital and being able to use it in the next iteration
        hist_bought_capital = np.array(self.ts.historic("real_amount_bought_as_capital_goods")[1:])
        delayed_bought_capital = np.zeros((self.ts.current("n_firms"), self.n_industries))
        for g in range(self.n_industries):
            delay = self.parameters["capital_inputs_delay"]["value"][g]
            if delay < hist_bought_capital.shape[0]:
                delayed_bought_capital[:, g] = hist_bought_capital[-delay - 1, :, g]

        return np.maximum(
            0.0,
            self.ts.current("capital_inputs_stock") - self.ts.current("used_capital_inputs") + delayed_bought_capital,
        )

    def compute_capital_inputs_stock_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        return (self.ts.current("capital_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_total_inventory_change(self) -> np.ndarray:
        return self.ts.current("price") * (self.ts.current("inventory") - self.ts.prev("inventory"))

    def compute_taxes_paid_on_production(self, taxes_less_subsidies_rates: np.ndarray) -> np.ndarray:
        return (
            taxes_less_subsidies_rates[self.states["Industry"]]
            * self.ts.current("production")
            * self.ts.current("price")
        )

    def compute_profits(self) -> np.ndarray:
        return (
            self.ts.current("price") * self.ts.current("production")
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("used_capital_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("interest_paid")
        )

    def compute_unit_costs(self) -> np.ndarray:
        return (
            self.ts.current("total_wage")
            + self.ts.current("used_intermediate_inputs_costs")
            + self.ts.current("used_capital_inputs_costs")
            + self.ts.current("taxes_paid_on_production")
        ) / self.ts.current("production")

    def compute_corporate_taxes_paid(self, tau_firm: float) -> np.ndarray:
        return tau_firm * np.maximum(0.0, self.ts.current("profits"))

    def compute_deposits(self) -> np.ndarray:
        return (
            self.ts.current("deposits")
            + self.ts.current("nominal_amount_sold_in_lcu")
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("used_capital_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("corporate_taxes_paid")
            - self.ts.current("interest_paid")
            + self.ts.current("received_credit")
            - self.ts.current("debt_installments")
        )

    def compute_gross_operating_surplus_mixed_income(self) -> np.ndarray:
        return (
            self.ts.current("nominal_amount_sold_in_lcu")
            + self.ts.current("price") * (self.ts.current("inventory") - self.ts.prev("inventory"))
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
        )

    def handle_insolvency(self, credit_market: CreditMarket) -> None:
        for firm_id in np.where(self.states["is_insolvent"])[0]:
            credit_market.remove_loans_to_firm(firm_id)
        self.functions["demography"].handle_firm_insolvency(
            current_firm_is_insolvent=self.states["is_insolvent"],
            current_firm_debts=self.ts.current("debt"),
            current_firm_deposits=self.ts.current("deposits"),
        )

    def compute_insolvency_rate(self) -> tuple[float, np.ndarray]:
        firm_insolvency_rate = self.states["is_insolvent"].mean()
        num_insolvent_firms_by_sector = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            num_insolvent_firms_by_sector[g] = np.sum(self.states["is_insolvent"][self.states["Industry"] == g])
        self.states["is_insolvent"] = np.full(self.ts.current("n_firms"), False)
        return firm_insolvency_rate, num_insolvent_firms_by_sector

    def compute_equity(self, current_good_prices: np.ndarray) -> np.ndarray:
        material = np.dot(self.ts.current("intermediate_inputs_stock"), current_good_prices)
        capital = np.dot(self.ts.current("capital_inputs_stock"), current_good_prices)
        return (
            self.ts.current("inventory") * self.ts.current("price")
            + material
            + capital
            + self.ts.current("deposits")
            - self.ts.current("debt")
        )

    def compute_total_debt(self) -> float:
        return self.ts.current("debt").sum()

    def compute_total_deposits(self) -> float:
        return self.ts.current("deposits").sum()
