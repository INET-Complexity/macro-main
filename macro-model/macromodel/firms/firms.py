from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
from macro_data import SyntheticFirms
from typing import Any, Callable

from macromodel.configurations import FirmsConfiguration
from macromodel.agents.agent import Agent
from macromodel.credit_market.credit_market import CreditMarket
from macromodel.firms.firm_ts import FirmTimeSeries
from macromodel.goods_market.value_type import ValueType
from macromodel.util.function_mapping import functions_from_model, update_functions


class Firms(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Callable],
        ts: FirmTimeSeries,
        states: dict[str, np.ndarray],
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        depreciation_rates: np.ndarray,
        capital_inputs_delay: np.ndarray,
        average_initial_price: np.ndarray,
        configuration: FirmsConfiguration,
    ):
        n_transactors = ts.current("n_firms")
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_transactors,
            n_transactors,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.REAL,
                "Seller Value Type": ValueType.REAL,
                "Buyer Priority": 1,
                "Seller Priority": 1,
            },
        )

        self.functions: dict[str, Any] = functions
        self.intermediate_inputs_productivity_matrix = intermediate_inputs_productivity_matrix
        self.capital_inputs_productivity_matrix = capital_inputs_productivity_matrix
        self.capital_inputs_depreciation_matrix = capital_inputs_depreciation_matrix
        self.goods_criticality_matrix = goods_criticality_matrix
        self.intermediate_inputs_utilisation_rate = intermediate_inputs_utilisation_rate
        self.capital_inputs_utilisation_rate = capital_inputs_utilisation_rate
        self.capital_inputs_delay = capital_inputs_delay
        self.depreciation_rates = depreciation_rates

        self.average_initial_price = average_initial_price

        self.configuration = configuration

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_firms: SyntheticFirms,
        configuration: FirmsConfiguration,
        country_name: str,
        all_country_names: list[str],
        goods_criticality_matrix: pd.DataFrame | np.ndarray,
        average_initial_price: np.ndarray,
    ):
        functions = functions_from_model(model=configuration.functions, loc="macromodel.firms")

        intermediate_inputs_productivity_matrix = synthetic_firms.intermediate_inputs_productivity_matrix
        capital_inputs_productivity_matrix = synthetic_firms.capital_inputs_productivity_matrix
        capital_inputs_depreciation_matrix = synthetic_firms.capital_inputs_depreciation_matrix
        if isinstance(goods_criticality_matrix, pd.DataFrame):
            goods_criticality_matrix = goods_criticality_matrix.values

        corr_employees = synthetic_firms.firm_data["Employees ID"]

        corr_employees = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_employees]

        data = synthetic_firms.firm_data.drop(columns=["Employees ID"]).astype(float).rename_axis("Firm ID")

        ts = FirmTimeSeries.from_data(
            data=data,
            intermediate_inputs_stock=synthetic_firms.intermediate_inputs_stock,
            used_intermediate_inputs=synthetic_firms.used_intermediate_inputs,
            capital_inputs_stock=synthetic_firms.capital_inputs_stock,
            used_capital_inputs=synthetic_firms.used_capital_inputs,
            initial_good_prices=average_initial_price,
            n_industries=len(synthetic_firms.industries),
            calculate_hill_exponent=configuration.calculate_hill_exponent,
        )

        states: dict[str, Any] = {}

        for state_name in [
            "Industry",
            "Corresponding Bank ID",
        ]:
            if state_name not in data.columns:
                raise ValueError("Missing " + state_name + " from the data for initialising firms.")
            states[state_name] = data[state_name].fillna(-1).values.astype(int)

        states["Employments"] = corr_employees
        states["is_insolvent"] = np.full(data.shape[0], False)
        states["Excess Demand"] = np.zeros(data.shape[0])

        states["Labour Productivity by Industry"] = synthetic_firms.labour_productivity_by_industry

        return cls(
            country_name,
            all_country_names,
            len(synthetic_firms.industries),
            functions,
            ts,
            states,
            intermediate_inputs_productivity_matrix,
            capital_inputs_productivity_matrix,
            capital_inputs_depreciation_matrix,
            goods_criticality_matrix,
            configuration.parameters.intermediate_inputs_utilisation_rate,
            configuration.parameters.capital_inputs_utilisation_rate,
            np.array(configuration.parameters.depreciation_rates),
            np.array(configuration.parameters.capital_inputs_delay),
            average_initial_price,
            configuration=configuration,
        )

    def reset(self, configuration: FirmsConfiguration) -> None:
        self.gen_reset()
        update_functions(model=configuration.functions, loc="macromodel.firms", functions=self.functions)

        current_inv = (
            self.ts.current("production")
            * configuration.functions.target_production.parameters["target_inventory_to_production_fraction"]
        )

        industries = self.states["Industry"]
        initial_good_prices = self.average_initial_price

        inter_inputs_stock = (
            1.0
            / configuration.parameters.intermediate_inputs_utilisation_rate
            * np.divide(
                self.ts.current("production"),
                self.intermediate_inputs_productivity_matrix[:, industries],
                out=np.zeros(self.intermediate_inputs_productivity_matrix[:, industries].shape),
                where=self.intermediate_inputs_productivity_matrix[:, industries] != 0.0,
            ).T
        )

        cap_inputs_stock = (
            1.0
            / configuration.parameters.capital_inputs_utilisation_rate
            * np.divide(
                self.ts.current("production"),
                self.capital_inputs_productivity_matrix[:, industries],
                out=np.zeros(self.capital_inputs_productivity_matrix[:, industries].shape),
                where=self.capital_inputs_productivity_matrix[:, industries] != 0.0,
            ).T
        )

        self.ts.reset_values(
            inventory=current_inv,
            initial_good_prices=initial_good_prices,
            intermediate_inputs_stock=inter_inputs_stock,
            capital_inputs_stock=cap_inputs_stock,
        )

        self.configuration = deepcopy(configuration)

    def update_number_of_firms(self) -> None:
        self.ts.n_firms.append(self.ts.current("n_firms"))
        self.ts.n_firms_by_industry.append(self.ts.current("n_firms_by_industry"))

    def set_estimates(
        self,
        current_estimated_growth: float,
        previous_average_good_prices: np.ndarray,
    ) -> None:
        self.ts.estimated_growth_by_firm.append(
            self.compute_estimated_growth_by_firm(previous_average_good_prices=previous_average_good_prices)
        )
        self.ts.estimated_demand.append(
            self.compute_estimated_demand(current_estimated_growth=current_estimated_growth)
        )

    def compute_estimated_growth_by_firm(
        self,
        previous_average_good_prices: np.ndarray,
        min_growth: float = -0.2,
        max_growth: float = 0.2,
    ) -> np.ndarray:
        if len(self.ts.historic("inventory")) == 1:
            prev_supply = self.ts.current("production") + self.ts.current("inventory")
        else:
            prev_supply = self.ts.current("production") + self.ts.prev("inventory")
        growth = self.functions["growth_estimator"].compute_growth(
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=prev_supply,
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
        )
        return np.maximum(min_growth, np.minimum(max_growth, growth))

    def compute_estimated_demand(
        self,
        current_estimated_growth: float,
    ) -> np.ndarray:
        return self.functions["demand_estimator"].compute_estimated_demand(
            previous_demand=self.ts.current("demand"),
            current_estimated_growth=current_estimated_growth,
            estimated_growth_by_firm=self.ts.current("estimated_growth_by_firm"),
        )

    def set_targets(
        self,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> None:
        self.ts.limiting_intermediate_inputs.append(
            self.functions["production"].compute_limiting_intermediate_inputs_stock(
                intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix[
                    :, self.states["Industry"]
                ].T,
                intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
                intermediate_inputs_utilisation_rate=self.intermediate_inputs_utilisation_rate,
                goods_criticality_matrix=self.goods_criticality_matrix,
            )
        )
        self.ts.limiting_capital_inputs.append(
            self.functions["production"].compute_limiting_capital_inputs_stock(
                capital_inputs_productivity_matrix=self.capital_inputs_productivity_matrix[
                    :, self.states["Industry"]
                ].T,
                capital_inputs_stock=self.ts.current("capital_inputs_stock"),
                capital_inputs_utilisation_rate=self.capital_inputs_utilisation_rate,
                goods_criticality_matrix=self.goods_criticality_matrix,
            )
        )
        self.ts.target_production.append(
            self.compute_target_production(
                bank_overdraft_rate_on_firm_deposits=bank_overdraft_rate_on_firm_deposits,
            )
        )
        self.ts.desired_labour_inputs.append(self.compute_desired_labour_inputs())
        self.ts.target_intermediate_inputs_production.append(self.compute_target_intermediate_inputs_production())
        self.ts.target_capital_inputs_production.append(self.compute_target_capital_inputs_production())

    def compute_estimated_profits(self, estimated_growth: float, estimated_inflation: float) -> np.ndarray:
        return self.functions["profit_estimator"].compute_estimated_profits(
            current_profits=self.ts.current("profits"),
            estimated_growth=estimated_growth,
            estimated_inflation=estimated_inflation,
        )

    def compute_target_production(
        self,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        return self.functions["target_production"].compute_target_production(
            current_estimated_demand=self.ts.current("estimated_demand"),
            initial_inventory=self.ts.initial("inventory"),
            previous_inventory=self.ts.current("inventory"),
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
            current_firm_deposits=self.ts.current("deposits"),
            interest_on_overdraft_rates=-bank_overdraft_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.minimum(0.0, self.ts.current("deposits")),
            interest_paid_on_loans=self.ts.current("interest_paid_on_loans"),
        )

    def compute_target_intermediate_inputs_production(self) -> np.ndarray:
        return self.functions["target_production"].compute_constrained_intermediate_inputs_target_production(
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

    def compute_target_capital_inputs_production(self) -> np.ndarray:
        return self.functions["target_production"].compute_constrained_capital_inputs_target_production(
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

    def compute_desired_labour_inputs(self) -> np.ndarray:
        return self.functions["desired_labour"].compute_desired_labour(
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
        )

    def compute_labour_inputs(self, corresponding_firm: np.ndarray, current_labour_inputs: np.ndarray) -> np.ndarray:
        labour_inputs_from_employees = np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            weights=current_labour_inputs[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )
        industry_labour_productivity_by_firm = self.states["Labour Productivity by Industry"][self.states["Industry"]]

        # Compute labour productivity
        self.ts.labour_productivity_factor.append(
            self.functions["labour_productivity"].compute_labour_productivity_factor(
                current_target_production=self.ts.current("target_production"),
                current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
                current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
                labour_inputs_from_employees=labour_inputs_from_employees,
                industry_labour_productivity_by_firm=industry_labour_productivity_by_firm,
            )
        )
        self.ts.labour_productivity.append(
            self.ts.current("labour_productivity_factor") * industry_labour_productivity_by_firm
        )

        # Compute labour inputs
        self.ts.labour_inputs.append(self.ts.current("labour_productivity") * labour_inputs_from_employees)
        self.ts.normalised_labour_inputs.append(industry_labour_productivity_by_firm * labour_inputs_from_employees)

        return labour_inputs_from_employees

    def compute_n_employees(self, corresponding_firm: np.ndarray) -> np.ndarray:
        return np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )

    def compute_production(self) -> np.ndarray:
        return self.functions["production"].compute_production(
            desired_production=self.ts.current("target_production"),
            current_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_intermediate_inputs"),
        )

    def compute_total_sales(self) -> np.ndarray:
        return self.ts.current("price") * self.ts.current("production") - self.ts.current("taxes_paid_on_production")

    def compute_wages_markup(self) -> np.ndarray:
        return self.functions["wage_setter"].compute_wage_tightness_markup(
            historic_desired_labour_inputs=self.ts.historic("desired_labour_inputs"),
            historic_realised_labour_inputs=self.ts.historic("labour_inputs"),
        )

    def compute_offered_wage_function(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        unemployment_benefits_by_individual: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        return self.functions["wage_setter"].get_offered_wage_given_labour_inputs_function(
            corresponding_firm=corresponding_firm,
            current_individual_labour_inputs=current_individual_labour_inputs,
            previous_employee_income=previous_employee_income,
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            industry_labour_productivity_by_firm=self.states["Labour Productivity by Industry"][
                self.states["Industry"]
            ],
            initial_wage_per_capita=self.ts.initial("real_wage_per_capita"),
            current_wage_per_capita=self.ts.current("real_wage_per_capita"),
            current_labour_productivity_factor=self.ts.current("labour_productivity_factor"),
            prev_labour_productivity_factor=self.ts.prev("labour_productivity_factor"),
            current_wage_tightness_markup=self.ts.current("wage_tightness_markup"),
            income_taxes=income_taxes,
            employee_social_insurance_tax=employee_social_insurance_tax,
            employer_social_insurance_tax=employer_social_insurance_tax,
            unemployment_benefits_by_individual=unemployment_benefits_by_individual,
        )

    def set_employee_income(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        current_individual_stating_new_job: np.ndarray,
        current_employee_income: np.ndarray,
        current_individual_offered_wage: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        estimated_ppi_inflation: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> np.ndarray:
        return self.functions["wage_setter"].set_employee_income(
            corresponding_firm=corresponding_firm,
            current_individual_labour_inputs=current_individual_labour_inputs,
            current_individual_stating_new_job=current_individual_stating_new_job,
            current_employee_income=current_employee_income,
            current_individual_offered_wage=current_individual_offered_wage,
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            labour_inputs_from_employees=labour_inputs_from_employees,
            industry_labour_productivity_by_firm=self.states["Labour Productivity by Industry"][
                self.states["Industry"]
            ],
            initial_wage_per_capita=self.ts.initial("real_wage_per_capita"),
            current_wage_per_capita=self.ts.current("real_wage_per_capita"),
            current_labour_productivity_factor=self.ts.current("labour_productivity_factor"),
            prev_labour_productivity_factor=self.ts.prev("labour_productivity_factor"),
            current_wage_tightness_markup=self.ts.current("wage_tightness_markup"),
            estimated_ppi_inflation=estimated_ppi_inflation,
            income_taxes=income_taxes,
            employee_social_insurance_tax=employee_social_insurance_tax,
            employer_social_insurance_tax=employer_social_insurance_tax,
        )

    def update_total_wages_paid(
        self,
        corresponding_firm: np.ndarray,
        individual_wages: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
        cpi: float,
    ) -> None:
        real_wages = np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            weights=individual_wages[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )
        self.ts.total_wage.append(
            cpi
            * (
                (1.0 + employer_social_insurance_tax)
                / (1 - employee_social_insurance_tax - income_taxes * (1 - employee_social_insurance_tax))
                * real_wages
            )
        )
        self.ts.real_wage_per_capita.append(
            self.ts.current("total_wage") / cpi / self.ts.current("number_of_employees")
        )

    def compute_price(
        self,
        current_estimated_ppi_inflation: np.ndarray,
        previous_average_good_prices: np.ndarray,
        ppi_during: np.ndarray,
    ) -> np.ndarray:
        return self.functions["prices"].compute_price(
            prev_prices=self.ts.current("price"),
            current_estimated_ppi_inflation=current_estimated_ppi_inflation,
            excess_demand=self.states["Excess Demand"],
            inventories=self.ts.current("inventory"),
            production=self.ts.current("production"),
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=(
                self.ts.current("production") + self.ts.current("inventory")
                if len(self.ts.historic("price")) == 1
                else self.ts.prev("production") + self.ts.current("inventory")
            ),
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
            curr_unit_costs=self.ts.current("unit_costs"),
            prev_unit_costs=(
                self.ts.current("unit_costs") if len(self.ts.historic("price")) == 1 else self.ts.prev("unit_costs")
            ),
            ppi_during=ppi_during,
            current_time=len(self.ts.historic("price")),
        )

    def compute_unconstrained_demand_for_intermediate_inputs(
        self,
    ) -> np.ndarray:
        return self.functions["target_intermediate_inputs"].compute_unconstrained_target_intermediate_inputs(
            current_target_production=self.ts.current("target_intermediate_inputs_production"),
            intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix[
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
            current_target_production=self.ts.current("target_capital_inputs_production"),
            capital_inputs_depreciation_matrix=self.capital_inputs_depreciation_matrix[:, self.states["Industry"]].T,
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

    def compute_target_credit(self, estimated_growth: float, estimated_inflation: float) -> None:
        estimated_corporate_taxes = (
            (1 + estimated_growth) * (1 + estimated_inflation) * self.ts.current("corporate_taxes_paid")
        )
        estimated_change_in_deposits = (
            self.ts.current("price") * self.ts.current("production")
            - self.ts.current("total_wage")
            - self.ts.current("labour_costs")
            - self.ts.current("taxes_paid_on_production")
            - estimated_corporate_taxes
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
        nom = np.bincount(
            self.states["Industry"],
            weights=self.ts.current("price_in_usd") * (self.ts.current("production") + self.ts.current("inventory")),
            minlength=self.n_industries,
        )
        real = np.bincount(
            self.states["Industry"],
            weights=self.ts.current("production") + self.ts.current("inventory"),
            minlength=self.n_industries,
        )
        avg_price = np.divide(nom, real, out=np.zeros(nom.shape), where=real != 0.0)
        avg_price[avg_price == 0.0] = self.ts.current("price_offered")[avg_price == 0.0]
        assert np.all(avg_price > 0.0)
        return avg_price

    def compute_maximum_excess_demand(self) -> np.ndarray:
        return self.functions["excess_demand"].set_maximum_excess_demand(
            current_production=self.ts.current("production"),
            target_production=self.ts.current("target_production"),
            limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            limiting_labour_inputs=self.ts.current("labour_inputs"),
        )

    def prepare_buying_goods(
        self,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
        assume_zero_growth: bool = False,
    ) -> None:
        # Target intermediate inputs
        if assume_zero_growth:
            self.ts.target_intermediate_inputs.append(self.ts.initial("target_intermediate_inputs"))
        else:
            self.ts.target_intermediate_inputs.append(
                self.functions["target_intermediate_inputs"].compute_target_intermediate_inputs(
                    unconstrained_target_intermediate_inputs=self.ts.current(
                        "unconstrained_target_intermediate_inputs"
                    ),
                    target_short_term_credit=self.ts.current("target_short_term_credit"),
                    received_short_term_credit=self.ts.current("received_short_term_credit"),
                    previous_good_prices=previous_good_prices,
                    expected_inflation=expected_inflation,
                )
            )

        # Target capital inputs
        if assume_zero_growth:
            self.ts.target_capital_inputs.append(self.ts.initial("target_capital_inputs"))
        else:
            self.ts.target_capital_inputs.append(
                self.functions["target_capital_inputs"].compute_target_capital_inputs(
                    unconstrained_target_capital_inputs=self.ts.current("unconstrained_target_capital_inputs"),
                    target_long_term_credit=self.ts.current("target_long_term_credit"),
                    received_long_term_credit=self.ts.current("received_long_term_credit"),
                    previous_good_prices=previous_good_prices,
                    expected_inflation=expected_inflation,
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
        self.set_maximum_excess_demand(self.compute_maximum_excess_demand())

    def prepare_goods_market_clearing(
        self,
        exchange_rate_usd_to_lcu: float,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> None:
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.prepare_buying_goods(
            previous_good_prices=previous_good_prices,
            expected_inflation=expected_inflation,
        )
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
        """
        print(
            "DBG",
            list(
                (
                    (self.ts.current("target_intermediate_inputs") - new_intermediate_inputs).sum(axis=0)
                    + (self.ts.current("target_capital_inputs") - new_capital_inputs).sum(axis=0)
                ).round(2)
            ),
        )
        """

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
                out=np.zeros(amount_ii.shape),
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
        return (1 - np.array(self.depreciation_rates)[self.states["Industry"]]) * np.maximum(
            0.0,
            self.ts.current("inventory") + self.ts.current("production") - self.ts.current("real_amount_sold"),
        )

    def compute_nominal_inventory(self, current_good_prices: np.ndarray) -> np.ndarray:
        return current_good_prices[self.states["Industry"]] * self.ts.current("inventory")

    def compute_used_intermediate_inputs(self):
        return self.functions["production"].compute_intermediate_inputs_used(
            realised_production=self.ts.current("production"),
            intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix[
                :, self.states["Industry"]
            ].T,
            intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            goods_criticality_matrix=self.goods_criticality_matrix,
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
            capital_inputs_depreciation_matrix=self.capital_inputs_depreciation_matrix[:, self.states["Industry"]].T,
            capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            goods_criticality_matrix=self.goods_criticality_matrix,
        )

    def compute_used_capital_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        return (self.ts.current("used_capital_inputs") * current_good_prices).sum(axis=1)

    def compute_expected_capital_inputs_stock_value(
        self,
        current_good_prices: np.ndarray,
        estimated_inflation: float,
    ) -> np.ndarray:
        return (1 + estimated_inflation) * (self.ts.current("capital_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_capital_inputs_stock(self) -> np.ndarray:
        # There might be a delay between buying new capital and being able to use it in the next iteration
        hist_bought_capital = np.array(self.ts.historic("real_amount_bought_as_capital_goods")[1:])
        delayed_bought_capital = np.zeros((self.ts.current("n_firms"), self.n_industries))
        for g in range(self.n_industries):
            delay = self.capital_inputs_delay[g]
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
        return np.divide(
            self.ts.current("total_wage")
            + self.ts.current("used_intermediate_inputs_costs")
            + self.ts.current("used_capital_inputs_costs")
            + self.ts.current("taxes_paid_on_production"),
            self.ts.current("production"),
            out=np.zeros_like(self.ts.current("production")),
            where=self.ts.current("production") != 0.0,
        )

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

    def handle_insolvency(self, credit_market: CreditMarket) -> float:
        self.states["is_insolvent"] = self.functions["demography"].handle_firm_insolvency(
            current_firm_is_insolvent=self.states["is_insolvent"],
            current_firm_equity=self.ts.current("equity"),
            current_firm_deposits=self.ts.current("deposits"),
        )

        # Remove loans
        insolvent_firms = np.where(self.states["is_insolvent"])[0]
        bad_firm_loans = credit_market.remove_loans_to_firm(insolvent_firms)

        # Update deposits
        new_firm_deposits = self.ts.current("deposits")
        new_firm_deposits[self.states["is_insolvent"]] = 0.0
        self.ts.deposits.pop()
        self.ts.deposits.append(new_firm_deposits)

        # Update equity
        new_firm_equity = self.ts.current("equity")
        new_firm_equity[self.states["is_insolvent"]] = 0.0
        self.ts.equity.pop()
        self.ts.equity.append(new_firm_equity)

        # Calculate the NPL ratio for firm loans
        total_loans_granted = (
            credit_market.ts.current("total_outstanding_loans_granted_firms_short_term")[0]
            + credit_market.ts.current("total_outstanding_loans_granted_firms_long_term")[0]
        )
        if total_loans_granted == 0.0:
            return 0.0
        else:
            return bad_firm_loans / total_loans_granted

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

    def compute_insolvency_rate(self) -> tuple[float, np.ndarray]:
        firm_insolvency_rate = self.states["is_insolvent"].mean()
        num_insolvent_firms_by_sector = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            num_insolvent_firms_by_sector[g] = np.sum(self.states["is_insolvent"][self.states["Industry"] == g])
        self.states["is_insolvent"] = np.full(self.ts.current("n_firms"), False)
        return firm_insolvency_rate, num_insolvent_firms_by_sector

    def compute_total_debt(self) -> float:
        return self.ts.current("debt").sum()

    def compute_total_deposits(self) -> float:
        return self.ts.current("deposits").sum()

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("firms", group)

    def save_industry_firms_df(self, group: h5py.Group):
        industry_firms_df = pd.DataFrame(
            data=np.array(self.states["Industry"]),
            index=pd.Index(range(self.ts.current("n_firms")), name="Firm ID"),
            columns=pd.Index(["Industry"], name="Field"),
        )

        group.create_dataset("industry_firms", data=industry_firms_df.values, dtype="int32")
        group["industry_firms"].attrs["columns"] = industry_firms_df.columns.to_list()

    def total_sales(self):
        return self.ts.get_aggregate("total_sales")

    def total_used_input_costs(self):
        return self.ts.get_aggregate("used_intermediate_inputs_costs")

    def total_bought_input_costs(self):
        return self.ts.get_aggregate("total_intermediate_inputs_bought_costs")

    def total_operating_surplus(self):
        return self.ts.get_aggregate("gross_operating_surplus_mixed_income")

    def total_wages(self):
        return self.ts.get_aggregate("total_wage")

    def total_inventory_change(self):
        return self.ts.get_aggregate("total_inventory_change")

    def total_capital_bought(self):
        return self.ts.get_aggregate("total_capital_inputs_bought_costs")

    def total_production(self):
        return self.ts.get_aggregate("production")

    def total_profits(self):
        return self.ts.get_aggregate("profits")

    def total_taxes_paid_on_production(self):
        return self.ts.get_aggregate("taxes_paid_on_production")
