import numpy as np

from abc import abstractmethod, ABC


class TargetProductionSetter(ABC):
    def __init__(
        self,
        existing_inventory_fraction: float,
        target_inventory_to_production_fraction: float,
        financial_constrains_fraction: float,
        maximum_debt_to_equity_ratio: float,
        intermediate_inputs_target_considers_labour_inputs: float,
        intermediate_inputs_target_considers_intermediate_inputs: float,
        intermediate_inputs_target_considers_capital_inputs: float,
        capital_inputs_target_considers_labour_inputs: float,
        capital_inputs_target_considers_intermediate_inputs: float,
        capital_inputs_target_considers_capital_inputs: float,
    ):
        self.existing_inventory_fraction = existing_inventory_fraction
        self.target_inventory_to_production_fraction = (
            target_inventory_to_production_fraction
        )
        self.financial_constrains_fraction = financial_constrains_fraction
        self.maximum_debt_to_equity_ratio = maximum_debt_to_equity_ratio

        self.intermediate_inputs_target_considers_labour_inputs = max(
            0.0, min(1.0, intermediate_inputs_target_considers_labour_inputs)
        )
        self.intermediate_inputs_target_considers_labour_inputs = (
            intermediate_inputs_target_considers_labour_inputs
        )
        self.intermediate_inputs_target_considers_intermediate_inputs = max(
            0.0,
            min(1.0, intermediate_inputs_target_considers_intermediate_inputs),
        )
        self.intermediate_inputs_target_considers_intermediate_inputs = (
            intermediate_inputs_target_considers_intermediate_inputs
        )
        self.intermediate_inputs_target_considers_capital_inputs = max(
            0.0, min(1.0, intermediate_inputs_target_considers_capital_inputs)
        )
        self.intermediate_inputs_target_considers_capital_inputs = (
            intermediate_inputs_target_considers_capital_inputs
        )

        self.capital_inputs_target_considers_labour_inputs = max(
            0.0, min(1.0, capital_inputs_target_considers_labour_inputs)
        )
        self.capital_inputs_target_considers_labour_inputs = (
            capital_inputs_target_considers_labour_inputs
        )
        self.capital_inputs_target_considers_intermediate_inputs = max(
            0.0, min(1.0, capital_inputs_target_considers_intermediate_inputs)
        )
        self.capital_inputs_target_considers_intermediate_inputs = (
            capital_inputs_target_considers_intermediate_inputs
        )
        self.capital_inputs_target_considers_capital_inputs = max(
            0.0, min(1.0, capital_inputs_target_considers_capital_inputs)
        )
        self.capital_inputs_target_considers_capital_inputs = (
            capital_inputs_target_considers_capital_inputs
        )

    @abstractmethod
    def compute_target_production(
        self,
        current_estimated_demand: np.ndarray,
        initial_inventory: np.ndarray,
        previous_inventory: np.ndarray,
        previous_production: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_loans_applied_for: np.ndarray,
        current_firm_deposits: np.ndarray,
        interest_on_overdraft_rates: np.ndarray,
        interest_paid_on_loans: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_constrained_intermediate_inputs_target_production(
        self,
        previous_production: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_loans_applied_for: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_constrained_capital_inputs_target_production(
        self,
        previous_production: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_loans_applied_for: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultTargetProductionSetter(TargetProductionSetter):
    def compute_target_production(
        self,
        current_estimated_demand: np.ndarray,
        initial_inventory: np.ndarray,
        previous_inventory: np.ndarray,
        previous_production: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_loans_applied_for: np.ndarray,
        current_firm_deposits: np.ndarray,
        interest_on_overdraft_rates: np.ndarray,
        interest_paid_on_loans: np.ndarray,
    ) -> np.ndarray:
        if self.financial_constrains_fraction == 0.0:
            return np.maximum(
                1e-12,
                current_estimated_demand
                - self.existing_inventory_fraction * previous_inventory
                + self.target_inventory_to_production_fraction
                * previous_production,
            )
        else:
            return np.maximum(
                1e-12,
                current_estimated_demand
                - self.existing_inventory_fraction * previous_inventory
                + self.target_inventory_to_production_fraction
                * previous_production
                - self.financial_constrains_fraction
                * previous_production
                * np.divide(
                    previous_loans_applied_for,
                    self.maximum_debt_to_equity_ratio * current_firm_equity
                    - current_firm_debt
                    + np.minimum(0.0, current_firm_deposits)
                    - interest_on_overdraft_rates
                    - interest_paid_on_loans,
                    out=np.zeros_like(previous_loans_applied_for),
                    where=self.maximum_debt_to_equity_ratio
                    * current_firm_equity
                    - current_firm_debt
                    + np.minimum(0.0, current_firm_deposits)
                    - interest_on_overdraft_rates
                    - interest_paid_on_loans
                    != 0.0,
                ),
            )

    def compute_constrained_intermediate_inputs_target_production(
        self,
        previous_production: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_loans_applied_for: np.ndarray,
    ) -> np.ndarray:
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.intermediate_inputs_target_considers_labour_inputs
            * (current_limiting_labour_inputs - current_target_production),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.intermediate_inputs_target_considers_intermediate_inputs
            * (
                current_limiting_intermediate_inputs - current_target_production
            ),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.intermediate_inputs_target_considers_capital_inputs
            * (current_limiting_capital_inputs - current_target_production),
        )

        return current_target_production

    def compute_constrained_capital_inputs_target_production(
        self,
        previous_production: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_loans_applied_for: np.ndarray,
    ) -> np.ndarray:
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.capital_inputs_target_considers_labour_inputs
            * (current_limiting_labour_inputs - current_target_production),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.capital_inputs_target_considers_intermediate_inputs
            * (
                current_limiting_intermediate_inputs - current_target_production
            ),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.capital_inputs_target_considers_capital_inputs
            * (current_limiting_capital_inputs - current_target_production),
        )

        return current_target_production
