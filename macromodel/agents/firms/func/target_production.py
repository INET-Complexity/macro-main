from abc import ABC, abstractmethod

import numpy as np


def clip(x: float) -> float:
    """Clip a value to the range [0, 1].

    Args:
        x (float): Value to clip

    Returns:
        float: Clipped value between 0 and 1
    """
    return max(0.0, min(1.0, x))


class TargetProductionSetter(ABC):
    """Abstract base class for determining firms' target production levels.

    This class defines strategies for setting production targets based on various factors:
    - Expected demand
    - Current inventory levels
    - Input constraints (labor, intermediate, capital)
    - Financial constraints (equity, debt, credit)

    The class uses multiple parameters to weight different factors:
    - Inventory management (existing and target ratios)
    - Financial constraints (debt/equity limits)
    - Input considerations (how much each input type affects targets)

    Attributes:
        existing_inventory_fraction (float): Weight given to current inventory in target calculation
        target_inventory_to_production_fraction (float): Desired inventory-to-production ratio
        financial_constrains_fraction (float): Weight of financial constraints on targets
        maximum_debt_to_equity_ratio (float): Maximum allowed debt/equity ratio
        intermediate_inputs_target_considers_labour_inputs (float): Weight of labor constraints on intermediate inputs
        intermediate_inputs_target_considers_intermediate_inputs (float): Weight of input constraints on intermediate inputs
        intermediate_inputs_target_considers_capital_inputs (float): Weight of capital constraints on intermediate inputs
        capital_inputs_target_considers_labour_inputs (float): Weight of labor constraints on capital inputs
        capital_inputs_target_considers_intermediate_inputs (float): Weight of input constraints on capital inputs
        capital_inputs_target_considers_capital_inputs (float): Weight of capital constraints on capital inputs
    """

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
        """Initialize the target production setter with configuration parameters.

        Args:
            existing_inventory_fraction (float): Weight of current inventory
            target_inventory_to_production_fraction (float): Desired inventory ratio
            financial_constrains_fraction (float): Weight of financial constraints
            maximum_debt_to_equity_ratio (float): Maximum debt/equity ratio
            intermediate_inputs_target_considers_labour_inputs (float): Labor weight for intermediates
            intermediate_inputs_target_considers_intermediate_inputs (float): Input weight for intermediates
            intermediate_inputs_target_considers_capital_inputs (float): Capital weight for intermediates
            capital_inputs_target_considers_labour_inputs (float): Labor weight for capital
            capital_inputs_target_considers_intermediate_inputs (float): Input weight for capital
            capital_inputs_target_considers_capital_inputs (float): Capital weight for capital
        """
        self.existing_inventory_fraction = existing_inventory_fraction
        self.target_inventory_to_production_fraction = target_inventory_to_production_fraction
        self.financial_constrains_fraction = financial_constrains_fraction
        self.maximum_debt_to_equity_ratio = maximum_debt_to_equity_ratio

        self.intermediate_inputs_target_considers_labour_inputs = clip(
            intermediate_inputs_target_considers_labour_inputs
        )
        self.intermediate_inputs_target_considers_labour_inputs = intermediate_inputs_target_considers_labour_inputs
        self.intermediate_inputs_target_considers_intermediate_inputs = clip(
            intermediate_inputs_target_considers_intermediate_inputs
        )
        self.intermediate_inputs_target_considers_intermediate_inputs = (
            intermediate_inputs_target_considers_intermediate_inputs
        )
        self.intermediate_inputs_target_considers_capital_inputs = clip(
            intermediate_inputs_target_considers_capital_inputs
        )
        self.intermediate_inputs_target_considers_capital_inputs = intermediate_inputs_target_considers_capital_inputs

        self.capital_inputs_target_considers_labour_inputs = clip(capital_inputs_target_considers_labour_inputs)
        self.capital_inputs_target_considers_labour_inputs = capital_inputs_target_considers_labour_inputs
        self.capital_inputs_target_considers_intermediate_inputs = clip(
            capital_inputs_target_considers_intermediate_inputs
        )
        self.capital_inputs_target_considers_intermediate_inputs = capital_inputs_target_considers_intermediate_inputs
        self.capital_inputs_target_considers_capital_inputs = clip(capital_inputs_target_considers_capital_inputs)
        self.capital_inputs_target_considers_capital_inputs = capital_inputs_target_considers_capital_inputs

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
        """Calculate target production levels for each firm.

        Args:
            current_estimated_demand (np.ndarray): Expected demand for each firm
            initial_inventory (np.ndarray): Starting inventory levels
            previous_inventory (np.ndarray): Previous period inventory
            previous_production (np.ndarray): Previous period production
            current_target_production (np.ndarray): Current production targets
            current_limiting_intermediate_inputs (np.ndarray): Input constraints
            current_limiting_capital_inputs (np.ndarray): Capital constraints
            current_firm_equity (np.ndarray): Current equity levels
            current_firm_debt (np.ndarray): Current debt levels
            previous_loans_applied_for (np.ndarray): Previous loan applications
            current_firm_deposits (np.ndarray): Current deposit balances
            interest_on_overdraft_rates (np.ndarray): Overdraft interest rates
            interest_paid_on_loans (np.ndarray): Interest payments on loans

        Returns:
            np.ndarray: Target production quantities for each firm
        """
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
        """Calculate constrained production targets considering intermediate input limitations.

        Args:
            previous_production (np.ndarray): Previous period production
            current_target_production (np.ndarray): Current production targets
            current_limiting_labour_inputs (np.ndarray): Labor constraints
            current_limiting_intermediate_inputs (np.ndarray): Input constraints
            current_limiting_capital_inputs (np.ndarray): Capital constraints
            current_firm_equity (np.ndarray): Current equity levels
            current_firm_debt (np.ndarray): Current debt levels
            previous_loans_applied_for (np.ndarray): Previous loan applications

        Returns:
            np.ndarray: Constrained production targets accounting for intermediate inputs
        """
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
        """Calculate constrained production targets considering capital input limitations.

        Args:
            previous_production (np.ndarray): Previous period production
            current_target_production (np.ndarray): Current production targets
            current_limiting_labour_inputs (np.ndarray): Labor constraints
            current_limiting_intermediate_inputs (np.ndarray): Input constraints
            current_limiting_capital_inputs (np.ndarray): Capital constraints
            current_firm_equity (np.ndarray): Current equity levels
            current_firm_debt (np.ndarray): Current debt levels
            previous_loans_applied_for (np.ndarray): Previous loan applications

        Returns:
            np.ndarray: Constrained production targets accounting for capital inputs
        """
        pass


class DefaultTargetProductionSetter(TargetProductionSetter):
    """Default implementation of target production setting.

    This class implements a production target setting strategy that considers:
    1. Expected demand adjusted for desired inventory levels
    2. Financial constraints based on debt/equity ratios
    3. Input constraints (labor, intermediate, capital) with configurable weights
    """

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
        """Calculate target production levels using the default strategy.

        Computes targets based on:
        1. Expected demand minus a fraction of existing inventory
        2. Plus a target buffer stock of inventory
        3. Adjusted for financial constraints if enabled

        Args:
            [same as abstract method]

        Returns:
            np.ndarray: Target production quantities for each firm
        """
        if self.financial_constrains_fraction == 0.0:
            return np.maximum(
                1e-12,
                current_estimated_demand
                - self.existing_inventory_fraction * previous_inventory
                + self.target_inventory_to_production_fraction * previous_production,
            )
        else:
            return np.maximum(
                1e-12,
                current_estimated_demand
                - self.existing_inventory_fraction * previous_inventory
                + self.target_inventory_to_production_fraction * previous_production
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
                    where=self.maximum_debt_to_equity_ratio * current_firm_equity
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
        """Adjust production targets based on intermediate input constraints.

        Modifies targets considering:
        1. Labor input constraints with configured weight
        2. Intermediate input constraints with configured weight
        3. Capital input constraints with configured weight

        Args:
            [same as abstract method]

        Returns:
            np.ndarray: Adjusted production targets considering input constraints
        """
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
            * (current_limiting_intermediate_inputs - current_target_production),
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
        """Adjust production targets based on capital input constraints.

        Modifies targets considering:
        1. Labor input constraints with configured weight
        2. Intermediate input constraints with configured weight
        3. Capital input constraints with configured weight

        Args:
            [same as abstract method]

        Returns:
            np.ndarray: Adjusted production targets considering capital constraints
        """
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
            * (current_limiting_intermediate_inputs - current_target_production),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.capital_inputs_target_considers_capital_inputs
            * (current_limiting_capital_inputs - current_target_production),
        )

        return current_target_production
