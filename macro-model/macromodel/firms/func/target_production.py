import numpy as np

from abc import abstractmethod, ABC


class TargetProductionSetter(ABC):
    def __init__(
        self,
        existing_inventory_fraction: float,
        target_inventory_to_production_fraction: float,
        maximum_growth_rate: float,
        maximum_debt_to_equity_ratio: float,
        consider_financials: bool,
    ):
        self.existing_inventory_fraction = existing_inventory_fraction
        self.target_inventory_to_production_fraction = target_inventory_to_production_fraction
        self.maximum_growth_rate = maximum_growth_rate
        self.maximum_debt_to_equity_ratio = maximum_debt_to_equity_ratio
        self.consider_financials = consider_financials

    @abstractmethod
    def compute_unconstrained_target_production(
        self,
        current_estimated_demand: np.ndarray,
        initial_inventory: np.ndarray,
        previous_inventory: np.ndarray,
        initial_production: np.ndarray,
        previous_production: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_constrained_target_production(
        self,
        current_unconstrained_target_production: np.ndarray,
        current_limiting_stock: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_firm_production: np.ndarray,
        previous_loans_applied_for: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultTargetProductionSetter(TargetProductionSetter):
    def compute_unconstrained_target_production(
        self,
        current_estimated_demand: np.ndarray,
        initial_inventory: np.ndarray,
        previous_inventory: np.ndarray,
        initial_production: np.ndarray,
        previous_production: np.ndarray,
    ) -> np.ndarray:
        return np.minimum(
            np.maximum(
                1e-12,
                (
                    current_estimated_demand
                    - self.existing_inventory_fraction * previous_inventory
                    + self.target_inventory_to_production_fraction * previous_production
                ),
            ),
            (1 + self.maximum_growth_rate) * previous_production,
        )

    def compute_constrained_target_production(
        self,
        current_unconstrained_target_production: np.ndarray,
        current_limiting_stock: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_debt: np.ndarray,
        previous_firm_production: np.ndarray,
        previous_loans_applied_for: np.ndarray,
    ) -> np.ndarray:
        limited_by_stock = np.minimum(current_unconstrained_target_production, current_limiting_stock)
        if self.consider_financials:
            limited_by_financials = np.divide(
                previous_firm_production
                * (self.maximum_debt_to_equity_ratio * current_firm_equity - current_firm_debt),
                previous_loans_applied_for,
                out=np.full(current_firm_equity.shape, np.inf),
                where=previous_loans_applied_for != 0.0,
            )
        else:
            limited_by_financials = np.full(current_firm_debt.shape, np.inf)

        return np.maximum(0.0, np.amin([limited_by_stock, limited_by_financials], axis=0))
