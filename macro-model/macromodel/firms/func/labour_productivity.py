import numpy as np

from abc import abstractmethod, ABC


class LabourProductivitySetter(ABC):
    def __init__(
        self,
        max_increase_in_work_effort: float,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
        work_effort_increase_speed: float,
    ):
        self.max_increase_in_work_effort = max_increase_in_work_effort
        self.consider_intermediate_inputs = consider_intermediate_inputs
        self.consider_capital_inputs = consider_capital_inputs
        self.work_effort_increase_speed = work_effort_increase_speed

    @abstractmethod
    def compute_labour_productivity_factor(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
    ) -> np.ndarray:
        pass


class WorkEffortLabourProductivitySetter(LabourProductivitySetter):
    def compute_labour_productivity_factor(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
    ) -> np.ndarray:
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.consider_intermediate_inputs
            * (
                current_limiting_intermediate_inputs - current_target_production
            ),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.consider_capital_inputs
            * (current_limiting_capital_inputs - current_target_production),
        )
        return 1.0 + self.work_effort_increase_speed * (
            np.minimum(
                self.max_increase_in_work_effort,
                current_target_production
                / (
                    labour_inputs_from_employees
                    * industry_labour_productivity_by_firm
                ),
            )
            - 1.0
        )
