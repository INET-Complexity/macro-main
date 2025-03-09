from abc import ABC, abstractmethod

import numpy as np


class LabourProductivitySetter(ABC):
    """Abstract base class for determining labor productivity adjustments.

    This class defines strategies for calculating labor productivity factors
    based on:
    - Work effort potential and limits
    - Input availability and constraints
    - Industry-specific productivity baselines
    - Adjustment speed parameters

    The productivity setting process considers:
    - Maximum allowable work effort increases
    - Complementarity with other inputs
    - Speed of productivity adjustments

    Attributes:
        max_increase_in_work_effort (float): Maximum allowed increase in
            productivity through work effort
        consider_intermediate_inputs (float): Weight given to intermediate
            input constraints (0 to 1)
        consider_capital_inputs (float): Weight given to capital input
            constraints (0 to 1)
        work_effort_increase_speed (float): Rate at which work effort
            adjustments are implemented
    """

    def __init__(
        self,
        max_increase_in_work_effort: float,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
        work_effort_increase_speed: float,
    ) -> None:
        """Initialize the labor productivity setter with adjustment parameters.

        Args:
            max_increase_in_work_effort (float): Maximum allowed increase in
                productivity through work effort
            consider_intermediate_inputs (float): Weight for intermediate
                input constraints (clipped to [0,1])
            consider_capital_inputs (float): Weight for capital input
                constraints (clipped to [0,1])
            work_effort_increase_speed (float): Rate of work effort adjustment
                implementation
        """
        self.max_increase_in_work_effort = max_increase_in_work_effort
        self.consider_intermediate_inputs = max(0.0, min(1.0, consider_intermediate_inputs))
        self.consider_intermediate_inputs = consider_intermediate_inputs
        self.consider_capital_inputs = max(0.0, min(1.0, consider_capital_inputs))
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
        """Calculate labor productivity adjustment factors for each firm.

        Determines appropriate productivity multipliers considering:
        - Production targets and constraints
        - Available inputs and their limitations
        - Current labor inputs and industry standards
        - Maximum allowable adjustments

        Args:
            current_target_production (np.ndarray): Target production levels
            current_limiting_intermediate_inputs (np.ndarray): Production possible
                with available intermediate inputs
            current_limiting_capital_inputs (np.ndarray): Production possible
                with available capital inputs
            labour_inputs_from_employees (np.ndarray): Current labor input levels
            industry_labour_productivity_by_firm (np.ndarray): Industry standard
                productivity levels by firm

        Returns:
            np.ndarray: Labor productivity adjustment factors by firm
        """
        pass


class WorkEffortLabourProductivitySetter(LabourProductivitySetter):
    """Implementation of productivity setting based on work effort.

    This class implements a strategy that:
    1. Adjusts production targets for input constraints
    2. Calculates required productivity increases
    3. Limits increases to feasible ranges
    4. Implements changes at specified speed

    The approach ensures that:
    - Productivity changes respect input complementarities
    - Adjustments stay within feasible bounds
    - Changes occur at appropriate speeds
    """

    def compute_labour_productivity_factor(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
    ) -> np.ndarray:
        """Calculate productivity factors based on work effort adjustments.

        The method:
        1. Adjusts targets for intermediate input constraints
        2. Further adjusts for capital input constraints
        3. Calculates required productivity increase
        4. Applies speed and maximum constraints

        Args:
            current_target_production (np.ndarray): Target production levels
            current_limiting_intermediate_inputs (np.ndarray): Intermediate
                input production capacity
            current_limiting_capital_inputs (np.ndarray): Capital input
                production capacity
            labour_inputs_from_employees (np.ndarray): Current labor inputs
            industry_labour_productivity_by_firm (np.ndarray): Industry
                standard productivity levels

        Returns:
            np.ndarray: Productivity adjustment factors, where 1.0 represents
                no change and values > 1.0 represent productivity increases
        """
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.consider_intermediate_inputs * (current_limiting_intermediate_inputs - current_target_production),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.consider_capital_inputs * (current_limiting_capital_inputs - current_target_production),
        )
        return 1.0 + self.work_effort_increase_speed * (
            np.minimum(
                self.max_increase_in_work_effort,
                current_target_production / (labour_inputs_from_employees * industry_labour_productivity_by_firm),
            )
            - 1.0
        )
