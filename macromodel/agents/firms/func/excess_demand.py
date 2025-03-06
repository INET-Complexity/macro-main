from abc import ABC, abstractmethod

import numpy as np


class ExcessDemandSetter(ABC):
    """Abstract base class for determining firms' excess demand levels.

    This class defines strategies for calculating excess demand based on:
    - Production constraints from various inputs
    - Target vs. actual production levels
    - Input-specific consideration weights

    Excess demand represents unfulfilled market demand due to:
    - Labor constraints
    - Intermediate input constraints
    - Capital input constraints

    Attributes:
        consider_intermediate_inputs (float): Weight given to intermediate
            input constraints (0 to 1)
        consider_capital_inputs (float): Weight given to capital input
            constraints (0 to 1)
        consider_labour_inputs (float): Weight given to labor input
            constraints (0 to 1)
    """

    def __init__(
        self,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
        consider_labour_inputs: float,
    ):
        """Initialize the excess demand setter with input consideration weights.

        Args:
            consider_intermediate_inputs (float): Weight for intermediate
                input constraints (clipped to [0,1])
            consider_capital_inputs (float): Weight for capital input
                constraints (clipped to [0,1])
            consider_labour_inputs (float): Weight for labor input
                constraints (clipped to [0,1])
        """
        self.consider_intermediate_inputs = max(0.0, min(1.0, consider_intermediate_inputs))
        self.consider_intermediate_inputs = consider_intermediate_inputs
        self.consider_capital_inputs = max(0.0, min(1.0, consider_capital_inputs))
        self.consider_capital_inputs = consider_capital_inputs
        self.consider_labour_inputs = max(0.0, min(1.0, consider_labour_inputs))
        self.consider_labour_inputs = consider_labour_inputs

    @abstractmethod
    def set_maximum_excess_demand(
        self,
        current_production: np.ndarray,
        target_production: np.ndarray,
        limiting_intermediate_inputs: np.ndarray,
        limiting_capital_inputs: np.ndarray,
        limiting_labour_inputs: np.ndarray,
    ) -> np.ndarray:
        """Calculate maximum excess demand considering all constraints.

        Determines how much additional demand can be satisfied beyond
        current production levels, subject to input constraints.

        Args:
            current_production (np.ndarray): Current production levels by firm
            target_production (np.ndarray): Desired production levels by firm
            limiting_intermediate_inputs (np.ndarray): Production possible with
                available intermediate inputs
            limiting_capital_inputs (np.ndarray): Production possible with
                available capital inputs
            limiting_labour_inputs (np.ndarray): Production possible with
                available labor inputs

        Returns:
            np.ndarray: Maximum excess demand that could be satisfied
        """
        pass


class ConstrainedExcessDemandSetter(ExcessDemandSetter):
    """Implementation of excess demand calculation with input constraints.

    This class implements a strategy that:
    1. Starts with target production levels
    2. Adjusts downward based on labor constraints
    3. Further adjusts for intermediate input constraints
    4. Finally considers capital input constraints

    The approach ensures that excess demand estimates are:
    - Feasible given input availability
    - Weighted by importance of each input type
    - Consistent with production technology
    """

    def set_maximum_excess_demand(
        self,
        current_production: np.ndarray,
        target_production: np.ndarray,
        limiting_intermediate_inputs: np.ndarray,
        limiting_capital_inputs: np.ndarray,
        limiting_labour_inputs: np.ndarray,
    ) -> np.ndarray:
        """Calculate constrained excess demand through sequential adjustment.

        Adjusts target production downward based on:
        1. Labor input constraints
        2. Intermediate input constraints
        3. Capital input constraints

        Each adjustment considers:
        - The gap between limiting and current production
        - The input-specific consideration weight
        - The remaining distance to target production

        Args:
            current_production (np.ndarray): Current production levels
            target_production (np.ndarray): Desired production levels
            limiting_intermediate_inputs (np.ndarray): Intermediate input capacity
            limiting_capital_inputs (np.ndarray): Capital input capacity
            limiting_labour_inputs (np.ndarray): Labor input capacity

        Returns:
            np.ndarray: Feasible production targets after all constraints
        """
        target_production = np.minimum(
            target_production,
            target_production
            + self.consider_labour_inputs * ((limiting_labour_inputs - current_production) - target_production),
        )
        target_production = np.minimum(
            target_production,
            target_production
            + self.consider_intermediate_inputs
            * ((limiting_intermediate_inputs - current_production) - target_production),
        )
        target_production = np.minimum(
            target_production,
            target_production
            + self.consider_capital_inputs * ((limiting_capital_inputs - current_production) - target_production),
        )

        return target_production


class ZeroExcessDemandSetter(ExcessDemandSetter):
    """Implementation that assumes no excess demand.

    This class implements a simplified strategy where:
    - All excess demand is ignored
    - Only current production is considered
    - No input constraints are evaluated

    This approach is useful for:
    - Model testing and validation
    - Scenarios focusing on supply-side dynamics
    - Simplified economic models
    """

    def set_maximum_excess_demand(
        self,
        current_production: np.ndarray,
        target_production: np.ndarray,
        limiting_intermediate_inputs: np.ndarray,
        limiting_capital_inputs: np.ndarray,
        limiting_labour_inputs: np.ndarray,
    ) -> np.ndarray:
        """Return zero excess demand regardless of inputs.

        A simplified implementation that assumes no excess demand exists,
        useful for testing or when demand is not the focus.

        Args:
            current_production (np.ndarray): Current production levels
            target_production (np.ndarray): Desired production levels (unused)
            limiting_intermediate_inputs (np.ndarray): Intermediate capacity (unused)
            limiting_capital_inputs (np.ndarray): Capital capacity (unused)
            limiting_labour_inputs (np.ndarray): Labor capacity (unused)

        Returns:
            np.ndarray: Zero array matching production shape
        """
        return np.zeros(current_production.shape)
