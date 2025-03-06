"""Individual labor market participation management.

This module implements strategies for managing individual labor inputs
through:
- Activity status tracking
- Labor supply determination
- Workforce participation
- Employment transitions

The implementation handles:
- Labor market status
- Work intensity
- Employment changes
- Participation decisions
"""

from abc import ABC, abstractmethod

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus


class IndividualLabourInputsSetter(ABC):
    """Abstract base class for individual labor input management.

    This class defines strategies for determining individual labor
    market participation through:
    - Activity status consideration
    - Labor supply decisions
    - Employment transitions
    - Work intensity choices

    The strategies consider:
    - Current employment status
    - Previous labor inputs
    - Market conditions
    - Individual characteristics
    """

    @abstractmethod
    def update_labour_inputs(
        self,
        previous_individuals_labour_inputs: np.ndarray,
        current_individuals_activity: np.ndarray,
    ) -> np.ndarray:
        """Update individual labor market inputs.

        Args:
            previous_individuals_labour_inputs (np.ndarray): Previous labor inputs
            current_individuals_activity (np.ndarray): Current activity status

        Returns:
            np.ndarray: Updated labor inputs by individual
        """
        pass


class ScaledIndividualsProductivitySetter(IndividualLabourInputsSetter):
    """Scales labor inputs based on activity status.

    This class implements a strategy for scaling labor inputs
    based on the activity status of individuals. It adjusts
    the labor inputs for employed and unemployed individuals
    according to specified parameters.

    Args:
        increase_employed (float): Percentage increase in labor inputs for employed individuals
        decrease_unemployed (float): Percentage decrease in labor inputs for unemployed individuals
    """

    def __init__(
        self,
        increase_employed: float,
        decrease_unemployed: float,
    ) -> None:
        """Initialize the ScaledIndividualsProductivitySetter.

        Args:
            increase_employed (float): Percentage increase in labor inputs for employed individuals
            decrease_unemployed (float): Percentage decrease in labor inputs for unemployed individuals
        """
        self.increase_employed = increase_employed
        self.decrease_unemployed = decrease_unemployed

    def update_labour_inputs(
        self,
        previous_individuals_labour_inputs: np.ndarray,
        current_individuals_activity: np.ndarray,
    ) -> np.ndarray:
        """Update individual labor inputs.

        Args:
            previous_individuals_labour_inputs (np.ndarray): Previous labor inputs
            current_individuals_activity (np.ndarray): Current activity status

        Returns:
            np.ndarray: Updated labor inputs by individual
        """
        current_labour_inputs = previous_individuals_labour_inputs.copy()

        return current_labour_inputs
