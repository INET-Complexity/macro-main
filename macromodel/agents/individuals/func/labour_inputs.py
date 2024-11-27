from abc import ABC, abstractmethod

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus


class IndividualLabourInputsSetter(ABC):
    @abstractmethod
    def update_labour_inputs(
        self,
        previous_individuals_labour_inputs: np.ndarray,
        current_individuals_activity: np.ndarray,
    ) -> np.ndarray:
        pass


class ScaledIndividualsProductivitySetter(IndividualLabourInputsSetter):
    def __init__(
        self,
        increase_employed: float,
        decrease_unemployed: float,
    ):
        self.increase_employed = increase_employed
        self.decrease_unemployed = decrease_unemployed

    def update_labour_inputs(
        self,
        previous_individuals_labour_inputs: np.ndarray,
        current_individuals_activity: np.ndarray,
    ) -> np.ndarray:
        current_labour_inputs = previous_individuals_labour_inputs.copy()
        current_labour_inputs[current_individuals_activity == ActivityStatus.EMPLOYED] *= 1 + self.increase_employed
        current_labour_inputs[current_individuals_activity == ActivityStatus.UNEMPLOYED] /= 1 + self.decrease_unemployed

        return current_labour_inputs
