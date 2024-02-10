import numpy as np
from abc import abstractmethod, ABC

from macromodel.individuals.individual_properties import ActivityStatus


class IncomeSetter(ABC):
    @abstractmethod
    def compute_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultIncomeSetter(IncomeSetter):
    def compute_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
    ) -> np.ndarray:
        income = np.zeros_like(current_individual_activity_status)

        # Employed individuals
        emp_ind = current_individual_activity_status == ActivityStatus.EMPLOYED
        income[emp_ind] = current_wage[emp_ind] + individual_social_benefits[emp_ind]

        # Unemployed individuals
        unemp_ind = current_individual_activity_status == ActivityStatus.UNEMPLOYED
        income[unemp_ind] = individual_social_benefits[unemp_ind]

        # Not-economically active individuals
        nea_ind = current_individual_activity_status == ActivityStatus.NOT_ECONOMICALLY_ACTIVE
        income[nea_ind] = individual_social_benefits[nea_ind]

        return income
