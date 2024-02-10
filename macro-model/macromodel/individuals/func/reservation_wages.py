import numpy as np
from abc import abstractmethod, ABC

from macromodel.individuals.individual_properties import ActivityStatus


class ReservationWageSetter(ABC):
    @abstractmethod
    def compute_reservation_wages(
        self,
        historic_wages: np.ndarray,
        current_individuals_activity: np.ndarray,
        current_unemployment_benefits_by_individual: float,
    ) -> np.ndarray:
        pass


class DefaultReservationWageSetter(ReservationWageSetter):
    def __init__(
        self,
        unemployed_reservation_wage_timespan: int,
    ):
        self.unemployed_reservation_wage_timespan = unemployed_reservation_wage_timespan

    def compute_reservation_wages(
        self,
        historic_wages: np.ndarray,
        current_individuals_activity: np.ndarray,
        current_unemployment_benefits_by_individual: float,
    ) -> np.ndarray:
        reservation_wages = np.zeros_like(current_individuals_activity)

        # For employed individuals
        reservation_wages[current_individuals_activity == ActivityStatus.EMPLOYED] = historic_wages[-1][
            current_individuals_activity == ActivityStatus.EMPLOYED
        ]

        # For unemployed individuals
        if self.unemployed_reservation_wage_timespan == 0:
            reservation_wages[current_individuals_activity == ActivityStatus.UNEMPLOYED] = (
                current_unemployment_benefits_by_individual
            )
        else:
            reservation_wages[current_individuals_activity == ActivityStatus.UNEMPLOYED] = np.maximum(
                current_unemployment_benefits_by_individual,
                np.array(historic_wages[-self.unemployed_reservation_wage_timespan :])[
                    :, current_individuals_activity == ActivityStatus.UNEMPLOYED
                ].mean(axis=0),
            )

        return reservation_wages
