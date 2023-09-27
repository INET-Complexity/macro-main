from abc import abstractmethod, ABC

import numpy as np


class FirmWageSetter(ABC):
    @abstractmethod
    def set_wages(
        self,
        firm_employments: list[np.ndarray],
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
    ) -> np.ndarray:
        pass


class DefaultFirmWageSetter(FirmWageSetter):
    def __init__(
        self,
        labour_market_tightness_markup_scale: float,
        markup_time_span: int,
    ):
        self.labour_market_tightness_markup_scale = labour_market_tightness_markup_scale
        self.markup_time_span = markup_time_span

    def set_wages(
        self,
        firm_employments: list[np.ndarray],
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
    ) -> np.ndarray:
        new_individual_wages = previous_employee_income.copy()
        for firm_ind in range(len(firm_employments)):
            # Labour market tightness mark-up
            markup = (
                self.labour_market_tightness_markup_scale
                * 1.0
                / self.markup_time_span
                * sum(
                    [
                        max(
                            0,
                            (
                                historic_desired_labour_inputs[-t][firm_ind]
                                - historic_realised_labour_inputs[-t][firm_ind]
                            )
                            / historic_desired_labour_inputs[-t][firm_ind],
                        )
                        if historic_desired_labour_inputs[-t][firm_ind] > 0
                        else 0.0
                        for t in range(
                            1,
                            min(
                                len(historic_desired_labour_inputs),
                                self.markup_time_span + 1,
                            ),
                        )
                    ]
                )
            )

            # Update wages
            if current_individual_labour_inputs[firm_employments[firm_ind]].sum() == 0.0:
                new_individual_wages[firm_employments[firm_ind]] = 0.0
            else:
                new_individual_wages[firm_employments[firm_ind]] = (
                    (1 + markup)
                    * previous_employee_income[firm_employments[firm_ind]].sum()
                    / current_individual_labour_inputs[firm_employments[firm_ind]].sum()
                    * current_individual_labour_inputs[firm_employments[firm_ind]]
                )

        return new_individual_wages
