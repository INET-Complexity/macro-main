from abc import ABC, abstractmethod

import numpy as np
from numba import boolean, float64, int64, njit


class HouseholdConsumption(ABC):
    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
    ):
        self.consumption_smoothing_fraction = consumption_smoothing_fraction
        self.consumption_smoothing_window = consumption_smoothing_window
        self.minimum_consumption_fraction = minimum_consumption_fraction

    @abstractmethod
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        pass


class DefaultHouseholdConsumption(HouseholdConsumption):
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        return self._compute_target_consumption(
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            take_consumption_weights_by_income_quantile=take_consumption_weights_by_income_quantile,
            tau_vat=tau_vat,
            consumption_smoothing_window=self.consumption_smoothing_window,
            consumption_smoothing_fraction=self.consumption_smoothing_fraction,
            minimum_consumption_fraction=self.minimum_consumption_fraction,
        )

    @staticmethod
    # @njit(
    #     float64[:, :](
    #         float64[:, :],  # historic_consumption_sum
    #         float64[:],  # saving_rates
    #         float64[:],  # income
    #         float64[:],  # household_benefits
    #         float64[:],  # consumption_weights
    #         float64[:, :],  # consumption_weights_by_income
    #         boolean,  # take_consumption_weights_by_income_quantile
    #         float64,  # tau_vat
    #         int64,  # consumption_smoothing_window
    #         float64,  # consumption_smoothing_fraction
    #         float64,  # minimum_consumption_fraction
    #     ),
    #     cache=True,
    # )
    @njit(cache=True)
    def _compute_target_consumption(
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,  # noqa
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        consumption_smoothing_window: int,
        consumption_smoothing_fraction: float,
        minimum_consumption_fraction: float,
    ) -> np.ndarray:
        smoothing_window = min(consumption_smoothing_window, len(historic_consumption_sum))
        # target_consumption = np.zeros((len(income), len(consumption_weights)))
        # if take_consumption_weights_by_income_quantile:
        #     pass
        #     """
        #     quintiles = partition_into_quintiles(income)
        #     historic_consumption_sum = np.array(historic_consumption)[1:][-smoothing_window:].sum(axis=0)
        #     for q in range(5):
        #         cons = (
        #             consumption_weights_by_income[:, q]
        #             if take_consumption_weights_by_income_quantile
        #             else consumption_weights
        #         )
        #         ind = np.where(quintiles == q)[0]
        #         target_consumption[ind] = (
        #             1.0
        #             / (1 + tau_vat)
        #             * np.outer(
        #                 cons,
        #                 np.maximum(
        #                     minimum_consumption_fraction * (1 - saving_rates[ind]) * household_benefits[ind],
        #                     (1 - saving_rates[ind]) * income[ind],
        #                     consumption_smoothing_fraction
        #                     * (1 + tau_vat)
        #                     * (1 / smoothing_window)
        #                     * historic_consumption_sum[ind],
        #                 ),
        #             ).T
        #         )
        #     return np.maximum(0.0, target_consumption)
        #     """
        # else:
        target_consumption = (
            1.0
            / (1 + tau_vat)
            * np.outer(
                consumption_weights,
                np.maximum(
                    minimum_consumption_fraction * (1 - saving_rates) * household_benefits,
                    (1 - saving_rates) * income,
                    consumption_smoothing_fraction
                    * (1 + tau_vat)
                    * (1 / smoothing_window)
                    * historic_consumption_sum[1:][-smoothing_window:].sum(axis=0),
                ),
            ).T
        )
        return np.maximum(0.0, target_consumption)


class ExogenousHouseholdConsumption(HouseholdConsumption):
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        target_consumption = np.maximum(
            0.0,
            (
                1.0
                / (1 + tau_vat)
                * np.outer(
                    consumption_weights,
                    (1 - saving_rates) * income,
                ).T
            ),
        )
        return (
            (1 + expected_inflation)
            * current_cpi
            / initial_cpi
            * 1.0
            / (1 + tau_vat)
            * exogenous_total_consumption[current_time]
            * target_consumption
            / target_consumption.sum()
        )
