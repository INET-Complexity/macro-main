import numpy as np

from inet_macromodel.util.partition import partition_into_quintiles

from abc import abstractmethod, ABC


class HouseholdConsumption(ABC):
    @abstractmethod
    def compute_target_consumption_before_ce(
        self,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        historic_consumption: list[np.ndarray],
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        pass

    def compute_target_consumption(
        self,
        income: np.ndarray,
        target_consumption_before_ce: np.ndarray,
        target_consumption_ce: np.ndarray,
        target_consumption_expansion_loans: np.ndarray,
        received_consumption_expansion_loans: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
    ) -> np.ndarray:
        pass


class DefaultHouseholdConsumption(HouseholdConsumption):
    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
    ):
        self.consumption_smoothing_fraction = consumption_smoothing_fraction
        self.consumption_smoothing_window = consumption_smoothing_window

    def compute_target_consumption_before_ce(
        self,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        historic_consumption: list[np.ndarray],
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        smoothing_window = min(self.consumption_smoothing_window, len(historic_consumption))
        target_consumption = np.zeros((len(income), len(consumption_weights)))
        quintiles = partition_into_quintiles(income)
        historic_consumption_sum = np.array(historic_consumption)[-smoothing_window:].sum(axis=0)
        for q in range(5):
            cons = (
                consumption_weights_by_income[:, q]
                if take_consumption_weights_by_income_quantile
                else consumption_weights
            )
            ind = np.where(quintiles == q)[0]
            target_consumption[ind] = (
                1.0
                / (1 + tau_vat)
                * np.outer(
                    cons,
                    np.maximum(
                        (1 - saving_rates[ind]) * household_benefits[ind],
                        (1 - saving_rates[ind]) * income[ind],
                        self.consumption_smoothing_fraction
                        * (1 + tau_vat)
                        * (1 / smoothing_window)
                        * historic_consumption_sum[ind],
                    ),
                ).T
            )
        return target_consumption

    def compute_target_consumption(
        self,
        income: np.ndarray,
        target_consumption_before_ce: np.ndarray,
        target_consumption_ce: np.ndarray,
        target_consumption_expansion_loans: np.ndarray,
        received_consumption_expansion_loans: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
    ) -> np.ndarray:
        target_consumption = np.zeros((len(income), len(consumption_weights)))
        quintiles = partition_into_quintiles(income)
        for q in range(5):
            cons = (
                consumption_weights_by_income[:, q]
                if take_consumption_weights_by_income_quantile
                else consumption_weights
            )
            ind = np.where(quintiles == q)[0]
            target_consumption[ind] = (
                target_consumption_before_ce[ind]
                + np.outer(
                    cons,
                    target_consumption_ce[ind]
                    * np.divide(
                        received_consumption_expansion_loans[ind],
                        target_consumption_expansion_loans[ind],
                        out=np.ones_like(received_consumption_expansion_loans[ind]),
                        where=target_consumption_expansion_loans[ind] != 0.0,
                    ),
                ).T
            )
        return target_consumption
