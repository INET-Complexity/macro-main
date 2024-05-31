import numpy as np

from abc import abstractmethod, ABC


class GrowthEstimator(ABC):
    @abstractmethod
    def compute_growth(
        self,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultGrowthEstimator(GrowthEstimator):
    def compute_growth(
        self,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
    ) -> np.ndarray:
        average_price_by_firm = prev_average_good_prices[current_firm_sectors]
        firm_growth_rates = np.zeros_like(prev_firm_prices)
        ind_canvas = np.logical_or(
            np.logical_and(
                prev_supply <= prev_demand,
                prev_firm_prices >= average_price_by_firm,
            ),
            np.logical_and(
                prev_supply > prev_demand,
                prev_firm_prices < average_price_by_firm,
            ),
        )
        firm_growth_rates[ind_canvas] = (
            np.divide(
                prev_demand[ind_canvas],
                prev_supply[ind_canvas],
                out=np.ones_like(prev_demand[ind_canvas]),
                where=prev_supply[ind_canvas] != 0.0,
            )
            - 1.0
        )
        return firm_growth_rates
