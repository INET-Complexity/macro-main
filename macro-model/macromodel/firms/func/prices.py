from abc import ABC, abstractmethod

import numpy as np


class PriceSetter(ABC):
    def __init__(
        self,
        price_setting_noise_std: float,
        price_setting_speed_gf: float,
        price_setting_speed_dp: float,
        price_setting_speed_cp: float,
    ):
        self.price_setting_noise_std = price_setting_noise_std
        self.price_setting_speed_gf = max(0.0, min(1.0, price_setting_speed_gf))
        self.price_setting_speed_gf = price_setting_speed_gf
        self.price_setting_speed_dp = max(0.0, min(1.0, price_setting_speed_dp))
        self.price_setting_speed_dp = price_setting_speed_dp
        self.price_setting_speed_cp = max(0.0, min(1.0, price_setting_speed_cp))
        self.price_setting_speed_cp = price_setting_speed_cp

    @abstractmethod
    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
    ) -> np.ndarray:
        pass


class DefaultPriceSetter(PriceSetter):
    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        min_inflation: float = -0.1,
        max_inflation: float = 0.1,
    ) -> np.ndarray:
        average_price_by_firm = prev_average_good_prices[current_firm_sectors]

        # Demand-pull inflation
        demand_pull_inflation = np.zeros_like(prev_firm_prices)
        ind_canvas = np.logical_or(
            np.logical_and(
                prev_supply <= prev_demand,
                prev_firm_prices < average_price_by_firm,
            ),
            np.logical_and(
                prev_supply > prev_demand,
                prev_firm_prices >= average_price_by_firm,
            ),
        )
        demand_pull_inflation[ind_canvas] = (
            np.divide(
                prev_demand[ind_canvas],
                prev_supply[ind_canvas],
                out=np.ones_like(prev_demand[ind_canvas]),
                where=prev_supply[ind_canvas] != 0.0,
            )
            - 1.0
        )
        demand_pull_inflation = np.maximum(min_inflation, np.minimum(max_inflation, demand_pull_inflation))

        # Cost-push inflation
        cost_push_inflation = (
            np.divide(
                curr_unit_costs,
                average_price_by_firm,
                out=np.ones_like(curr_unit_costs),
                where=average_price_by_firm != 0.0,
            )
            - 1.0
        )
        cost_push_inflation = np.maximum(min_inflation, np.minimum(max_inflation, cost_push_inflation))

        return np.maximum(
            1e-2,
            prev_prices
            * (1 + np.random.normal(0.0, self.price_setting_noise_std, prev_prices.shape))
            * (1 + self.price_setting_speed_gf * current_estimated_ppi_inflation)
            * (1 + self.price_setting_speed_dp * demand_pull_inflation)
            * (1 + self.price_setting_speed_cp * cost_push_inflation),
        )


class ExogenousPriceSetter(PriceSetter):
    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        min_inflation: float = -0.1,
        max_inflation: float = 0.1,
    ) -> np.ndarray:
        return ppi_during[current_time]
