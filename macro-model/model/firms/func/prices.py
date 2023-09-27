import numpy as np

from abc import abstractmethod, ABC


class PriceSetter(ABC):
    def __init__(self, price_setting_noise_std: float, price_setting_speed: float):
        self.price_setting_noise_std = price_setting_noise_std
        self.price_setting_speed = price_setting_speed

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
    ) -> np.ndarray:
        pass


class ConstantPriceSetter(PriceSetter):
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
    ) -> np.ndarray:
        return np.ones_like(prev_prices)


class SupplyDemandPriceSetter(PriceSetter):
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
    ) -> np.ndarray:
        new_prices = np.zeros_like(prev_prices)
        for firm_id in range(prev_prices.shape[0]):
            if excess_demand[firm_id] > 0:
                if inventories[firm_id] + production[firm_id] == 0.0:
                    new_prices[firm_id] = prev_prices[firm_id] * (
                        1 + np.random.normal(0.0, self.price_setting_noise_std)
                    )
                else:
                    new_prices[firm_id] = prev_prices[firm_id] * (
                        1
                        + (excess_demand[firm_id] / (inventories[firm_id] + production[firm_id]))
                        * (1 + np.random.normal(0.0, self.price_setting_noise_std))
                    )
            elif inventories[firm_id] > 0:
                if inventories[firm_id] + production[firm_id] == 0.0:
                    new_prices[firm_id] = prev_prices[firm_id] * (
                        1 + np.random.normal(0.0, self.price_setting_noise_std)
                    )
                else:
                    new_prices[firm_id] = prev_prices[firm_id] * (
                        1
                        - (inventories[firm_id] / (inventories[firm_id] + production[firm_id]))
                        * (1 + np.random.normal(0.0, self.price_setting_noise_std))
                    )
            else:
                new_prices[firm_id] = prev_prices[firm_id] * (1 + np.random.normal(0.0, self.price_setting_noise_std))
        return (1 + current_estimated_ppi_inflation) * new_prices


class CANVASPriceSetter(PriceSetter):
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
    ) -> np.ndarray:
        # Demand-pull inflation
        demand_pull_inflation = np.zeros_like(prev_firm_prices)
        ind_canvas = np.logical_or(
            np.logical_and(
                prev_supply <= prev_demand, prev_firm_prices < prev_average_good_prices[current_firm_sectors]
            ),
            np.logical_and(
                prev_supply > prev_demand, prev_firm_prices >= prev_average_good_prices[current_firm_sectors]
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

        # Cost-push inflation
        cost_push_inflation = curr_unit_costs / prev_unit_costs - 1.0

        return (
            prev_prices
            * (1 + self.price_setting_speed * current_estimated_ppi_inflation)
            * (1 + self.price_setting_speed * demand_pull_inflation)
            * (1 + self.price_setting_speed * cost_push_inflation)
        )
