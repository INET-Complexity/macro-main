from abc import ABC, abstractmethod

import numpy as np


class PriceSetter(ABC):
    """Abstract base class for determining firms' price-setting strategies.

    This class defines strategies for calculating prices based on:
    - Market conditions (supply, demand, inventories)
    - Cost factors (unit costs, inflation)
    - Competitive positioning (sector averages)
    - Adjustment speeds and noise

    The price setting process considers:
    - General inflation expectations
    - Demand-pull inflation pressures
    - Cost-push inflation pressures
    - Random price variations

    Attributes:
        price_setting_noise_std (float): Standard deviation of random
            price adjustments
        price_setting_speed_gf (float): Speed of general inflation
            pass-through (0 to 1)
        price_setting_speed_dp (float): Speed of demand-pull inflation
            adjustments (0 to 1)
        price_setting_speed_cp (float): Speed of cost-push inflation
            adjustments (0 to 1)
    """

    def __init__(
        self,
        price_setting_noise_std: float,
        price_setting_speed_gf: float,
        price_setting_speed_dp: float,
        price_setting_speed_cp: float,
    ):
        """Initialize the price setter with adjustment parameters.

        Args:
            price_setting_noise_std (float): Standard deviation of random
                price adjustments
            price_setting_speed_gf (float): Speed of general inflation
                pass-through (clipped to [0,1])
            price_setting_speed_dp (float): Speed of demand-pull inflation
                adjustments (clipped to [0,1])
            price_setting_speed_cp (float): Speed of cost-push inflation
                adjustments (clipped to [0,1])
        """
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
        """Calculate prices for each firm based on market conditions.

        Determines appropriate prices considering:
        - Previous prices and inflation expectations
        - Supply-demand balance and inventories
        - Cost changes and sector averages
        - Market positioning and competition

        Args:
            prev_prices (np.ndarray): Previous period's prices
            current_estimated_ppi_inflation (float): Expected PPI inflation
            excess_demand (np.ndarray): Excess demand by firm
            inventories (np.ndarray): Current inventory levels
            production (np.ndarray): Current production levels
            prev_average_good_prices (np.ndarray): Previous sector averages
            prev_firm_prices (np.ndarray): Previous firm-specific prices
            prev_supply (np.ndarray): Previous period's supply
            prev_demand (np.ndarray): Previous period's demand
            current_firm_sectors (np.ndarray): Sector ID for each firm
            curr_unit_costs (np.ndarray): Current unit costs
            prev_unit_costs (np.ndarray): Previous unit costs
            ppi_during (np.ndarray): PPI time series
            current_time (int): Current period index

        Returns:
            np.ndarray: Updated prices by firm
        """
        pass


class DefaultPriceSetter(PriceSetter):
    """Default implementation of price setting with multiple inflation sources.

    This class implements a strategy that adjusts prices based on:
    1. General inflation expectations
    2. Demand-pull inflation from market conditions
    3. Cost-push inflation from unit cost changes
    4. Random variations

    The approach ensures that:
    - Prices respond to market imbalances
    - Cost changes are passed through
    - Competitive positioning is maintained
    - Prices remain positive
    """

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
        """Calculate prices using the default multi-factor strategy.

        The method:
        1. Maps sector average prices to firms
        2. Calculates demand-pull inflation based on market position
        3. Calculates cost-push inflation from unit costs
        4. Combines all factors with random noise

        Price changes are allowed when either:
        - High price (>= sector avg) and excess supply
        - Low price (< sector avg) and excess demand

        Args:
            [same as parent class]
            min_inflation (float, optional): Lower bound on inflation rates.
                Defaults to -0.1 (-10%).
            max_inflation (float, optional): Upper bound on inflation rates.
                Defaults to 0.1 (10%).

        Returns:
            np.ndarray: Updated prices by firm, guaranteed to be positive
        """
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
    """Implementation of price setting using exogenous price paths.

    This class implements a simplified strategy where:
    - Prices follow a pre-determined path
    - Market conditions are ignored
    - Cost changes are ignored
    - No random variations are added

    This approach is useful for:
    - Model testing and validation
    - Policy analysis with controlled prices
    - Scenarios with external price determination
    """

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
        """Set prices according to exogenous PPI path.

        Simply returns the pre-determined PPI value for the current period,
        ignoring all market conditions and other parameters.

        Args:
            [same as parent class, all unused except:]
            ppi_during (np.ndarray): PPI time series
            current_time (int): Current period index
            min_inflation (float, optional): Unused. Defaults to -0.1.
            max_inflation (float, optional): Unused. Defaults to 0.1.

        Returns:
            np.ndarray: Price level from exogenous PPI path
        """
        return ppi_during[current_time]
