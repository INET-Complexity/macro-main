"""Rest of the World price determination module.

This module implements approaches for determining Rest of the World
prices in international markets. It provides mechanisms for:

1. Price Setting:
   - Initial price adjustment
   - Domestic price level response
   - Dynamic price convergence

2. Price Dynamics:
   - Inflation-based updating
   - Speed of adjustment
   - Price floor enforcement

The module implements inflation-based price setting that ensures
price level convergence while maintaining positive prices.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d

from macromodel.agents.firms.func.prices import PriceSetter


class RoWPriceSetter(ABC):
    """Abstract base class for Rest of World price determination.

    Provides interface for computing ROW prices based on domestic
    price levels and adjustment parameters.
    """

    @abstractmethod
    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute ROW prices.

        Args:
            initial_price (np.ndarray): Base prices
            aggregate_country_price_index (float): Domestic price level
            adjustment_speed (float): Price adjustment parameter

        Returns:
            np.ndarray: Computed ROW prices
        """
        pass


class InflationRoWPriceSetter(RoWPriceSetter):
    """Inflation-based price determination implementation.

    Adjusts prices based on domestic price level changes while
    ensuring a minimum positive price level.
    """

    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute prices using inflation adjustment.

        Adjusts initial prices based on:
        - Domestic price level deviations
        - Adjustment speed parameter
        - Minimum price floor (0.001)

        Args:
            initial_price (np.ndarray): Base prices
            aggregate_country_price_index (float): Domestic price level
            adjustment_speed (float): Price adjustment parameter

        Returns:
            np.ndarray: Adjusted prices with minimum floor
        """
        return np.maximum(
            1e-3,
            (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0)) * initial_price,
        )


class SectorExogenousROWPriceSetter(InflationRoWPriceSetter):
    """ROW price setter that overrides selected industries with exogenous price paths.

    Non-overridden industries follow the default inflation-adjustment rule.
    Overridden industries use a normalised exogenous sectoral price path:

        price[t] = initial_price[i] * (file_price[t] / file_price[initial_year])

    All ROW firms in an overridden industry receive the same path, anchored to
    their individual initial_price so relative price levels are preserved.

    The input CSV must have years as the index and industry codes as columns.
    Industry positions are resolved at runtime from their names, so multiple
    firms per industry are handled automatically.

    Attributes:
        firm_exo_prices: SectorExoPrices container holding sector-level price
            trajectories (injected after instantiation).
        overriden_industries: Per-firm ordered list of sector names matching
            the ROW price array (injected after instantiation). One entry per
            firm; all firms sharing a sector name receive the same exogenous path.
    """

    def __init__(self):
        self.firm_exo_prices = None
        self.overriden_industries: list[str] = []
        self._call_count: int = 0

    def _normalised_price(self, industry_name: str, current_quarter: int) -> float:
        """Interpolate the exogenous price for an industry and normalise to the initial year.

        current_quarter is 1-based (quarter 1 = Q1 of initial_year).
        Converts the quarterly index to a fractional calendar year, linearly interpolates
        the CSV price series, and divides by the value at initial_year.
        """
        initial_year = self.firm_exo_prices.initial_year
        series = self.firm_exo_prices.prices[industry_name]
        years = series.index.astype(float).values
        prices = series.values.astype(float)
        fn = interp1d(years, prices, bounds_error=False, fill_value="extrapolate")
        yr = initial_year + (current_quarter - 1) / 4
        return float(fn(yr)) / float(fn(initial_year))

    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute ROW prices, overriding listed sectors with exogenous sector paths.

        First computes the full inflation-adjusted price array via
        InflationRoWPriceSetter, then replaces the price of every ROW firm
        belonging to an overridden sector with the normalised exogenous sector price:

            price[i] = initial_price[i] * (sector_price[t] / sector_price[initial_year])

        The sector price trajectory is the same for all firms in that sector;
        initial_price anchors each firm individually so relative price levels
        within a sector are preserved. Call count is used as the time index since
        the ROW interface does not receive an explicit current_time argument.

        Args:
            initial_price: Per-firm anchor prices for each ROW firm.
            aggregate_country_price_index: Domestic price level (used by default rule).
            adjustment_speed: Price adjustment speed parameter (used by default rule).

        Returns:
            np.ndarray: Per-firm ROW prices; firms in overridden sectors follow the
                exogenous sector path, all others follow the inflation-adjustment rule.
        """
        price = super().compute_price(
            initial_price=initial_price,
            aggregate_country_price_index=aggregate_country_price_index,
            adjustment_speed=adjustment_speed,
        )

        current_quarter = self._call_count
        self._call_count += 1

        if self.firm_exo_prices is None or self.firm_exo_prices.prices is None or len(self.overriden_industries) == 0:
            return price

        for industry_name in self.firm_exo_prices.prices.columns:
            if industry_name not in self.overriden_industries:
                continue
            ratio = self._normalised_price(industry_name, current_quarter)
            for idx in [i for i, name in enumerate(self.overriden_industries) if name == industry_name]:
                price[idx] = initial_price[idx] * ratio

        return price


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
