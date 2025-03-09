"""Base agent implementation for the macroeconomic model.

This module provides the foundational Agent class that all economic actors
(firms, households, banks, etc.) inherit from. It implements core functionality for:
- Tracking agent states over time
- Participating in goods market transactions
- Managing transaction records
- Handling data persistence
"""

from copy import deepcopy
from typing import Any, Optional

import numpy as np
from numba import float64, int64, njit

from macromodel.timeseries import TimeSeries


class Agent:
    """Base class for all economic agents in the simulation.

    This class provides core functionality for economic agents, particularly
    focused on goods market participation and state tracking. It manages both
    buying and selling activities, maintaining transaction records and time series
    data for various economic variables.

    Attributes:
        country_name (str): Country the agent belongs to
        all_country_names (list[str]): All countries in the simulation
        n_industries (int): Number of industries in the economy
        n_transactors_sell (int): Number of selling entities for this agent
        n_transactors_buy (int): Number of buying entities for this agent
        states (dict[str, Any]): Current state variables for the agent
        initial_states (dict[str, Any]): Initial state variables (for resets)
        transactor_settings (dict[str, Any]): Settings for market transactions
        transactor_buyer_states (dict): Current state for buying activities
        transactor_seller_states (dict): Current state for selling activities
        exchange_rate_usd_to_lcu (float): Exchange rate from USD to local currency
        ts (TimeSeries): Time series data for the agent's variables
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        n_transactors_sell: int,
        n_transactors_buy: int,
        ts: TimeSeries,
        states: dict[str, Any],
        transactor_settings: Optional[dict[str, Any]] = None,
    ):
        """Initialize a new economic agent.

        Args:
            country_name (str): Country the agent belongs to
            all_country_names (list[str]): All countries in the simulation
            n_industries (int): Number of industries in the economy
            n_transactors_sell (int): Number of selling entities for this agent
            n_transactors_buy (int): Number of buying entities for this agent
            ts (TimeSeries): Time series container for the agent's variables
            states (dict[str, Any]): Initial state variables
            transactor_settings (Optional[dict[str, Any]]): Settings for market transactions
        """
        self.country_name = country_name
        self.all_country_names = all_country_names
        self.n_industries = n_industries
        self.n_transactors_sell = n_transactors_sell
        self.n_transactors_buy = n_transactors_buy
        self.states = states

        self.initial_states = deepcopy(states)

        self.transactor_settings = transactor_settings if transactor_settings else {}

        self.transactor_buyer_states = {}
        self.transactor_seller_states = {}
        self.exchange_rate_usd_to_lcu = None

        # Initiate the time series
        self.ts = ts
        self.initiate_ts()

    def initiate_ts(self) -> None:
        """Initialize time series variables for goods market transactions.

        Sets up tracking for:
        - Sales amounts (real and nominal) by country
        - Purchase amounts (real and nominal) by country
        - Excess demand
        """
        if self.n_transactors_buy > 0 and self.n_transactors_sell > 0:
            self.ts["real_amount_sold"] = np.full(self.n_transactors_sell, np.nan)
            for country_name in self.all_country_names:
                self.ts["real_amount_sold_to_" + country_name] = np.full(self.n_transactors_sell, np.nan)
            self.ts["nominal_amount_sold_in_lcu"] = np.full(self.n_transactors_sell, np.nan)
            for country_name in self.all_country_names:
                self.ts["nominal_amount_sold_in_lcu_to_" + country_name] = np.full(self.n_transactors_sell, np.nan)
            self.ts["real_excess_demand"] = np.full(self.n_transactors_sell, np.nan)

            self.ts["nominal_amount_spent_in_usd"] = np.full((self.n_transactors_buy, self.n_industries), np.nan)
            for country_name in self.all_country_names:
                self.ts["nominal_amount_spent_in_usd_to_" + country_name] = np.full(
                    (self.n_transactors_buy, self.n_industries), np.nan
                )
            self.ts["nominal_amount_spent_in_lcu"] = np.full((self.n_transactors_buy, self.n_industries), np.nan)
            for country_name in self.all_country_names:
                self.ts["nominal_amount_spent_in_lcu_to_" + country_name] = np.full(
                    (self.n_transactors_buy, self.n_industries), np.nan
                )
            self.ts["real_amount_bought"] = np.full((self.n_transactors_buy, self.n_industries), np.nan)
            for country_name in self.all_country_names:
                self.ts["real_amount_bought_from_" + country_name] = np.full(
                    (self.n_transactors_buy, self.n_industries), np.nan
                )

    def __str__(self):
        """Get string representation of the agent.

        Returns:
            str: Agent class name and country
        """
        return f"{self.__class__.__name__}({self.country_name})"

    def set_goods_to_buy(self, buy_init: np.ndarray) -> None:
        """Set initial goods quantities to buy.

        Args:
            buy_init (np.ndarray): Initial quantities to buy
        """
        self.transactor_buyer_states["Initial Goods"] = buy_init

    def set_goods_to_sell(self, sell_init: np.ndarray) -> None:
        """Set initial goods quantities to sell.

        Args:
            sell_init (np.ndarray): Initial quantities to sell
        """
        self.transactor_seller_states["Initial Goods"] = sell_init

    def set_maximum_excess_demand(self, max_excess_demand: np.ndarray) -> None:
        """Set maximum allowed excess demand.

        Args:
            max_excess_demand (np.ndarray): Maximum excess demand values
        """
        self.transactor_seller_states["Remaining Excess Goods"] = max_excess_demand

    def set_prices(self, sell_price: np.ndarray) -> None:
        """Set selling prices for goods.

        Args:
            sell_price (np.ndarray): Prices for goods to sell
        """
        self.transactor_seller_states["Prices"] = sell_price

    def set_seller_industries(self, industries: np.ndarray) -> None:
        """Set industry assignments for sellers.

        Args:
            industries (np.ndarray): Industry IDs for sellers
        """
        self.transactor_seller_states["Industries"] = industries

    def set_exchange_rate(self, exchange_rate_usd_to_lcu: float) -> None:
        """Set the exchange rate from USD to local currency.

        Args:
            exchange_rate_usd_to_lcu (float): Exchange rate value
        """
        self.exchange_rate_usd_to_lcu = exchange_rate_usd_to_lcu

    def prepare(self) -> None:
        """Prepare the agent for goods market transactions.

        This method initializes all necessary states for participation in the goods market.
        It sets up both buying and selling states, including:

        Buyer States:
        - Value Type: How values are interpreted (e.g., nominal vs real)
        - Priority: Transaction priority in the market clearing process
        - Remaining Goods: Copy of initial goods to track unfulfilled purchases
        - Nominal Amount spent: Tracks spending by industry
        - Real Amount bought: Tracks quantities bought by industry

        Seller States:
        - Value Type: How values are interpreted (e.g., nominal vs real)
        - Priority: Transaction priority in the market clearing process
        - Remaining Goods: Copy of initial goods to track unsold inventory
        - Real Amount sold: Tracks quantities sold
        - Real Excess Demand: Tracks excess demand for goods

        All transaction amounts are tracked separately for each country to enable
        detailed international trade analysis.
        """
        # Value type and priority
        self.transactor_buyer_states["Value Type"] = self.transactor_settings["Buyer Value Type"]
        self.transactor_buyer_states["Priority"] = self.transactor_settings["Buyer Priority"]
        self.transactor_seller_states["Value Type"] = self.transactor_settings["Seller Value Type"]
        self.transactor_seller_states["Priority"] = self.transactor_settings["Seller Priority"]

        # Remaining goods to buy or to sell
        self.transactor_buyer_states["Remaining Goods"] = self.transactor_buyer_states["Initial Goods"].copy()
        self.transactor_seller_states["Remaining Goods"] = self.transactor_seller_states["Initial Goods"].copy()

        # Amount sold
        self.transactor_seller_states["Real Amount sold"] = np.zeros(
            self.transactor_seller_states["Initial Goods"].shape
        )
        for country_name in self.all_country_names:
            self.transactor_seller_states["Real Amount sold to " + country_name] = np.zeros(
                self.transactor_seller_states["Initial Goods"].shape
            )

        # Amount spent
        self.transactor_buyer_states["Nominal Amount spent"] = np.zeros(
            self.transactor_buyer_states["Initial Goods"].shape
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name] = np.zeros(
                self.transactor_buyer_states["Initial Goods"].shape
            )

        # Amount bought
        self.transactor_buyer_states["Real Amount bought"] = np.zeros(
            self.transactor_buyer_states["Initial Goods"].shape
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Real Amount bought from " + country_name] = np.zeros(
                self.transactor_buyer_states["Initial Goods"].shape
            )

        # Excess demand
        self.transactor_seller_states["Real Excess Demand"] = np.zeros(
            self.transactor_seller_states["Initial Goods"].shape
        )

    def record(self, rounding: int = 16) -> None:
        """Record current transaction states to time series.

        This method processes and stores the results of goods market transactions,
        handling both domestic and international trade flows. It performs the following:

        1. Rounds all transaction values for numerical stability:
           - Real quantities (sold, bought, excess demand)
           - Nominal amounts (spent in both USD and local currency)

        2. Updates seller-side time series (if prices are set):
           - Real quantities sold (total and by destination country)
           - Nominal sales in local currency (total and by destination)
           - Excess demand tracking

        3. Updates buyer-side time series:
           - Nominal amounts spent (in both USD and local currency)
           - Real quantities bought (total and by source country)

        All monetary values are converted between USD and local currency using
        the current exchange rate. Values are stored in both currencies to
        facilitate both domestic and international economic analysis.

        Args:
            rounding (int, optional): Number of decimal places for rounding.
                Defaults to 16. Higher precision is used to minimize cumulative
                rounding errors in large-scale simulations.
        """
        # Round
        self.transactor_seller_states["Real Amount sold"] = round_pos(
            self.transactor_seller_states["Real Amount sold"], rounding
        )
        for country_name in self.all_country_names:
            self.transactor_seller_states["Real Amount sold to " + country_name] = round_pos(
                self.transactor_seller_states["Real Amount sold to " + country_name], rounding
            )
        self.transactor_seller_states["Real Excess Demand"] = round_pos(
            self.transactor_seller_states["Real Excess Demand"], rounding
        )
        self.transactor_buyer_states["Nominal Amount spent"] = round_pos2(
            self.transactor_buyer_states["Nominal Amount spent"], rounding
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name] = round_pos2(
                self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name], rounding
            )
        self.transactor_buyer_states["Real Amount bought"] = round_pos2(
            self.transactor_buyer_states["Real Amount bought"], rounding
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Real Amount bought from " + country_name] = round_pos2(
                self.transactor_buyer_states["Real Amount bought from " + country_name], rounding
            )

        # Update time series for sellers
        if "Prices" in self.transactor_seller_states.keys():
            self.ts.real_amount_sold.append(self.transactor_seller_states["Real Amount sold"])
            for country_name in self.all_country_names:
                self.ts.dicts["real_amount_sold_to_" + country_name].append(
                    self.transactor_seller_states["Real Amount sold to " + country_name]
                )
            self.ts.nominal_amount_sold_in_lcu.append(
                self.exchange_rate_usd_to_lcu
                * self.transactor_seller_states["Prices"]
                * self.transactor_seller_states["Real Amount sold"]
            )
            for country_name in self.all_country_names:
                self.ts.dicts["nominal_amount_sold_in_lcu_to_" + country_name].append(
                    self.exchange_rate_usd_to_lcu
                    * self.transactor_seller_states["Prices"]
                    * self.transactor_seller_states["Real Amount sold to " + country_name]
                )
            self.ts.real_excess_demand.append(self.transactor_seller_states["Real Excess Demand"])
        else:
            self.ts.real_amount_sold.append(np.full(self.n_transactors_sell, np.nan))
            for country_name in self.all_country_names:
                self.ts.dicts["real_amount_sold_to_" + country_name].append(np.full(self.n_transactors_sell, np.nan))
            self.ts.nominal_amount_sold_in_lcu.append(np.full(self.n_transactors_sell, np.nan))
            for country_name in self.all_country_names:
                self.ts.dicts["nominal_amount_sold_in_lcu_to_" + country_name].append(
                    np.full(self.n_transactors_sell, np.nan)
                )
            self.ts.real_excess_demand.append(np.full(self.n_transactors_sell, np.nan))

        # Update time series for buyers
        self.ts.nominal_amount_spent_in_usd.append(self.transactor_buyer_states["Nominal Amount spent"])
        for country_name in self.all_country_names:
            self.ts.dicts["nominal_amount_spent_in_usd_to_" + country_name].append(
                self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name]
            )
        self.ts.nominal_amount_spent_in_lcu.append(
            self.exchange_rate_usd_to_lcu * self.transactor_buyer_states["Nominal Amount spent"]
        )
        for country_name in self.all_country_names:
            self.ts.dicts["nominal_amount_spent_in_lcu_to_" + country_name].append(
                self.exchange_rate_usd_to_lcu
                * self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name]
            )
        self.ts.real_amount_bought.append(self.transactor_buyer_states["Real Amount bought"])
        for country_name in self.all_country_names:
            self.ts.dicts["real_amount_bought_from_" + country_name].append(
                self.transactor_buyer_states["Real Amount bought from " + country_name]
            )

    def gen_reset(self) -> None:
        """Reset the agent to its initial state.

        Restores initial states and resets time series data.
        """
        self.states = deepcopy(self.initial_states)
        self.ts.reset()


@njit(cache=True)
def round_pos(x: np.ndarray, decimals: int) -> np.ndarray:
    """Round values and ensure they are non-negative.

    Args:
        x (np.ndarray): Array to round
        decimals (int): Number of decimal places

    Returns:
        np.ndarray: Rounded, non-negative array
    """
    r = np.round(x, decimals)
    return np.maximum(0.0, r)


@njit(cache=True)
def round_pos2(x: np.ndarray, decimals: int) -> np.ndarray:
    """Round 2D array values and ensure they are non-negative.

    Args:
        x (np.ndarray): 2D array to round
        decimals (int): Number of decimal places

    Returns:
        np.ndarray: Rounded, non-negative array
    """
    r = np.round(x, decimals)
    return np.maximum(0.0, r)
