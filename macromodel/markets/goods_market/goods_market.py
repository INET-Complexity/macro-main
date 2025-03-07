"""Core implementation of the goods market mechanism.

This module provides the central implementation of the goods market, managing
the interaction between buyers and sellers across industries and countries.
It handles market clearing, supply chain tracking, and trade flows using
configurable clearing mechanisms and trade proportion management.

The market supports:
- Multiple clearing algorithms (default, pro-rata, water bucket)
- Priority-based buyer/seller matching
- Supply chain persistence and tracking
- International trade with origin/destination proportions
- Price and quantity adjustments
- Real and nominal value handling
"""

from copy import deepcopy
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from macromodel.agents.agent import Agent
from macromodel.configurations import GoodsMarketConfiguration
from macromodel.markets.goods_market.goods_market_ts import (
    create_goods_market_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions

SupplyChain = dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]]


class GoodsMarket:
    """Goods market implementation managing economic transactions.

    This class implements a comprehensive goods market system that handles
    transactions between buyers and sellers across multiple industries and
    countries. It manages market clearing, supply chain relationships, and
    trade flows using configurable mechanisms.

    The market operates in discrete steps:
    1. Initialization with industry structure and participants
    2. Setting of trade proportions and priorities
    3. Market clearing through configured mechanism
    4. Recording of transactions and supply chain updates

    Attributes:
        n_industries (int): Number of industries in the economy
        functions (dict[str, Any]): Market functions (clearing, etc.)
        ts (TimeSeries): Time series tracking market metrics
        goods_market_participants (dict[str, list[Agent]]): Market participants by country
        states (dict[str, Any]): Market state variables
        initial_states (dict[str, Any]): Initial market conditions
        buyer_priorities (dict[str, np.ndarray]): Buyer priority rankings
        seller_priorities (dict[str, np.ndarray]): Seller priority rankings
        row_index (int): Index for Rest of World in country arrays
    """

    def __init__(
        self,
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        goods_market_participants: dict[str, list[Agent]],
        states: dict[str, Any],
        buyer_priorities: dict[str, np.ndarray],
        seller_priorities: dict[str, np.ndarray],
        row_index: int,
    ):
        """Initialize the goods market.

        Args:
            n_industries (int): Number of industries
            functions (dict[str, Any]): Market functions
            ts (TimeSeries): Time series tracker
            goods_market_participants (dict[str, list[Agent]]): Market participants
            states (dict[str, Any]): Market states
            buyer_priorities (dict[str, np.ndarray]): Priority rankings for buyers by country.
                Higher values indicate higher priority in purchasing goods. These priorities
                affect the order in which buyers are matched with sellers during market
                clearing. For example, critical industries might have higher priorities
                to ensure they receive necessary inputs.
            seller_priorities (dict[str, np.ndarray]): Priority rankings for sellers by country.
                Higher values indicate higher priority in selling goods. These priorities
                influence which sellers are chosen first during market clearing. For example,
                domestic sellers might have higher priorities than foreign sellers to
                promote domestic trade.
            row_index (int): Rest of World index
        """
        self.n_industries = n_industries
        self.functions = functions
        self.ts = ts
        self.goods_market_participants = goods_market_participants
        self.states = states
        self.initial_states = deepcopy(states)
        self.buyer_priorities = buyer_priorities
        self.seller_priorities = seller_priorities
        self.row_index = row_index

    @classmethod
    def from_data(
        cls,
        n_industries: int,
        configuration: GoodsMarketConfiguration,
        goods_market_participants: dict[str, list[Agent]],
        origin_trade_proportions: np.ndarray,
        destin_trade_proportions: np.ndarray,
        initial_supply_chain: Optional[SupplyChain] = None,
        row_index: int = -1,
    ) -> "GoodsMarket":
        """Create a goods market instance from configuration data.

        Initializes a new goods market with the specified industry structure,
        participants, and trade relationships. Sets up the market functions,
        time series tracking, and initial states.

        Args:
            n_industries (int): Number of industries
            configuration (GoodsMarketConfiguration): Market configuration
            goods_market_participants (dict[str, list[Agent]]): Market participants
            origin_trade_proportions (np.ndarray): Historical or target proportions of trade
                flows from origin countries. Shape: (n_countries * n_countries * n_industries).
                These proportions represent the share of each industry's output from each
                country that historically flows to each destination country. Used as a
                baseline for trade pattern determination. For example, if country A
                historically sources 30% of its steel from country B, this would be
                reflected in these proportions.
            destin_trade_proportions (np.ndarray): Historical or target proportions of trade
                flows to destination countries. Shape: (n_countries * n_countries * n_industries).
                These proportions represent the share of each industry's imports in each
                destination country that historically comes from each origin country.
                Complements origin_trade_proportions by providing the destination country's
                perspective on trade patterns. For example, if 40% of country A's steel
                imports come from country B, this would be captured here.
            initial_supply_chain (Optional[SupplyChain]): Initial supply chain state. If None,
                initializes an empty supply chain for each industry.
            row_index (int): Rest of World index. Defaults to -1.

        Returns:
            GoodsMarket: Initialized goods market instance

        Note:
            The trade proportions are reshaped into (n_countries, n_countries, n_industries)
            arrays where:
            - First dimension: Origin country
            - Second dimension: Destination country
            - Third dimension: Industry
            These proportions influence but do not strictly determine actual trade flows,
            which also depend on market clearing conditions, priorities, and other factors.
        """
        # Get corresponding functions
        functions = functions_from_model(configuration.functions, loc="macromodel.markets.goods_market")
        n_countries = int(np.sqrt(len(origin_trade_proportions) / n_industries))

        states = {
            "origin_trade_proportions": origin_trade_proportions.reshape((n_countries, n_countries, n_industries)),
            "destin_trade_proportions": destin_trade_proportions.reshape((n_countries, n_countries, n_industries)),
            "previous_supply_chain": None,
        }

        # Create the corresponding time series object
        ts = create_goods_market_timeseries(n_industries)

        if initial_supply_chain is None:
            states["current_supply_chain"] = {g: {} for g in range(n_industries)}
        else:
            states["current_supply_chain"] = initial_supply_chain

        buyer_priorities = {}
        seller_priorities = {}

        for c in goods_market_participants.keys():
            seller_priorities[c] = np.array(
                [
                    goods_market_participants[c][i].transactor_settings["Buyer Priority"]
                    for i in range(len(goods_market_participants[c]))
                ]
            )
            buyer_priorities[c] = np.array(
                [
                    goods_market_participants[c][i].transactor_settings["Seller Priority"]
                    for i in range(len(goods_market_participants[c]))
                ]
            )

        return cls(
            n_industries,
            functions,
            ts,
            goods_market_participants,
            states=states,
            buyer_priorities=buyer_priorities,
            seller_priorities=seller_priorities,
            row_index=row_index,
        )

    def reset(self, configuration: GoodsMarketConfiguration):
        """Reset the market to its initial state.

        Resets time series and states to initial conditions and updates
        market functions with new configuration.

        Args:
            configuration (GoodsMarketConfiguration): New market configuration
        """
        self.ts.reset()
        self.states = deepcopy(self.initial_states)
        update_functions(model=configuration.functions, loc="macromodel.agents.goods_market", functions=self.functions)

    def prepare(self, collect_sd: bool = True) -> None:
        """Prepare the market for clearing.

        Prepares market participants and optionally collects supply/demand data.
        Initializes supply chain tracking for the current period.

        Args:
            collect_sd (bool): Whether to collect supply/demand data. Defaults to True.
        """
        # Prepare agents
        self.functions["clearing"].prepare(goods_market_participants=self.goods_market_participants)
        if collect_sd:
            total_supply, total_demand = self.functions["clearing"].collect_all_supply_and_demand(
                goods_market_participants=self.goods_market_participants,
                n_industries=self.n_industries,
            )
            self.ts.total_industry_supply.append(total_supply)
            self.ts.total_industry_demand.append(total_demand)

        # Update the supply chain
        #
        # self.states["previous_supply_chain"] = deepcopy(  # turned off for now
        #     self.states["current_supply_chain"]
        # )
        #
        self.states["current_supply_chain"] = {g: {} for g in range(self.n_industries)}

    def clear(self) -> None:
        """Execute market clearing.

        Runs the configured market clearing mechanism to match buyers with
        sellers and execute transactions. Updates supply chain relationships
        based on the resulting trades.
        """
        self.functions["clearing"].clear(
            goods_market_participants=self.goods_market_participants,
            n_industries=self.n_industries,
            default_origin_trade_proportions=self.states["origin_trade_proportions"],
            default_destin_trade_proportions=self.states["destin_trade_proportions"],
            buyer_priorities=self.buyer_priorities,
            previous_supply_chain=self.states["previous_supply_chain"],
            current_supply_chain=self.states["current_supply_chain"],
            row_index=self.row_index,
        )

    def record(self) -> None:
        """Record market outcomes.

        Triggers recording of transaction outcomes for all market participants.
        """
        self.functions["clearing"].record(goods_market_participants=self.goods_market_participants)

    def save_to_h5(self, h5_file: pd.HDFStore) -> None:
        """Save market state to HDF5.

        Saves time series data to an HDF5 file for later analysis.

        Args:
            h5_file (pd.HDFStore): Open HDF5 file to save to
        """
        group = h5_file.create_group("GM")
        self.ts.write_to_h5("GM", group)


def format_array(arr):
    return np.array2string(arr, formatter={"float_kind": lambda x: "{:.2e}".format(x)})
