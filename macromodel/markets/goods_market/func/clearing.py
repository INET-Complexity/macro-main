"""Market clearing implementations for the goods market.

This module provides different strategies for clearing the goods market, matching
buyers with sellers while respecting various constraints and priorities. It implements
multiple clearing mechanisms:

1. Default Clearing:
   - Random matching with priority weights
   - Supply chain persistence
   - Excess demand handling

2. Pro-rata Clearing:
   - Proportional allocation based on demands
   - Aggregate supply/demand balancing

3. Water Bucket Clearing:
   - Network flow based allocation
   - Trade proportion preservation
   - Priority-based routing

Each clearing mechanism can be configured to handle:
- Domestic vs international trade preferences
- Supply chain relationships
- Price and quantity adjustments
- Priority-based allocation
- Minimum fill rates
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from macromodel.agents.agent import Agent
from macromodel.markets.goods_market.func.lib_default import (
    check_buyers_left,
    check_sellers_left,
    clean_rounding_errors,
    get_random_buyer,
    get_random_seller,
    handle_hypothetical_transaction,
    handle_transaction,
    update_supply_chain,
)
from macromodel.markets.goods_market.func.lib_pro_rata import (
    collect_buyer_info,
    collect_seller_info,
)
from macromodel.markets.goods_market.func.lib_water_bucket import (
    clear_water_bucket,
    fill_buckets,
    get_buyer_priorities,
    get_seller_priorities_deterministic,
    get_seller_priorities_stochastic,
    get_trade_proportions,
)
from macromodel.markets.goods_market.value_type import ValueType


class GoodsMarketClearer(ABC):
    """Abstract base class for goods market clearing mechanisms.

    This class defines the interface and common functionality for all market
    clearing implementations. It handles the matching of buyers and sellers
    while respecting various economic constraints and preferences.

    The clearing process follows these general steps:
    1. Preparation of market participants
    2. Collection of supply and demand information
    3. Execution of clearing algorithm
    4. Recording of transactions

    Attributes:
        real_country_prioritisation (float): Weight given to real countries vs ROW [0,1]
        prio_high_prio_buyers (bool): Whether to prioritize high-priority buyers
        prio_high_prio_sellers (bool): Whether to prioritize high-priority sellers
        prio_domestic_sellers (bool): Whether to prioritize domestic sellers
        probability_keeping_previous_seller (float): Chance to maintain supply chains
        price_temperature (float): Price sensitivity parameter
        trade_temperature (float): Trade flow sensitivity parameter
        seller_selection_distribution_type (str): Method for selecting sellers
        seller_minimum_fill (float): Minimum order fill rate for sellers
        buyer_minimum_fill_macro (float): Minimum fill rate for macro buyers
        buyer_minimum_fill_micro (float): Minimum fill rate for micro buyers
        deterministic (bool): Whether to use deterministic matching
        consider_trade_proportions (bool): Whether to use historical trade patterns
        consider_buyer_priorities (bool): Whether to consider buyer priorities
        additionally_available_factor (float): Extra capacity factor
        price_markup (float): Price adjustment factor
        remedy_rounding_errors (bool): Whether to fix rounding errors
        allow_additional_row_exports (bool): Whether to allow extra ROW exports
    """

    def __init__(
        self,
        real_country_prioritisation: float,
        prio_high_prio_buyers: bool,
        prio_high_prio_sellers: bool,
        prio_domestic_sellers: bool,
        probability_keeping_previous_seller: float,
        price_temperature: float,
        trade_temperature: float,
        seller_selection_distribution_type: str,
        seller_minimum_fill: float,
        buyer_minimum_fill_macro: float,
        buyer_minimum_fill_micro: float,
        deterministic: bool,
        consider_trade_proportions: bool,
        consider_buyer_priorities: bool,
        additionally_available_factor: float,
        price_markup: float,
        remedy_rounding_errors: bool = True,
        allow_additional_row_exports: bool = True,
    ):
        """Initialize market clearer with configuration parameters.

        Args:
            real_country_prioritisation (float): Weight for real countries [0,1]
            prio_high_prio_buyers (bool): Prioritize high-priority buyers
            prio_high_prio_sellers (bool): Prioritize high-priority sellers
            prio_domestic_sellers (bool): Prioritize domestic sellers
            probability_keeping_previous_seller (float): Supply chain persistence
            price_temperature (float): Price sensitivity
            trade_temperature (float): Trade sensitivity
            seller_selection_distribution_type (str): Seller selection method
            seller_minimum_fill (float): Minimum seller fill rate
            buyer_minimum_fill_macro (float): Minimum macro buyer fill
            buyer_minimum_fill_micro (float): Minimum micro buyer fill
            deterministic (bool): Use deterministic matching
            consider_trade_proportions (bool): Use trade patterns
            consider_buyer_priorities (bool): Use buyer priorities
            additionally_available_factor (float): Extra capacity
            price_markup (float): Price adjustment
            remedy_rounding_errors (bool): Fix rounding errors
            allow_additional_row_exports (bool): Allow extra ROW exports
        """
        # Ensure prioritisation is between 0 and 1
        self.real_country_prioritisation = max(0.0, min(1.0, real_country_prioritisation))
        self.real_country_prioritisation = real_country_prioritisation

        # Priority flags
        self.prio_high_prio_buyers = prio_high_prio_buyers
        self.prio_high_prio_sellers = prio_high_prio_sellers
        self.prio_domestic_sellers = prio_domestic_sellers

        # Market behavior parameters
        self.probability_keeping_previous_seller = probability_keeping_previous_seller
        self.price_temperature = price_temperature
        self.trade_temperature = trade_temperature
        self.seller_selection_distribution_type = seller_selection_distribution_type

        # Fill rate requirements
        self.seller_minimum_fill = seller_minimum_fill
        self.buyer_minimum_fill_macro = buyer_minimum_fill_macro
        self.buyer_minimum_fill_micro = buyer_minimum_fill_micro

        # Market clearing behavior flags
        self.deterministic = deterministic
        self.consider_trade_proportions = consider_trade_proportions
        self.consider_buyer_priorities = consider_buyer_priorities

        # Additional parameters
        self.additionally_available_factor = additionally_available_factor
        self.price_markup = price_markup
        self.remedy_rounding_errors = remedy_rounding_errors
        self.allow_additional_row_exports = allow_additional_row_exports

    @staticmethod
    def prepare(goods_market_participants: dict[str, list[Agent]]) -> None:
        """Prepare market participants for clearing.

        Calls prepare() on all market participants to initialize their
        trading states for the current period.

        Args:
            goods_market_participants (dict[str, list[Agent]]): Market participants by country
        """
        for country_name in goods_market_participants.keys():
            for transactor in goods_market_participants[country_name]:
                transactor.prepare()

    @staticmethod
    def collect_all_supply_and_demand(
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        buyer_high_prio_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect aggregate supply and demand data.

        Gathers total supply and demand information across all industries and
        participants, optionally filtering for high-priority buyers only.

        Args:
            goods_market_participants (dict[str, list[Agent]]): Market participants
            n_industries (int): Number of industries
            buyer_high_prio_only (bool): Whether to only include high-priority buyers

        Returns:
            tuple[np.ndarray, np.ndarray]: Aggregate nominal supply and demand arrays
        """
        # Collect seller (supply) information
        (
            total_real_supply,
            aggr_real_supply,
            total_nominal_supply,
            aggr_nominal_supply,
            average_goods_price,
        ) = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
        )

        # Collect buyer (demand) information
        (
            total_real_demand,
            aggr_real_demand,
            total_nominal_demand,
            aggr_nominal_demand,
        ) = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=average_goods_price,
            n_industries=n_industries,
            high_prio_only=buyer_high_prio_only,
        )

        return aggr_nominal_supply, aggr_nominal_demand

    @abstractmethod
    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        """Execute market clearing algorithm.

        Abstract method that must be implemented by concrete clearers to match
        buyers with sellers and execute transactions.

        Args:
            goods_market_participants (dict[str, list[Agent]]): Market participants
            n_industries (int): Number of industries
            default_origin_trade_proportions (np.ndarray): Historical origin shares
            default_destin_trade_proportions (np.ndarray): Historical destination shares
            buyer_priorities (dict[str, np.ndarray]): Buyer priority rankings
            previous_supply_chain (dict): Previous period's supply chain
            current_supply_chain (dict): Current period's supply chain
            row_index (int): Rest of World index. Defaults to -1.
        """
        pass

    @staticmethod
    def record(goods_market_participants: dict[str, list[Agent]]) -> None:
        """Record market clearing outcomes.

        Calls record() on all market participants to save their final
        trading positions for the current period.

        Args:
            goods_market_participants (dict[str, list[Agent]]): Market participants
        """
        for country_name in goods_market_participants.keys():
            for transactor in goods_market_participants[country_name]:
                transactor.record()


class NoGoodsMarketClearer(GoodsMarketClearer):
    """Null implementation of market clearing.

    This implementation does nothing during clearing, effectively creating
    a market with no transactions. Useful for testing and debugging.
    """

    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        """No-op implementation of market clearing.

        Args:
            goods_market_participants (dict[str, list[Agent]]): Unused
            n_industries (int): Unused
            default_origin_trade_proportions (np.ndarray): Unused
            default_destin_trade_proportions (np.ndarray): Unused
            buyer_priorities (dict[str, np.ndarray]): Unused
            previous_supply_chain (dict): Unused
            current_supply_chain (dict): Unused
            row_index (int): Unused
        """
        pass


class DefaultGoodsMarketClearer(GoodsMarketClearer):
    """Default implementation of market clearing.

    This implementation uses a random matching algorithm with priorities and
    constraints. For each industry, it:
    1. Matches buyers and sellers based on priorities and preferences
    2. Executes transactions and updates supply chains
    3. Handles excess demand through additional matching rounds
    4. Maintains supply chain relationships where possible
    """

    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        """Execute the default market clearing algorithm.

        Processes each industry sequentially, matching buyers with sellers
        until either supply or demand is exhausted. Then handles any
        remaining excess demand.

        Args:
            goods_market_participants (dict[str, list[Agent]]): Market participants
            n_industries (int): Number of industries
            default_origin_trade_proportions (np.ndarray): Historical origin shares
            default_destin_trade_proportions (np.ndarray): Historical destination shares
            buyer_priorities (dict[str, np.ndarray]): Buyer priority rankings
            previous_supply_chain (dict): Previous period's supply chain
            current_supply_chain (dict): Current period's supply chain
            row_index (int): Rest of World index

        Note:
            The clearing process for each industry follows these steps:
            1. Check for active buyers and sellers
            2. Match buyers and sellers considering:
               - Priorities and preferences
               - Previous supply chain relationships
               - Price and trade sensitivities
            3. Execute transactions and update supply chains
            4. Handle excess demand if any remains
        """
        # Process each industry separately
        for g in range(n_industries):
            # Check if there are any buyers or sellers left
            if (
                not check_buyers_left(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    field="Remaining Goods",
                )
            ) or (
                not check_sellers_left(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    field="Remaining Goods",
                )
            ):
                continue

            # Main clearing loop: match buyers and sellers until no more matches possible
            while True:
                # Select a random buyer based on priorities and preferences
                buyer, buyer_ind = get_random_buyer(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio=self.prio_high_prio_buyers,
                    field="Remaining Goods",
                )

                # Select a seller considering:
                # - Previous supply chain relationships
                # - Price competitiveness
                # - Domestic vs international preferences
                seller, seller_ind = get_random_seller(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    chosen_buyer=buyer,
                    chosen_buyer_ind=buyer_ind,
                    previous_supply_chain=previous_supply_chain,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio_sellers=self.prio_high_prio_sellers,
                    prio_domestic_sellers=self.prio_domestic_sellers,
                    probability_keeping_previous_seller=self.probability_keeping_previous_seller,
                    price_temperature=self.price_temperature,
                    field="Remaining Goods",
                    distribution_type=self.seller_selection_distribution_type,
                )

                # Execute the transaction between matched buyer and seller
                handle_transaction(
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

                # Record the transaction in the supply chain
                update_supply_chain(
                    current_supply_chain=current_supply_chain,
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

                # Check if we've exhausted either supply or demand
                if (
                    not check_buyers_left(
                        industry=g,
                        goods_market_participants=goods_market_participants,
                        field="Remaining Goods",
                    )
                ) or (
                    not check_sellers_left(
                        industry=g,
                        goods_market_participants=goods_market_participants,
                        field="Remaining Goods",
                    )
                ):
                    break

            # Handle excess demand after main clearing
            # Copy remaining demand to excess demand field
            for country_name in goods_market_participants.keys():
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_buyer_states["Value Type"] != ValueType.NONE:
                        transactor.transactor_buyer_states["Remaining Excess Goods"] = (
                            transactor.transactor_buyer_states["Remaining Goods"].copy()
                        )

            # Try to satisfy excess demand if possible
            while check_buyers_left(
                industry=g,
                goods_market_participants=goods_market_participants,
                field="Remaining Excess Goods",
            ):
                if not check_sellers_left(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    field="Remaining Goods",
                ):
                    break

                # Get a random buyer
                buyer, buyer_ind = get_random_buyer(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio=self.prio_high_prio_buyers,
                    field="Remaining Excess Goods",
                )

                # Get a random seller
                seller, seller_ind = get_random_seller(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    chosen_buyer=buyer,
                    chosen_buyer_ind=buyer_ind,
                    previous_supply_chain=previous_supply_chain,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio_sellers=self.prio_high_prio_sellers,
                    prio_domestic_sellers=self.prio_domestic_sellers,
                    probability_keeping_previous_seller=self.probability_keeping_previous_seller,
                    price_temperature=self.price_temperature,
                    field="Remaining Excess Goods",
                    distribution_type=self.seller_selection_distribution_type,
                )

                # Handle hypothetical transaction
                handle_hypothetical_transaction(
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

        # Clean up
        if self.remedy_rounding_errors:
            clean_rounding_errors(goods_market_participants=goods_market_participants)


class WaterBucketGoodsMarketClearer(GoodsMarketClearer):
    """Network flow-based market clearing implementation using the water bucket algorithm.

    This implementation models market clearing as a network flow problem, where supply
    and demand are distributed through the network like water flowing through buckets.
    The algorithm:

    1. Trade Flow Management:
       - Uses origin and destination trade proportions to guide flows
       - Adjusts flows based on price differentials between countries
       - Maintains historical trade relationships while allowing for adaptation

    2. Priority-Based Allocation:
       - Handles high-priority buyers (e.g., critical industries) first
       - Supports both deterministic and stochastic seller selection
       - Allows for domestic market preference

    3. Multi-Stage Clearing:
       - First clears according to trade proportions if enabled
       - Then performs general clearing for remaining supply/demand
       - Finally handles excess demand and additional ROW exports

    4. Price Sensitivity:
       - Incorporates price differences in seller selection
       - Adjusts trade flows based on price competitiveness
       - Applies markups for additional ROW exports

    Key Features:
    - Network flow optimization for efficient allocation
    - Support for minimum fill rates for both buyers and sellers
    - Flexible priority system for critical market participants
    - Sophisticated handling of international trade flows
    - Robust excess demand management
    """

    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        """Execute the water bucket market clearing algorithm.

        This method implements a sophisticated market clearing mechanism that models
        trade flows like water flowing through a network of buckets. It operates in
        multiple stages to ensure efficient and fair allocation of resources.

        Args:
            goods_market_participants: Dict mapping country names to lists of trading agents
            n_industries: Number of industries in the model
            default_origin_trade_proportions: Historical proportions of trade flows from origin
                countries, shape (n_countries, n_countries, n_industries)
            default_destin_trade_proportions: Historical proportions of trade flows to destination
                countries, shape (n_countries, n_countries, n_industries)
            buyer_priorities: Dict mapping country names to priority arrays for buyers
            previous_supply_chain: Previous period's supply chain relationships
            current_supply_chain: Current period's supply chain relationships
            row_index: Index for Rest of World in the country arrays

        Algorithm Stages:
        1. Price Collection:
           - Gathers average prices by country and industry
           - Handles missing prices using ROW averages
           - Ensures positive prices for all industries

        2. Trade Proportion-Based Clearing:
           - If enabled, first clears according to historical trade patterns
           - Adjusts proportions based on price competitiveness
           - Processes each country pair sequentially

        3. General Clearing:
           - Clears remaining supply and demand without trade constraints
           - Uses price-weighted matching for efficiency

        4. Excess Demand Handling:
           - Processes any remaining excess demand
           - Applies water bucket algorithm for fair distribution

        5. Additional ROW Exports:
           - If enabled, allows extra purchases from ROW
           - Applies price markup for additional exports
           - Prioritizes critical industries
        """
        n_countries = len(goods_market_participants.keys())

        # Calculate average prices by country and industry
        average_prices_by_country = np.zeros((n_countries + 1, n_industries))
        for ind, country_name in enumerate(goods_market_participants.keys()):
            for gmp in goods_market_participants[country_name]:
                if gmp.transactor_settings["Seller Value Type"] != ValueType.NONE:
                    average_prices_by_country[ind] = gmp.ts.current("price_offered")
                    break
        (
            _,
            _,
            _,
            _,
            average_prices_by_country[-1],
        ) = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
        )
        assert np.all(average_prices_by_country[0:-1] > 0.0)

        # Clear markets using trade proportions if enabled
        if self.consider_trade_proportions:
            # Get price-adjusted trade proportions
            origin_trade_proportions, destin_trade_proportions = get_trade_proportions(
                n_countries=n_countries,
                default_origin_trade_proportions=default_origin_trade_proportions,
                default_destin_trade_proportions=default_destin_trade_proportions,
                average_prices_by_country=average_prices_by_country,
                temperature=self.trade_temperature,
                real_country_prioritisation=self.real_country_prioritisation,
                row_index=row_index,
            )
            # Clear each country pair using trade proportions
            for c1 in range(n_countries):
                for c2 in range(n_countries):
                    self.perform_clearing(
                        goods_market_participants=goods_market_participants,
                        n_industries=n_industries,
                        average_prices_by_country=average_prices_by_country,
                        buyer_priorities=buyer_priorities,
                        start_country=c1,
                        end_country=c2,
                        origin_trade_proportions=origin_trade_proportions,
                        destin_trade_proportions=destin_trade_proportions,
                    )

        # Clear remaining supply and demand without trade proportion constraints
        self.perform_clearing(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
            average_prices_by_country=average_prices_by_country,
            buyer_priorities=buyer_priorities,
        )

        # Handle any remaining excess demand
        self.distribute_excess_demand_water_bucket(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
        )

        # Allow additional ROW exports if enabled
        if self.allow_additional_row_exports and self.additionally_available_factor > 0.0:
            self.handle_additional_row_exports(
                goods_market_participants=goods_market_participants,
                n_industries=n_industries,
            )

    def perform_clearing(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        average_prices_by_country: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        start_country: Optional[int] = None,
        end_country: Optional[int] = None,
        origin_trade_proportions: Optional[np.ndarray] = None,
        destin_trade_proportions: Optional[np.ndarray] = None,
    ) -> None:
        """Execute market clearing for a specific pair of countries or all countries.

        This method implements the core water bucket clearing algorithm, either for
        a specific country pair or for all countries. It handles both trade
        proportion-based clearing and general clearing.

        Args:
            goods_market_participants: Dict mapping country names to lists of trading agents
            n_industries: Number of industries in the model
            average_prices_by_country: Average prices by country and industry
            buyer_priorities: Dict mapping country names to priority arrays for buyers
            start_country: Optional index of origin country for bilateral clearing
            end_country: Optional index of destination country for bilateral clearing
            origin_trade_proportions: Optional trade proportions from origin countries
            destin_trade_proportions: Optional trade proportions to destination countries

        The clearing process:
        1. Extracts relevant trade proportions for the country pair if specified
        2. Collects supply information considering trade proportions
        3. Collects demand information considering trade proportions
        4. Executes the water bucket clearing algorithm with:
           - Price-based seller selection
           - Priority-based buyer allocation
           - Minimum fill rate guarantees
           - Supply chain persistence
        """
        # Extract relevant trade proportions for the country pair
        if origin_trade_proportions is None or destin_trade_proportions is None:
            current_origin_trade_proportions = None
            current_destin_trade_proportions = None
        else:
            current_origin_trade_proportions = origin_trade_proportions[start_country, end_country]
            current_destin_trade_proportions = destin_trade_proportions[start_country, end_country]

        # Collect supply information
        (
            total_real_supply,
            aggr_real_supply,
            _,
            _,
            emp_goods_prices,
        ) = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
            from_country=start_country,
            trade_proportions=current_destin_trade_proportions,
        )

        # Handle missing prices using country or ROW averages
        if start_country is None:
            emp_goods_prices[np.isnan(emp_goods_prices)] = average_prices_by_country[-1][np.isnan(emp_goods_prices)]
            emp_goods_prices[emp_goods_prices == 0.0] = average_prices_by_country[-1][emp_goods_prices == 0.0]
        else:
            emp_goods_prices[np.isnan(emp_goods_prices)] = average_prices_by_country[start_country][
                np.isnan(emp_goods_prices)
            ]
            emp_goods_prices[emp_goods_prices == 0.0] = average_prices_by_country[start_country][
                emp_goods_prices == 0.0
            ]
            emp_goods_prices[emp_goods_prices == 0.0] = average_prices_by_country[-1][emp_goods_prices == 0.0]

        # Collect demand information
        (
            total_real_demand,
            aggr_real_demand,
            _,
            _,
        ) = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=emp_goods_prices,
            n_industries=n_industries,
            to_country=end_country,
            trade_proportions=current_origin_trade_proportions,
        )

        # Execute water bucket clearing
        clear_water_bucket(
            goods_market_participants=goods_market_participants,
            buyer_priority=buyer_priorities,
            n_industries=n_industries,
            total_real_supply=total_real_supply,
            aggr_real_supply=aggr_real_supply,
            average_goods_price=emp_goods_prices,
            total_real_demand=total_real_demand,
            aggr_real_demand=aggr_real_demand,
            from_country=start_country,
            to_country=end_country,
            origin_trade_proportions=current_origin_trade_proportions,
            destin_trade_proportions=current_destin_trade_proportions,
            price_temperature=self.price_temperature,
            distribution_type=self.seller_selection_distribution_type,
            seller_minimum_fill=self.seller_minimum_fill,
            buyer_minimum_fill_macro=self.buyer_minimum_fill_macro,
            buyer_minimum_fill_micro=self.buyer_minimum_fill_micro,
            deterministic=self.deterministic,
            consider_buyer_priorities=self.consider_buyer_priorities,
        )

    def distribute_excess_demand_water_bucket(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
    ) -> None:
        """Distribute excess demand using the water bucket algorithm.

        This method handles any remaining excess demand after the main clearing
        rounds. It uses the water bucket algorithm to fairly distribute remaining
        supply to excess demand.

        Args:
            goods_market_participants: Dict mapping country names to lists of trading agents
            n_industries: Number of industries in the model

        The distribution process:
        1. Collects initial supply and excess demand information
        2. For each industry with both supply and excess demand:
           a. Calculates allocation based on supply shares
           b. Adjusts for real country prioritization
           c. Distributes using the water bucket algorithm with:
              - Price-based seller selection
              - Minimum fill rate guarantees
              - Priority-based allocation
        """
        # Collect initial values
        _, _, total_nominal_supply, aggr_nominal_supply, average_price = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
            use_initial=True,
        )
        _, excess_real_demand, _, _ = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=average_price,
            n_industries=n_industries,
        )

        # Distribute excess demand by industry
        for g in range(n_industries):
            if aggr_nominal_supply[g] == 0.0 or excess_real_demand[g] == 0.0:
                continue

            # Calculate allocation based on supply shares
            current_alloc = np.array(
                [
                    excess_real_demand[g] * total_nominal_supply[country_name][g] / aggr_nominal_supply[g]
                    for country_name in goods_market_participants.keys()
                ]
            )
            # Adjust ROW allocation based on real country prioritization
            current_alloc[-1] *= 1 - max(0.0, min(1.0, self.real_country_prioritisation))
            if current_alloc.sum() == 0.0:
                continue
            current_alloc *= excess_real_demand[g] / current_alloc.sum()

            # Distribute to each country
            for country_ind, country_name in enumerate(goods_market_participants.keys()):
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                        ind = transactor.transactor_seller_states["Industries"] == g
                        if self.deterministic:
                            _, seller_priorities = get_seller_priorities_deterministic(
                                productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                prices=transactor.transactor_seller_states["Prices"][ind],
                                price_temperature=self.price_temperature,
                                distribution_type=self.seller_selection_distribution_type,
                            )
                        else:
                            _, seller_priorities = get_seller_priorities_stochastic(
                                productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                prices=transactor.transactor_seller_states["Prices"][ind],
                                price_temperature=self.price_temperature,
                                distribution_type=self.seller_selection_distribution_type,
                            )
                        transactor.transactor_seller_states["Real Excess Demand"][ind] = fill_buckets(
                            capacities=transactor.transactor_seller_states["Remaining Excess Goods"][ind],
                            fill_amount=current_alloc[country_ind],
                            priorities=seller_priorities,
                            minimum_fill=self.seller_minimum_fill,
                        )

    def handle_additional_row_exports(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
    ) -> None:
        """Handle additional exports from Rest of World.

        This method allows for additional purchases from ROW beyond the normal
        clearing process, typically used to handle supply shortages in critical
        industries.

        Args:
            goods_market_participants: Dict mapping country names to lists of trading agents
            n_industries: Number of industries in the model

        The process:
        1. Collects initial ROW export capacity and high-priority demand
        2. Applies price markup to ROW exports
        3. Distributes additional exports:
           a. Proportionally to country demand
           b. Only to high-priority buyers
           c. With minimum fill rate guarantees
        4. Updates both buyer and seller states
        """
        # Collect initial ROW exports and high-priority demand
        _, aggr_real_supply, _, _, average_price = collect_seller_info(
            goods_market_participants={"ROW": goods_market_participants["ROW"]},
            n_industries=n_industries,
            use_initial=True,
        )
        # Apply price markup
        average_price *= 1 + self.price_markup
        additional_real_demand_by_country, additional_real_demand, _, _ = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=average_price,
            n_industries=n_industries,
            high_prio_only=True,
            exclude_row=True,
        )

        # Distribute additional exports by industry
        for g in range(n_industries):
            if aggr_real_supply[g] == 0.0 or additional_real_demand[g] == 0.0:
                continue

            # Process each country's high-priority buyers
            for country_name in goods_market_participants.keys():
                if country_name == "ROW":
                    continue

                # Calculate country's supply allocation
                country_supply = (
                    self.additionally_available_factor
                    * aggr_real_supply.sum()
                    * additional_real_demand_by_country[country_name].sum()
                    / additional_real_demand.sum()
                    * additional_real_demand_by_country[country_name][g]
                    / additional_real_demand_by_country[country_name].sum()
                )
                if country_supply == 0.0:
                    continue

                # Distribute to high-priority buyers
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_buyer_states["Priority"] == 1:
                        # Get buyer priorities
                        buyer_priorities = get_buyer_priorities(
                            n_buyers=transactor.transactor_buyer_states["Remaining Goods"].shape[0]
                        )
                        # Allocate additional supply
                        real_amount_bought = fill_buckets(
                            capacities=transactor.transactor_buyer_states["Remaining Goods"][:, g],
                            fill_amount=country_supply,
                            priorities=buyer_priorities,
                            minimum_fill=self.buyer_minimum_fill_micro,
                        )
                        if np.sum(real_amount_bought) == 0.0:
                            continue

                        # Update buyer states
                        transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_amount_bought
                        transactor.transactor_buyer_states["Real Amount bought from ROW"][:, g] += real_amount_bought
                        transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                            average_price[g] * real_amount_bought
                        )
                        transactor.transactor_buyer_states["Nominal Amount spent on Goods from ROW"][:, g] += (
                            average_price[g] * real_amount_bought
                        )
                        transactor.transactor_buyer_states["Remaining Goods"][:, g] -= real_amount_bought

                        # Update ROW seller states
                        ind = goods_market_participants["ROW"][0].transactor_seller_states["Industries"] == g
                        goods_market_participants["ROW"][0].transactor_seller_states["Real Amount sold"][
                            ind
                        ] += real_amount_bought.sum() / np.sum(ind)
                        goods_market_participants["ROW"][0].transactor_seller_states[
                            "Real Amount sold to " + country_name
                        ][ind] += real_amount_bought.sum() / np.sum(ind)
                        goods_market_participants["ROW"][0].transactor_seller_states["Remaining Goods"][
                            ind
                        ] -= real_amount_bought.sum() / np.sum(ind)
