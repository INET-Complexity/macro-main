"""Water bucket algorithm implementation for market clearing.

This module implements a sophisticated market clearing mechanism based on the water bucket
algorithm, which models trade flows like water flowing through a network of interconnected
buckets. The algorithm provides efficient and fair allocation of goods between buyers and
sellers while respecting various economic constraints.

Core Concepts:

1. Water Bucket Analogy:
   - Supply sources are like water sources (taps)
   - Demand sinks are like buckets to be filled
   - Trade flows are like water flowing through pipes
   - Priorities determine which buckets get filled first
   - Minimum fill rates ensure basic needs are met

2. Trade Flow Management:
   - Origin trade proportions: Control outflow from sources
   - Destination trade proportions: Control inflow to sinks
   - Price-based adjustments: Modify flow rates based on prices
   - Priority-based routing: Direct flows to critical sectors first

3. Price Sensitivity:
   - Temperature parameter controls price sensitivity
   - Lower temperatures → more price-sensitive allocation
   - Higher temperatures → more uniform allocation
   - Exponential decay model for price effects

4. Priority Systems:
   - Deterministic vs stochastic priority assignment
   - High-priority buyers get first access
   - Minimum fill rates for critical sectors
   - Domestic vs international preferences

Key Functions:
- get_trade_proportions: Calculates price-adjusted trade flows
- fill_buckets: Core water bucket allocation algorithm
- get_seller_priorities: Determines seller order (stochastic/deterministic)
- clear_water_bucket: Main market clearing implementation

Example:
Consider three countries trading steel:
1. Country A: Major producer, competitive prices
2. Country B: High domestic demand, higher prices
3. Country C: Small producer, critical industries

The algorithm will:
1. Adjust trade proportions based on price differences
2. Prioritize critical industry demands
3. Ensure minimum supply to each market
4. Optimize remaining allocation efficiently
"""

from typing import Optional, Tuple

import numpy as np
from numba import njit

from macromodel.agents.agent import Agent
from macromodel.markets.goods_market.value_type import ValueType


@njit(cache=True)
def get_trade_proportions(
    n_countries: int,
    default_origin_trade_proportions: np.ndarray,
    default_destin_trade_proportions: np.ndarray,
    average_prices_by_country: np.ndarray,
    temperature: float,
    real_country_prioritisation: float,
    row_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate price-adjusted trade proportions for market clearing.

    This function adjusts historical trade proportions based on current prices and
    preferences. It uses a temperature parameter to control price sensitivity and
    handles special treatment of Rest of World (ROW) trade.

    The adjustment process:
    1. For origin proportions:
       - Applies price sensitivity using exponential decay
       - Higher prices → Lower proportion (exp(-temperature * price))
       - Normalizes to maintain sum = 1 for each destination

    2. For destination proportions:
       - Maintains historical patterns with ROW adjustment
       - Applies real country prioritization
       - Normalizes to maintain sum = 1 for each origin

    Args:
        n_countries: Number of countries in the model
        default_origin_trade_proportions: Historical proportions of exports from each origin
            Shape: (n_countries, n_countries, n_industries)
            Example: [0.3, 0.5, 0.2] means country 0 sends 30% to country 0,
                    50% to country 1, and 20% to country 2
        default_destin_trade_proportions: Historical proportions of imports to each destination
            Shape: (n_countries, n_countries, n_industries)
            Example: [0.4, 0.4, 0.2] means country 0 receives 40% from country 0,
                    40% from country 1, and 20% from country 2
        average_prices_by_country: Average prices for each country and industry
            Shape: (n_countries + 1, n_industries), last row is ROW
        temperature: Price sensitivity parameter
            Higher values → More sensitive to price differences
            Lower values → More similar to historical proportions
        real_country_prioritisation: Weight given to real countries vs ROW [0,1]
            1.0 = Fully prioritize real countries
            0.0 = No special treatment for real countries
        row_index: Index for Rest of World in the country arrays

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted origin and destination proportions
            Same shapes as inputs but with price and priority adjustments

    Example:
        If temperature = 1.0 and country A's prices are 20% higher than country B:
        - Country A's export share might decrease by factor of exp(-0.2) ≈ 0.82
        - Country B's share would increase proportionally
        - Final shares are normalized to sum to 1.0
    """
    # Calculate origin trade proportions with price sensitivity
    origin_trade_proportions = np.zeros_like(default_origin_trade_proportions)
    for c1 in range(n_countries):
        # Apply price-based adjustment using exponential decay
        origin_trade_proportions[c1] = (
            np.exp(-temperature * average_prices_by_country[c1]) * default_origin_trade_proportions[c1]
        )

    # Normalize origin proportions for each destination
    for c2 in range(n_countries):
        origin_trade_proportions[:, c2] /= np.sum(origin_trade_proportions[:, c2], axis=0)

    # Handle destination trade proportions
    destin_trade_proportions = default_destin_trade_proportions.copy()
    # Adjust ROW trade based on real country prioritization
    destin_trade_proportions[:, row_index] = (
        1 - max(0.0, min(1.0, real_country_prioritisation))
    ) * default_destin_trade_proportions[:, row_index]

    # Normalize destination proportions for each origin
    for c1 in range(n_countries):
        destin_trade_proportions[c1] /= np.sum(destin_trade_proportions[c1], axis=0)

    return origin_trade_proportions, destin_trade_proportions


@njit(cache=True)
def invert_permutation(p: np.ndarray) -> np.ndarray:
    """Invert a permutation array.

    This utility function inverts a permutation array, which is needed to
    map back from sorted priorities to original indices.

    Args:
        p: Permutation array where p[i] gives the new position of element i

    Returns:
        np.ndarray: Inverse permutation where result[p[i]] = i

    Example:
        If p = [2,0,1] (meaning element 0 goes to position 2,
                               element 1 goes to position 0,
                               element 2 goes to position 1)
        Then result = [1,2,0] (meaning position 0 came from element 1,
                                    position 1 came from element 2,
                                    position 2 came from element 0)
    """
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


@njit(cache=True)
def fill_buckets(
    capacities: np.ndarray,
    fill_amount: float,
    priorities: np.ndarray,
    minimum_fill: float,
) -> np.ndarray:
    """Core water bucket allocation algorithm.

    This function implements the main water bucket allocation logic, distributing
    a fixed amount of supply (water) across multiple buckets (demands) according
    to their capacities, priorities, and minimum fill requirements.

    The allocation process:
    1. Handle special cases (NaN capacities, zero supply)
    2. Apply minimum fill rates to all buckets
    3. Fill remaining capacity in priority order
    4. Handle any leftover amount

    Args:
        capacities: Maximum amount each bucket can receive
            Shape: (n_buckets,)
            Example: [100, 50, 75] means first bucket can take 100 units,
                    second 50 units, third 75 units
        fill_amount: Total amount to distribute across all buckets
            Must be positive float
        priorities: Order in which to fill buckets
            Shape: (n_buckets,)
            Example: [2,0,1] means fill bucket 2 first, then 0, then 1
        minimum_fill: Fraction of capacity guaranteed to each bucket [0,1]
            Example: 0.2 means each bucket gets at least 20% of its
            capacity (if enough total supply exists)

    Returns:
        np.ndarray: Amount allocated to each bucket
            Shape: (n_buckets,)
            Sum of allocations equals min(fill_amount, sum(capacities))

    Example:
        capacities = [100, 50, 75]
        fill_amount = 150
        priorities = [2, 0, 1]
        minimum_fill = 0.2

        Process:
        1. Minimum fill: Each bucket gets 20% of capacity
           - Bucket 0: 20 units
           - Bucket 1: 10 units
           - Bucket 2: 15 units
           Total: 45 units

        2. Remaining 105 units allocated by priority:
           - Bucket 2 (first): Fill to capacity (60 more units)
           - Bucket 0 (second): Fill remaining (45 units)
           - Bucket 1 (third): Nothing left

        Final allocation: [65, 10, 75]
    """
    # Handle special case of NaN capacities
    if np.sum(capacities) == np.sum(capacities) + 1:
        return np.full_like(capacities, fill_amount / len(capacities))

    # Handle zero capacity or zero fill amount
    if np.sum(capacities) == 0 or fill_amount == 0.0:
        return np.zeros_like(capacities)

    # Sort capacities by priority
    capacities_sorted = capacities[priorities]
    filled_capacities = np.zeros_like(capacities_sorted)

    # Apply minimum fill rate if specified
    if minimum_fill > 0.0:
        filled_capacities += np.minimum(
            capacities_sorted,
            capacities_sorted / np.sum(capacities_sorted) * minimum_fill * fill_amount,
        )
    filled_ind = np.where((capacities_sorted - filled_capacities).cumsum() < fill_amount - np.sum(filled_capacities))[0]
    filled_capacities[filled_ind] = capacities_sorted[filled_ind]

    # Handle any leftover amount
    if len(filled_ind) < len(filled_capacities):
        filled_capacities[len(filled_ind)] += fill_amount - np.sum(filled_capacities)
        filled_capacities[len(filled_ind)] = min(
            filled_capacities[len(filled_ind)],
            capacities_sorted[len(filled_ind)],
        )

    # Map back to original bucket order
    return filled_capacities[invert_permutation(priorities)]


# njit no possible
def get_seller_priorities_stochastic(
    productions: np.ndarray,
    prices: np.ndarray,
    price_temperature: float,
    distribution_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate stochastic seller priorities based on production and prices.

    This function determines the order in which sellers are matched with buyers,
    using a probabilistic approach that considers both production volumes and
    prices. The stochastic nature helps prevent market concentration and
    promotes diversity in trade relationships.

    The priority calculation:
    1. Normalize production volumes to get production-based weights
    2. Calculate price-based weights using exponential decay
    3. Combine weights using specified distribution type
    4. Generate random permutation based on combined weights

    Args:
        productions: Production volumes for each seller
            Shape: (n_sellers,)
        prices: Prices charged by each seller
            Shape: (n_sellers,)
        price_temperature: Price sensitivity parameter
            Higher values → More sensitive to price differences
            Lower values → More uniform distribution
        distribution_type: How to combine production and price weights
            "multiplicative": weights = production_weight * price_weight
            "additive": weights = 0.5 * (production_weight + price_weight)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Combined distribution weights
            - Random permutation of seller indices based on weights

    Example:
        productions = [100, 50, 75]
        prices = [1.0, 0.8, 1.2]
        price_temperature = 1.0
        distribution_type = "multiplicative"

        Process:
        1. Production weights = [0.44, 0.22, 0.33]
        2. Price weights = [0.37, 0.45, 0.30]
        3. Combined weights = [0.16, 0.10, 0.10]
        4. Random permutation favoring higher weights
    """
    # Handle case of no production
    if np.sum(productions) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )

    # Calculate production-based distribution
    distribution_production = productions / np.sum(productions)

    # Calculate price-based distribution using exponential decay
    distribution_prices = np.exp(-price_temperature * prices)
    if np.sum(distribution_prices) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_prices /= np.sum(distribution_prices)

    # Combine distributions based on specified type
    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)

    # Ensure no zero probabilities and normalize
    distribution[distribution == 0.0] = 1e-20
    distribution /= np.sum(distribution)

    # Generate random permutation based on distribution
    return distribution, np.random.choice(len(distribution), len(distribution), replace=False, p=distribution)


@njit(cache=True)
def get_seller_priorities_deterministic(
    productions: np.ndarray,
    prices: np.ndarray,
    price_temperature: float,
    distribution_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate deterministic seller priorities based on production and prices.

    Similar to the stochastic version, but produces a deterministic ordering
    based on the combined weights. This ensures consistent matching patterns
    when reproducibility is desired.

    The priority calculation:
    1. Normalize production volumes to get production-based weights
    2. Calculate price-based weights using exponential decay
    3. Combine weights using specified distribution type
    4. Sort sellers by combined weights (highest to lowest)

    Args:
        productions: Production volumes for each seller
            Shape: (n_sellers,)
        prices: Prices charged by each seller
            Shape: (n_sellers,)
        price_temperature: Price sensitivity parameter
            Higher values → More sensitive to price differences
            Lower values → More uniform distribution
        distribution_type: How to combine production and price weights
            "multiplicative": weights = production_weight * price_weight
            "additive": weights = 0.5 * (production_weight + price_weight)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Combined distribution weights
            - Sorted permutation of seller indices (highest to lowest weight)

    Example:
        productions = [100, 50, 75]
        prices = [1.0, 0.8, 1.2]
        price_temperature = 1.0
        distribution_type = "multiplicative"

        Process:
        1. Production weights = [0.44, 0.22, 0.33]
        2. Price weights = [0.37, 0.45, 0.30]
        3. Combined weights = [0.16, 0.10, 0.10]
        4. Sorted order = [0, 2, 1]
    """
    # Handle case of no production
    if np.sum(productions) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )

    # Calculate production-based distribution
    distribution_production = productions / np.sum(productions)

    # Calculate price-based distribution using exponential decay
    distribution_prices = np.exp(-price_temperature * prices)
    if np.sum(distribution_prices) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_prices /= np.sum(distribution_prices)

    # Combine distributions based on specified type
    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)

    # Return distribution and sorted indices (highest to lowest)
    return distribution, np.argsort(distribution)[::-1]


@njit(cache=True)
def get_buyer_priorities(n_buyers: int) -> np.ndarray:
    """Generate random buyer priorities.

    Creates a random permutation of buyer indices, used when specific
    priority ordering is not needed.

    Args:
        n_buyers: Number of buyers to generate priorities for

    Returns:
        np.ndarray: Random permutation of indices [0, n_buyers-1]
    """
    return np.random.choice(n_buyers, n_buyers, replace=False)


@njit(cache=True)
def get_transactor_buyer_priorities(priorities: np.ndarray, prioritise: bool) -> np.ndarray:
    """Generate buyer priorities considering high/low priority status.

    This function creates a permutation of buyer indices that respects priority
    levels. When prioritization is enabled, high-priority buyers are placed
    before low-priority buyers, with random ordering within each group.

    Args:
        priorities: Binary array indicating priority status (1=high, 0=low)
            Shape: (n_buyers,)
        prioritise: Whether to respect priority levels
            If True: High priority buyers are placed first
            If False: Random ordering regardless of priority

    Returns:
        np.ndarray: Permutation of buyer indices respecting priorities

    Example:
        priorities = [1, 0, 1, 0, 1]
        prioritise = True

        Process:
        1. Split into high/low priority:
           high_prio = [0, 2, 4]
           low_prio = [1, 3]
        2. Randomly permute within each group
        3. Concatenate: high priority first, then low priority
    """
    if prioritise:
        # Split into high and low priority groups
        high_prio, low_prio = (
            np.where(priorities == 1)[0],
            np.where(priorities == 0)[0],
        )
        # Random permutation within each group, high priority first
        return np.concatenate(
            (
                np.random.choice(high_prio, len(high_prio), replace=False),
                np.random.choice(low_prio, len(low_prio), replace=False),
            )
        )
    else:
        # Random permutation ignoring priorities
        return np.random.choice(len(priorities), len(priorities), replace=False)


def clear_water_bucket(
    goods_market_participants: dict[str, list[Agent]],
    buyer_priority: dict[str, np.ndarray],
    n_industries: int,
    total_real_supply: dict[str, np.ndarray],
    aggr_real_supply: np.ndarray,
    average_goods_price: np.ndarray,
    total_real_demand: dict[str, np.ndarray],
    aggr_real_demand: np.ndarray,
    price_temperature: float,
    distribution_type: str,
    seller_minimum_fill: float,
    buyer_minimum_fill_macro: float,
    buyer_minimum_fill_micro: float,
    deterministic: bool,
    consider_buyer_priorities: bool,
    sell_high_prio_only: bool = False,
    buy_high_prio_only: bool = False,
    from_country: Optional[int] = None,
    to_country: Optional[int] = None,
    origin_trade_proportions: Optional[np.ndarray] = None,
    destin_trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    with_buyer_value_type: Optional[ValueType] = None,
) -> None:
    """Execute the water bucket market clearing algorithm.

    This is the core market clearing function that implements the water bucket algorithm.
    It models trade flows like water flowing through a network, where supply sources
    are like taps and demand sinks are like buckets to be filled. The algorithm
    ensures efficient and fair allocation while respecting various economic constraints.

    The clearing process operates in multiple stages:

    1. Country Selection:
       - Can clear specific country pairs or all countries
       - Optional exclusion of Rest of World (ROW)
       - Handles bilateral and multilateral clearing

    2. Trade Flow Management:
       - Uses origin/destination trade proportions if provided
       - Adjusts flows based on price differentials
       - Respects minimum fill rates for both buyers and sellers

    3. Priority-Based Allocation:
       - High-priority buyers/sellers can be processed first
       - Supports both macro (country) and micro (firm) level priorities
       - Can operate in deterministic or stochastic mode

    4. Market Clearing Logic:
       - If supply > demand: Sellers distribute to buyers
       - If demand > supply: Buyers compete for available supply
       - Maintains minimum fill rates for critical sectors

    Args:
        goods_market_participants: Dict mapping country names to lists of trading agents
        buyer_priority: Dict mapping country names to priority arrays for buyers
        n_industries: Number of industries in the model
        total_real_supply: Dict mapping country names to real supply arrays
            Shape per country: (n_industries,)
        aggr_real_supply: Aggregate real supply across all countries
            Shape: (n_industries,)
        average_goods_price: Average prices by industry
            Shape: (n_industries,)
        total_real_demand: Dict mapping country names to real demand arrays
            Shape per country: (n_industries,)
        aggr_real_demand: Aggregate real demand across all countries
            Shape: (n_industries,)
        price_temperature: Price sensitivity parameter
            Higher values → More sensitive to price differences
            Lower values → More uniform allocation
        distribution_type: How to combine production and price weights
            "multiplicative" or "additive"
        seller_minimum_fill: Minimum fill rate guaranteed to sellers [0,1]
        buyer_minimum_fill_macro: Minimum fill rate for macro buyers [0,1]
        buyer_minimum_fill_micro: Minimum fill rate for micro buyers [0,1]
        deterministic: Whether to use deterministic priority ordering
        consider_buyer_priorities: Whether to respect buyer priority levels
        sell_high_prio_only: Whether to only process high-priority sellers
        buy_high_prio_only: Whether to only process high-priority buyers
        from_country: Optional index of origin country for bilateral clearing
        to_country: Optional index of destination country for bilateral clearing
        origin_trade_proportions: Optional proportions for export flows
        destin_trade_proportions: Optional proportions for import flows
        exclude_row: Whether to exclude Rest of World from clearing
        with_buyer_value_type: Optional filter for buyer value types

    Example:
    Consider a three-country world trading steel:
    - Country A: Major producer (1000 units), competitive prices ($100/unit)
    - Country B: High domestic demand (800 units), higher prices ($120/unit)
    - Country C: Critical industry needs (200 units), limited production

    The algorithm will:
    1. Ensure minimum supply to critical industries in Country C
    2. Allocate remaining supply based on price competitiveness
    3. Respect historical trade patterns if specified
    4. Handle any excess demand through additional mechanisms

    Notes:
    - The algorithm is highly configurable through its parameters
    - It can operate at different levels of granularity (country/firm)
    - Supports both deterministic and stochastic matching
    - Handles special cases like ROW and priority sectors
    """
    # Determine countries to process
    if from_country is None:
        from_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in from_country_names:
                from_country_names.remove("ROW")
    else:
        from_country_names = [list(goods_market_participants.keys())[from_country]]

    if to_country is None:
        to_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in to_country_names:
                to_country_names.remove("ROW")
    else:
        to_country_names = [list(goods_market_participants.keys())[to_country]]

    # Process each industry
    for g in range(n_industries):
        # Get trade proportions for current industry
        if origin_trade_proportions is None or destin_trade_proportions is None:
            origin_trade_prop = 1.0
            destin_trade_prop = 1.0
        else:
            origin_trade_prop = origin_trade_proportions[g]
            destin_trade_prop = destin_trade_proportions[g]

        # Skip if no supply or demand
        if aggr_real_supply[g] == 0 or aggr_real_demand[g] == 0:
            continue

        # Case 1: Supply exceeds demand - sellers distribute to buyers
        if aggr_real_supply[g] > aggr_real_demand[g]:
            # Process each origin country
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    # Check seller priority if needed
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            # Find sellers in current industry
                            ind = transactor.transactor_seller_states["Industries"] == g
                            if np.any(transactor.transactor_seller_states["Remaining Goods"][ind] > 0.0):
                                # Get seller priorities (deterministic or stochastic)
                                if deterministic:
                                    _, seller_priorities = get_seller_priorities_deterministic(
                                        productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                        prices=transactor.transactor_seller_states["Prices"][ind],
                                        price_temperature=price_temperature,
                                        distribution_type=distribution_type,
                                    )
                                else:
                                    _, seller_priorities = get_seller_priorities_stochastic(
                                        productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                        prices=transactor.transactor_seller_states["Prices"][ind],
                                        price_temperature=price_temperature,
                                        distribution_type=distribution_type,
                                    )

                                # Calculate real amount to distribute
                                real_amount = fill_buckets(
                                    capacities=np.minimum(
                                        destin_trade_prop * transactor.transactor_seller_states["Initial Goods"][ind],
                                        transactor.transactor_seller_states["Remaining Goods"][ind],
                                    ),
                                    fill_amount=total_real_supply[country_name][g]
                                    / aggr_real_supply[g]
                                    * aggr_real_demand[g],
                                    priorities=seller_priorities,
                                    minimum_fill=seller_minimum_fill,
                                )
                                for rec_country in total_real_demand.keys():
                                    transactor.transactor_seller_states["Real Amount sold to " + rec_country][ind] += (
                                        real_amount * total_real_demand[rec_country][g] / aggr_real_demand[g]
                                    )
                                transactor.transactor_seller_states["Real Amount sold"][ind] += real_amount
                                transactor.transactor_seller_states["Remaining Goods"][ind] -= real_amount

            # Buyer
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    # Check buyer eligibility
                    if (
                        with_buyer_value_type is None
                        or transactor.transactor_buyer_states["Value Type"] == with_buyer_value_type
                    ):
                        if transactor.transactor_buyer_states["Priority"] == 1 or not buy_high_prio_only:
                            # Calculate real demand considering trade proportions
                            real_prop_rem = np.minimum(
                                origin_trade_prop * transactor.transactor_buyer_states["Initial Goods"][:, g],
                                transactor.transactor_buyer_states["Remaining Goods"][:, g],
                            )

                            # Convert nominal to real if needed
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                real_prop_rem /= average_goods_price[g]

                            # Update buyer's nominal spending
                            transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                average_goods_price[g] * real_prop_rem
                            )
                            # Update total amount bought
                            transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_prop_rem

                            # Process each seller country
                            for sell_country in total_real_supply.keys():
                                # Calculate and update bilateral trade amounts
                                bilateral_amount = (
                                    real_prop_rem * total_real_supply[sell_country][g] / aggr_real_supply[g]
                                )
                                transactor.transactor_buyer_states[
                                    "Nominal Amount spent on Goods from " + sell_country
                                ][:, g] += (
                                    (average_goods_price[g] * real_prop_rem)
                                    * total_real_supply[sell_country][g]
                                    / aggr_real_supply[g]
                                )
                                transactor.transactor_buyer_states["Real Amount bought from " + sell_country][:, g] += (
                                    real_prop_rem * total_real_supply[sell_country][g] / aggr_real_supply[g]
                                )
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                    average_goods_price[g] * real_prop_rem
                                )
                            else:
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= real_prop_rem
        else:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            # Find sellers in current industry
                            ind = transactor.transactor_seller_states["Industries"] == g
                            rem_min = np.minimum(
                                destin_trade_prop * transactor.transactor_seller_states["Initial Goods"][ind],
                                transactor.transactor_seller_states["Remaining Goods"][ind],
                            )
                            transactor.transactor_seller_states["Real Amount sold"][ind] += rem_min
                            for buy_country in total_real_demand.keys():
                                transactor.transactor_seller_states["Real Amount sold to " + buy_country][ind] += (
                                    rem_min / aggr_real_demand[g] * total_real_demand[buy_country][g]
                                )
                            transactor.transactor_seller_states["Remaining Goods"][ind] -= rem_min

            # Buyer
            for country_name in to_country_names:
                # Buyer prioritisation
                transactor_buyer_priorities = get_transactor_buyer_priorities(
                    priorities=buyer_priority[country_name],
                    prioritise=consider_buyer_priorities,
                )
                transactor_real_cap = np.zeros(len(goods_market_participants[country_name]))
                for i in range(len(goods_market_participants[country_name])):
                    rem_amount = np.minimum(
                        origin_trade_prop
                        * goods_market_participants[country_name][i].transactor_buyer_states["Initial Goods"][:, g],
                        goods_market_participants[country_name][i].transactor_buyer_states["Remaining Goods"][:, g],
                    ).sum()
                    if (
                        goods_market_participants[country_name][i].transactor_buyer_states["Value Type"]
                        == ValueType.REAL
                    ):
                        transactor_real_cap[i] = rem_amount
                    else:
                        transactor_real_cap[i] = rem_amount / average_goods_price[g]
                transactor_total_real_supply = fill_buckets(
                    capacities=transactor_real_cap,
                    fill_amount=total_real_demand[country_name][g] / aggr_real_demand[g] * aggr_real_supply[g],
                    priorities=transactor_buyer_priorities,
                    minimum_fill=buyer_minimum_fill_macro,
                )
                if np.sum(np.isnan(transactor_total_real_supply)) > 0:
                    # print(average_goods_price[g], transactor_total_real_supply)
                    # print(transactor_real_cap)
                    # print(total_real_demand[country_name][g] / aggr_real_demand[g] * aggr_real_supply[g])
                    # exit()
                    raise ValueError("Nan in transactor_total_real_supply")

                # Iterate over buyers
                for i, transactor in enumerate(goods_market_participants[country_name]):
                    prop_real = np.minimum(
                        origin_trade_prop * transactor.transactor_buyer_states["Initial Goods"][:, g],
                        transactor.transactor_buyer_states["Remaining Goods"][:, g],
                    )
                    if transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                        prop_real /= average_goods_price[g]
                    buyer_priorities = get_buyer_priorities(
                        n_buyers=transactor.transactor_buyer_states["Remaining Goods"].shape[0]
                    )
                    real_amount_bought = fill_buckets(
                        capacities=prop_real,
                        fill_amount=float(transactor_total_real_supply[i]),
                        priorities=buyer_priorities,
                        minimum_fill=buyer_minimum_fill_micro,
                    )
                    if np.sum(np.isnan(real_amount_bought)) > 0:
                        # print(average_goods_price[g], real_amount_bought)
                        # print(prop_real)
                        # print(float(transactor_total_real_supply[i]))
                        # print(type(transactor))
                        # exit()
                        raise ValueError("Nan in real_amount_bought")
                    for sell_country in total_real_supply.keys():
                        real_amount_bought_by_country = (
                            real_amount_bought * total_real_supply[sell_country][g] / aggr_real_supply[g]
                        )
                        transactor.transactor_buyer_states["Real Amount bought from " + sell_country][:, g] += (
                            real_amount_bought_by_country
                        )
                        transactor.transactor_buyer_states["Nominal Amount spent on Goods from " + sell_country][
                            :, g
                        ] += average_goods_price[g] * real_amount_bought_by_country
                    if np.isnan(average_goods_price[g]) or np.sum(np.isnan(real_amount_bought)) > 0:
                        # print(average_goods_price[g], real_amount_bought)
                        # exit()
                        raise ValueError("Nan in average_goods_price or real_amount_bought")
                    transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                        average_goods_price[g] * real_amount_bought
                    )
                    transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_amount_bought
                    if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                        transactor.transactor_buyer_states["Remaining Goods"][:, g] -= real_amount_bought
                    else:
                        transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                            average_goods_price[g] * real_amount_bought
                        )
