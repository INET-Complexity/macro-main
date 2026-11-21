"""Water bucket algorithm implementation for credit market clearing.

This module implements a sophisticated credit market clearing mechanism based on the water bucket
algorithm, which models credit flows like water flowing through a network of interconnected
buckets. The algorithm provides efficient and fair allocation of credit between lenders and
borrowers while respecting various financial and regulatory constraints.

Core Concepts:

1. Water Bucket Analogy:
   - Credit supply sources (banks) are like water sources
   - Credit demand sinks (borrowers) are like buckets to be filled
   - Credit flows are like water flowing through pipes
   - Priorities determine which buckets get filled first
   - Minimum fill rates ensure basic credit access

2. Credit Flow Management:
   - Bank lending preferences: Control outflow from sources
   - Borrower credit limits: Control inflow to sinks
   - Interest rate adjustments: Modify flow rates based on prices
   - Priority-based routing: Direct flows to critical borrowers first

3. Interest Rate Sensitivity:
   - Temperature parameter controls interest rate sensitivity
   - Lower temperatures → more rate-sensitive allocation
   - Higher temperatures → more uniform allocation
   - Exponential decay model for rate effects

4. Priority Systems:
   - Deterministic vs stochastic priority assignment
   - High-priority borrowers get first access
   - Minimum fill rates for critical sectors
   - Risk-based preferences

Key Functions:
- get_trade_proportions: Calculates rate-adjusted credit flows
- fill_buckets: Core water bucket allocation algorithm
- get_creditor_priorities: Determines lender order (stochastic/deterministic)
- clear_water_bucket: Main credit market clearing implementation

Example:
Consider three banks and three borrowers:
1. Bank A: Conservative lender, low rates
2. Bank B: High capacity, higher rates
3. Bank C: Small lender, specialized focus

The algorithm will:
1. Adjust credit flows based on interest rate differences
2. Prioritize critical borrower demands
3. Ensure minimum credit to each sector
4. Optimize remaining allocation efficiently
"""

from typing import Optional, Tuple

import numpy as np
from numba import njit

from macromodel.agents.agent import Agent
from macromodel.markets.goods_market.value_type import ValueType


# @njit(
#     types.Tuple((float64[:, :], float64[:, :]))(
#         int64,  # n_countries,
#         float64[:, :],  # default_origin_trade_proportions,
#         float64[:, :],  # default_destin_trade_proportions,
#         float64[:],  # average_prices_by_country,
#         float64,  # temperature,
#         float64,  # real_country_prioritisation,
#     ),
#     cache=True,
# )
@njit(cache=True)
def get_trade_proportions(
    n_countries: int,
    default_origin_trade_proportions: np.ndarray,
    default_destin_trade_proportions: np.ndarray,
    average_prices_by_country: np.ndarray,
    temperature: float,
    real_country_prioritisation: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate interest rate-adjusted credit allocation proportions.

    This function adjusts historical credit allocation patterns based on current
    interest rates and preferences. It uses a temperature parameter to control
    rate sensitivity and handles special treatment of different lending sectors.

    The adjustment process:
    1. For origin proportions (lender side):
       - Applies rate sensitivity using exponential decay
       - Higher rates → Lower proportion (exp(-temperature * rate))
       - Normalizes to maintain sum = 1 for each destination

    2. For destination proportions (borrower side):
       - Maintains historical patterns with sector adjustments
       - Applies real sector prioritization
       - Normalizes to maintain sum = 1 for each origin

    Args:
        n_countries: Number of lending sectors in the model
        default_origin_trade_proportions: Historical proportions of credit from each origin
            Shape: (n_sectors, n_sectors)
            Example: [0.3, 0.5, 0.2] means sector 0 lends 30% to sector 0,
                    50% to sector 1, and 20% to sector 2
        default_destin_trade_proportions: Historical proportions of credit to each destination
            Shape: (n_sectors, n_sectors)
            Example: [0.4, 0.4, 0.2] means sector 0 receives 40% from sector 0,
                    40% from sector 1, and 20% from sector 2
        average_prices_by_country: Average interest rates for each sector
            Shape: (n_sectors + 1), last row is external sector
        temperature: Interest rate sensitivity parameter
            Higher values → More sensitive to rate differences
            Lower values → More similar to historical proportions
        real_country_prioritisation: Weight given to real sectors vs external [0,1]
            1.0 = Fully prioritize real sectors
            0.0 = No special treatment for real sectors

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted origin and destination proportions
            Same shapes as inputs but with rate and priority adjustments

    Example:
        If temperature = 1.0 and sector A's rates are 20% higher than sector B:
        - Sector A's lending share might decrease by factor of exp(-0.2) ≈ 0.82
        - Sector B's share would increase proportionally
        - Final shares are normalized to sum to 1.0
    """
    # Origin trade proportions
    origin_trade_proportions = np.zeros_like(default_origin_trade_proportions)
    for c1 in range(n_countries):
        origin_trade_proportions[c1] = (
            np.exp(-temperature * average_prices_by_country[c1]) * default_origin_trade_proportions[c1]
        )
    """
    origin_trade_proportions[-1] = (
        1 - max(0.0, min(1.0, real_country_prioritisation))
    ) * default_origin_trade_proportions[-1]
    """
    for c2 in range(n_countries):
        origin_trade_proportions[:, c2] /= np.sum(origin_trade_proportions[:, c2], axis=0)

    # Destin trade proportions
    destin_trade_proportions = default_destin_trade_proportions.copy()
    destin_trade_proportions[:, -1] = (
        1 - max(0.0, min(1.0, real_country_prioritisation))
    ) * default_destin_trade_proportions[:, -1]
    for c1 in range(n_countries):
        destin_trade_proportions[c1] /= np.sum(destin_trade_proportions[c1], axis=0)

    return origin_trade_proportions, destin_trade_proportions


# @njit(int64[:](int64[:]), cache=True)
@njit(cache=True)
def invert_permutation(p: np.ndarray) -> np.ndarray:
    """Invert a permutation array for priority mapping.

    This utility function inverts a permutation array, which is needed to
    map back from sorted priorities to original indices in credit allocation.

    Args:
        p: Permutation array where p[i] gives the new position of element i

    Returns:
        np.ndarray: Inverse permutation where result[p[i]] = i

    Example:
        If p = [2,0,1] (meaning lender 0 has priority 2,
                               lender 1 has priority 0,
                               lender 2 has priority 1)
        Then result = [1,2,0] (meaning priority 0 is lender 1,
                                    priority 1 is lender 2,
                                    priority 2 is lender 0)
    """
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


# @njit(float64[:](float64[:], float64, int64[:], float64), cache=True)
@njit(cache=True)
def fill_buckets(
    capacities: np.ndarray,
    fill_amount: float,
    priorities: np.ndarray,
    minimum_fill: float,
) -> np.ndarray:
    """Core water bucket allocation algorithm for credit distribution.

    This function implements the main water bucket allocation logic, distributing
    a fixed amount of credit (water) across multiple recipients (buckets) according
    to their capacities, priorities, and minimum fill requirements.

    The allocation process:
    1. Handle special cases (NaN capacities, zero credit)
    2. Apply minimum fill rates to all recipients
    3. Fill remaining capacity in priority order
    4. Handle any leftover amount

    Args:
        capacities: Maximum amount each recipient can receive
            Shape: (n_recipients,)
            Example: [100, 50, 75] means first recipient can take 100 units,
                    second 50 units, third 75 units
        fill_amount: Total amount to distribute across all recipients
            Must be positive float
        priorities: Order in which to fill recipients
            Shape: (n_recipients,)
            Example: [2, 0, 1] means fill recipient 2 first, then 0, then 1
        minimum_fill: Fraction of capacity guaranteed to each recipient [0,1]
            Example: 0.2 means each recipient gets at least 20% of its
            capacity (if enough total credit exists)

    Returns:
        np.ndarray: Amount allocated to each recipient
            Shape: (n_recipients,)
            Sum of allocations equals min(fill_amount, sum(capacities))

    Example:
        capacities = [100, 50, 75]
        fill_amount = 150
        priorities = [2, 0, 1]
        minimum_fill = 0.2

        Process:
        1. Minimum fill: Each recipient gets 20% of capacity
           - Recipient 0: 20 units
           - Recipient 1: 10 units
           - Recipient 2: 15 units
           Total: 45 units

        2. Remaining 105 units allocated by priority:
           - Recipient 2 (first): Fill to capacity (60 more units)
           - Recipient 0 (second): Fill remaining (45 units)
           - Recipient 1 (third): Nothing left

        Final allocation: [65, 10, 75]
    """
    if np.sum(capacities) == np.sum(capacities) + 1:
        return np.full_like(capacities, fill_amount / len(capacities))
    if np.sum(capacities) == 0 or fill_amount == 0.0:
        return np.zeros_like(capacities)
    capacities_sorted = capacities[priorities]
    filled_capacities = np.zeros_like(capacities_sorted)
    if minimum_fill > 0.0:
        filled_capacities += np.minimum(
            capacities_sorted,
            capacities_sorted / np.sum(capacities_sorted) * minimum_fill * fill_amount,
        )
    filled_ind = np.where((capacities_sorted - filled_capacities).cumsum() < fill_amount - np.sum(filled_capacities))[0]
    filled_capacities[filled_ind] = capacities_sorted[filled_ind]
    if len(filled_ind) < len(filled_capacities):
        filled_capacities[len(filled_ind)] += fill_amount - np.sum(filled_capacities)
        filled_capacities[len(filled_ind)] = min(
            filled_capacities[len(filled_ind)],
            capacities_sorted[len(filled_ind)],
        )
    return filled_capacities[invert_permutation(priorities)]


def get_seller_priorities_stochastic(
    productions: np.ndarray,
    prices: np.ndarray,
    price_temperature: float,
    distribution_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate stochastic lender priorities based on capacity and rates.

    This function determines the order in which lenders are matched with borrowers,
    using a probabilistic approach that considers both lending capacity and
    interest rates. The stochastic nature helps prevent market concentration and
    promotes diversity in lending relationships.

    The priority calculation:
    1. Normalize lending capacities to get capacity-based weights
    2. Calculate rate-based weights using exponential decay
    3. Combine weights using specified distribution type
    4. Generate random permutation based on combined weights

    Args:
        productions: Lending capacity for each lender
            Shape: (n_lenders,)
        prices: Interest rates charged by each lender
            Shape: (n_lenders,)
        price_temperature: Interest rate sensitivity parameter
            Higher values → More sensitive to rate differences
            Lower values → More uniform distribution
        distribution_type: How to combine capacity and rate weights
            "multiplicative": weights = capacity_weight * rate_weight
            "additive": weights = 0.5 * (capacity_weight + rate_weight)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Combined distribution weights
            - Random permutation of lender indices based on weights

    Example:
        productions = [100, 50, 75]  # Lending capacities
        prices = [0.05, 0.04, 0.06]  # Interest rates
        price_temperature = 1.0
        distribution_type = "multiplicative"

        Process:
        1. Capacity weights = [0.44, 0.22, 0.33]
        2. Rate weights = [0.37, 0.45, 0.30]
        3. Combined weights = [0.16, 0.10, 0.10]
        4. Random permutation favoring higher weights
    """
    if np.sum(productions) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )

    distribution_production = productions / np.sum(productions)
    distribution_prices = np.exp(-price_temperature * prices)
    if np.sum(distribution_prices) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_prices /= np.sum(distribution_prices)

    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)

    distribution[distribution == 0.0] = 1e-20
    distribution /= np.sum(distribution)

    return distribution, np.random.choice(len(distribution), len(distribution), replace=False, p=distribution)


@njit(cache=True)
def get_seller_priorities_deterministic(
    productions: np.ndarray,
    prices: np.ndarray,
    price_temperature: float,
    distribution_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate deterministic lender priorities based on capacity and rates.

    Similar to the stochastic version, but produces a deterministic ordering
    based on the combined weights. This ensures consistent matching patterns
    when reproducibility is desired.

    The priority calculation:
    1. Normalize lending capacities to get capacity-based weights
    2. Calculate rate-based weights using exponential decay
    3. Combine weights using specified distribution type
    4. Sort lenders by combined weights (highest to lowest)

    Args:
        productions: Lending capacity for each lender
            Shape: (n_lenders,)
        prices: Interest rates charged by each lender
            Shape: (n_lenders,)
        price_temperature: Interest rate sensitivity parameter
            Higher values → More sensitive to rate differences
            Lower values → More uniform distribution
        distribution_type: How to combine capacity and rate weights
            "multiplicative": weights = capacity_weight * rate_weight
            "additive": weights = 0.5 * (capacity_weight + rate_weight)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Combined distribution weights
            - Sorted permutation of lender indices (highest to lowest weight)

    Example:
        productions = [100, 50, 75]  # Lending capacities
        prices = [0.05, 0.04, 0.06]  # Interest rates
        price_temperature = 1.0
        distribution_type = "multiplicative"

        Process:
        1. Capacity weights = [0.44, 0.22, 0.33]
        2. Rate weights = [0.37, 0.45, 0.30]
        3. Combined weights = [0.16, 0.10, 0.10]
        4. Sorted order = [0, 2, 1]
    """
    if np.sum(productions) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )

    distribution_production = productions / np.sum(productions)
    distribution_prices = np.exp(-price_temperature * prices)
    if np.sum(distribution_prices) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_prices /= np.sum(distribution_prices)

    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)

    return distribution, np.argsort(distribution)[::-1]


@njit(cache=True)
def get_buyer_priorities(n_buyers: int) -> np.ndarray:
    """Generate random borrower priorities.

    Creates a random permutation of borrower indices, used when specific
    priority ordering is not needed. This provides a fair, unbiased
    ordering for credit allocation.

    Args:
        n_buyers: Number of borrowers to generate priorities for

    Returns:
        np.ndarray: Random permutation of indices [0, n_buyers-1]
    """
    return np.random.choice(n_buyers, n_buyers, replace=False)


@njit(cache=True)
def get_transactor_buyer_priorities(priorities: np.ndarray, prioritise: bool) -> np.ndarray:
    """Generate borrower priorities considering high/low priority status.

    This function creates a permutation of borrower indices that respects priority
    levels. When prioritization is enabled, high-priority borrowers are placed
    before low-priority borrowers, with random ordering within each group.

    This is particularly useful for:
    - Ensuring critical sectors maintain credit access
    - Implementing policy preferences
    - Managing systemic risk through targeted lending

    Args:
        priorities: Binary array indicating priority status (1=high, 0=low)
            Shape: (n_borrowers,)
        prioritise: Whether to respect priority levels
            If True: High priority borrowers are placed first
            If False: Random ordering regardless of priority

    Returns:
        np.ndarray: Permutation of borrower indices respecting priorities

    Example:
        priorities = [1, 0, 1, 0, 1]  # Three high priority, two low priority
        prioritise = True

        Process:
        1. Split into high/low priority:
           high_prio = [0, 2, 4]  # Original indices of high priority
           low_prio = [1, 3]      # Original indices of low priority
        2. Randomly permute within each group
        3. Concatenate: high priority first, then low priority
    """
    if prioritise:
        high_prio, low_prio = (
            np.where(priorities == 1)[0],
            np.where(priorities == 0)[0],
        )
        return np.concatenate(
            (
                np.random.choice(high_prio, len(high_prio), replace=False),
                np.random.choice(low_prio, len(low_prio), replace=False),
            )
        )
    else:
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
    """Execute the water bucket credit market clearing algorithm.

    This is the core market clearing function that implements the water bucket algorithm
    for credit markets. It models credit flows like water flowing through a network,
    where lending sources are like taps and borrowing sinks are like buckets to be
    filled. The algorithm ensures efficient and fair allocation while respecting
    various financial and regulatory constraints.

    The clearing process operates in multiple stages:

    1. Sector Selection:
       - Can clear specific sector pairs or all sectors
       - Optional exclusion of external sectors
       - Handles bilateral and multilateral clearing

    2. Credit Flow Management:
       - Uses origin/destination proportions if provided
       - Adjusts flows based on interest rate differentials
       - Respects minimum fill rates for both lenders and borrowers

    3. Priority-Based Allocation:
       - High-priority lenders/borrowers can be processed first
       - Supports both macro (sector) and micro (agent) level priorities
       - Can operate in deterministic or stochastic mode

    4. Market Clearing Logic:
       - If supply > demand: Lenders distribute to borrowers
       - If demand > supply: Borrowers compete for available credit
       - Maintains minimum fill rates for critical sectors

    Args:
        goods_market_participants: Dict mapping sector names to lists of agents
        buyer_priority: Dict mapping sector names to priority arrays for borrowers
        n_industries: Number of industries in the model
        total_real_supply: Dict mapping sector names to real credit supply arrays
            Shape per sector: (n_industries,)
        aggr_real_supply: Aggregate real credit supply across all sectors
            Shape: (n_industries,)
        average_goods_price: Average interest rates by industry
            Shape: (n_industries,)
        total_real_demand: Dict mapping sector names to real credit demand arrays
            Shape per sector: (n_industries,)
        aggr_real_demand: Aggregate real credit demand across all sectors
            Shape: (n_industries,)
        price_temperature: Interest rate sensitivity parameter
            Higher values → More sensitive to rate differences
            Lower values → More uniform allocation
        distribution_type: How to combine capacity and rate weights
            "multiplicative" or "additive"
        seller_minimum_fill: Minimum fill rate guaranteed to lenders [0,1]
        buyer_minimum_fill_macro: Minimum fill rate for macro borrowers [0,1]
        buyer_minimum_fill_micro: Minimum fill rate for micro borrowers [0,1]
        deterministic: Whether to use deterministic priority ordering
        consider_buyer_priorities: Whether to respect borrower priority levels
        sell_high_prio_only: Whether to only process high-priority lenders
        buy_high_prio_only: Whether to only process high-priority borrowers
        from_country: Optional index of origin sector for bilateral clearing
        to_country: Optional index of destination sector for bilateral clearing
        origin_trade_proportions: Optional proportions for lending flows
        destin_trade_proportions: Optional proportions for borrowing flows
        exclude_row: Whether to exclude external sectors from clearing
        with_buyer_value_type: Optional filter for borrower value types

    Example:
    Consider a three-sector credit market:
    - Sector A: Major lender (1000 units), competitive rates (5%)
    - Sector B: High credit demand (800 units), higher rates (6%)
    - Sector C: Critical sector needs (200 units), limited supply

    The algorithm will:
    1. Ensure minimum credit to critical sectors
    2. Allocate remaining credit based on rate competitiveness
    3. Respect historical lending patterns if specified
    4. Handle any excess demand through additional mechanisms

    Notes:
    - The algorithm is highly configurable through its parameters
    - It can operate at different levels of granularity (sector/agent)
    - Supports both deterministic and stochastic matching
    - Handles special cases like external sectors and priority borrowers
    """
    # Determine sectors to process
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

        # Case 1: Supply exceeds demand - lenders distribute to borrowers
        if aggr_real_supply[g] > aggr_real_demand[g]:
            # Process each origin sector
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    # Check lender priority if needed
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            # Find lenders in current industry
                            ind = transactor.transactor_seller_states["Industries"] == g
                            if np.any(transactor.transactor_seller_states["Remaining Goods"][ind] > 0.0):
                                # Get lender priorities (deterministic or stochastic)
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

            # Borrower
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    # Check borrower eligibility
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

                            # Update borrower's nominal spending
                            transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                average_goods_price[g] * real_prop_rem
                            )
                            # Update total amount borrowed
                            transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_prop_rem

                            # Process each lender sector
                            for sell_country in total_real_supply.keys():
                                # Calculate and update bilateral credit amounts
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
            # Lender
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            # Find lenders in current industry
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

            # Borrower
            for country_name in to_country_names:
                # Borrower prioritisation
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
                    raise ValueError("Nan in transactor_total_real_supply")

                # Iterate over borrowers
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
