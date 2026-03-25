"""Pro-rata credit market clearing utility functions.

This module implements a proportional (pro-rata) credit market clearing mechanism where
credit supply and demand are matched based on relative shares rather than individual
matches. This approach ensures fair distribution of credit when supply and demand
are imbalanced.

Key Features:
1. Credit Supply Collection:
   - Aggregate real and nominal credit supply by sector
   - Calculate average interest rates
   - Handle lending preferences and priorities
   - Track lending capacity utilization

2. Credit Demand Collection:
   - Aggregate real and nominal credit demand by sector
   - Convert between value types (real/nominal)
   - Apply borrowing constraints and limits
   - Consider borrower priorities

3. Even Distribution:
   - Proportional allocation when supply > demand
   - Fair rationing when demand > supply
   - Priority-based credit distribution
   - Minimum credit allocation guarantees

4. Excess Demand Handling:
   - Track unfulfilled credit demand
   - Distribute excess demand proportionally
   - Update lender capacity utilization
   - Monitor credit market pressures

The pro-rata mechanism is particularly useful for:
- Ensuring fair credit allocation in supply-constrained markets
- Handling large numbers of lenders and borrowers efficiently
- Maintaining stable lending relationships
- Managing priority-based credit distribution
- Implementing policy-driven credit allocation
"""

from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

from macromodel.agents.agent import Agent
from macromodel.markets.goods_market.value_type import ValueType


# @njit(float64[:](float64[:], int64[:], int64), parallel=True, cache=True)
@njit(parallel=True, cache=True)
def get_split_sum(val: np.ndarray, groups: np.ndarray, n_industries: int) -> np.ndarray:
    """Calculate sum of values by industry group for credit market analysis.

    This function efficiently computes the sum of credit-related values for each
    industry group using parallel processing. It's optimized with Numba for
    performance on large credit market datasets.

    The function is used to:
    1. Aggregate credit supply/demand by industry
    2. Calculate total lending capacity by sector
    3. Sum up credit exposures for risk analysis

    Args:
        val: Array of credit-related values to sum (e.g., loan amounts)
        groups: Array of industry indices for each value
        n_industries: Total number of industries in the model

    Returns:
        np.ndarray: Array of sums for each industry

    Example:
        val = [1000000, 2000000, 3000000, 4000000]  # Credit amounts
        groups = [0, 0, 1, 1]  # Industry indices
        n_industries = 2
        Returns: [3000000, 7000000]  # Total credit for industries 0 and 1
    """
    split_sum = np.zeros(n_industries)
    for _g in prange(n_industries):
        split_sum[_g] = val[groups == _g].sum()
    return split_sum


def collect_seller_info(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
    high_prio_only: bool = False,
    from_country: Optional[int] = None,
    trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    use_initial: bool = False,
) -> Tuple[
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Collect and aggregate lender information across all participants.

    This function gathers credit supply information from all lenders, considering
    priorities, lending preferences, and value types. It calculates both real
    and nominal credit supply totals and average interest rates.

    The collection process:
    1. Initialize supply tracking:
       - Set up dictionaries for real and nominal supply by sector
       - Handle initial vs remaining credit capacity
       - Apply lending proportion constraints

    2. Aggregate by lender type:
       - Process high-priority lenders first if specified
       - Convert between real and nominal values
       - Track bilateral lending relationships
       - Apply sector-specific lending limits

    3. Calculate market metrics:
       - Total credit supply by sector
       - Average interest rates
       - Lending capacity utilization
       - Market concentration measures

    Args:
        goods_market_participants: Dict mapping sector names to lists of lending agents
        n_industries: Number of industries/sectors in the model
        high_prio_only: Whether to only consider high-priority lenders
        from_country: Optional specific sector index to collect from
        trade_proportions: Optional array of lending proportion constraints
        exclude_row: Whether to exclude Rest of World sector
        use_initial: Whether to use initial rather than remaining credit capacity

    Returns:
        Tuple containing:
        - Dict mapping sectors to real credit supply arrays
        - Aggregate real credit supply array
        - Dict mapping sectors to nominal credit supply arrays
        - Aggregate nominal credit supply array
        - Average interest rate array by sector

    Example:
        For banking sector across three industries:
        1. Collect supply:
           Bank A: $10M at 5% rate
           Bank B: $5M at 6% rate
           Bank C: $3M at 4.5% rate
        2. Calculate aggregates:
           Total real supply: $18M
           Average rate: 5.2%
           Capacity utilization: 75%
    """
    init_field, rem_field = "Initial Goods", "Remaining Goods"
    if use_initial:
        rem_field = "Initial Goods"
    if trade_proportions is None:
        trade_proportions = np.ones(n_industries)
    if from_country is None:
        country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in country_names:
                country_names.remove("ROW")
    else:
        country_names = [list(goods_market_participants.keys())[from_country]]
    total_real_supply = {c: np.zeros(n_industries) for c in country_names}
    total_nominal_supply = {c: np.zeros(n_industries) for c in country_names}
    for country_name in country_names:
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_seller_states["Priority"] == 1 or not high_prio_only:
                if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                    total_real_supply[country_name] += get_split_sum(
                        np.minimum(
                            trade_proportions[transactor.transactor_seller_states["Industries"]]
                            * transactor.transactor_seller_states[init_field],
                            transactor.transactor_seller_states[rem_field],
                        ),
                        transactor.transactor_seller_states["Industries"],
                        n_industries,
                    )
                    total_nominal_supply[country_name] += get_split_sum(
                        np.minimum(
                            trade_proportions[transactor.transactor_seller_states["Industries"]]
                            * transactor.transactor_seller_states[init_field],
                            transactor.transactor_seller_states[rem_field],
                        )
                        * transactor.transactor_seller_states["Prices"],
                        transactor.transactor_seller_states["Industries"],
                        n_industries,
                    )

    # Calculate the average price
    aggr_real_supply = np.stack(list(total_real_supply.values()), axis=0).sum(axis=0)
    aggr_nominal_supply = np.stack(list(total_nominal_supply.values()), axis=0).sum(axis=0)
    price = np.divide(
        aggr_nominal_supply,
        aggr_real_supply,
        out=np.zeros(n_industries),
        where=aggr_real_supply != 0,
    )

    return (
        total_real_supply,
        aggr_real_supply,
        total_nominal_supply,
        aggr_nominal_supply,
        price,
    )


def collect_buyer_info(
    goods_market_participants: dict[str, list[Agent]],
    average_price: np.ndarray,
    n_industries: int,
    high_prio_only: bool = False,
    to_country: Optional[int] = None,
    trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    with_value_type: Optional[ValueType] = None,
) -> Tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Collect and aggregate borrower information across all participants.

    This function gathers credit demand information from all borrowers, handling
    both real and nominal value types. It converts between value types using
    average interest rates and applies borrowing constraints.

    The collection process:
    1. Initialize demand tracking:
       - Set up dictionaries for real and nominal demand by sector
       - Apply borrowing proportion constraints
       - Handle value type filtering

    2. Aggregate by borrower type:
       - Process high-priority borrowers first if specified
       - Convert between real and nominal values
       - Apply sector-specific borrowing limits
       - Track bilateral borrowing relationships

    3. Calculate market metrics:
       - Total credit demand by sector
       - Demand composition by value type
       - Borrowing capacity utilization
       - Sector concentration measures

    Args:
        goods_market_participants: Dict mapping sector names to lists of borrowing agents
        average_price: Array of average interest rates by sector
        n_industries: Number of industries/sectors in the model
        high_prio_only: Whether to only consider high-priority borrowers
        to_country: Optional specific sector index to collect from
        trade_proportions: Optional array of borrowing proportion constraints
        exclude_row: Whether to exclude Rest of World sector
        with_value_type: Optional filter for specific value type

    Returns:
        Tuple containing:
        - Dict mapping sectors to real credit demand arrays
        - Aggregate real credit demand array
        - Dict mapping sectors to nominal credit demand arrays
        - Aggregate nominal credit demand array

    Example:
        For corporate sector across two industries:
        1. Collect demand:
           Firm A:
           - Real borrowers: $8M
           - Nominal borrowers: $10M (at 5% rate = $9.52M real)
           Firm B:
           - Real borrowers: $4M
           - Nominal borrowers: $6M (at 5% rate = $5.71M real)
        2. Calculate aggregates:
           Total real demand: $27.23M
           Total nominal: $28.6M
    """
    init_field, rem_field = "Initial Goods", "Remaining Goods"
    if trade_proportions is None:
        trade_proportions = np.ones(n_industries)
    if to_country is None:
        country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in country_names:
                country_names.remove("ROW")
    else:
        country_names = [list(goods_market_participants.keys())[to_country]]
    total_real_demand = {c: np.zeros(n_industries) for c in country_names}
    total_nominal_demand = {c: np.zeros(n_industries) for c in country_names}
    for country_name in country_names:
        for transactor in goods_market_participants[country_name]:
            if with_value_type is None or transactor.transactor_buyer_states["Value Type"] == with_value_type:
                if transactor.transactor_buyer_states["Priority"] == 1 or not high_prio_only:
                    if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                        total_real_demand[country_name] += np.minimum(
                            trade_proportions * transactor.transactor_buyer_states[init_field],
                            transactor.transactor_buyer_states[rem_field],
                        ).sum(axis=0)
                        total_nominal_demand[country_name] += (
                            np.minimum(
                                trade_proportions * transactor.transactor_buyer_states[init_field],
                                transactor.transactor_buyer_states[rem_field],
                            ).sum(axis=0)
                            * average_price
                        )
                    elif transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                        total_real_demand[country_name] += np.divide(
                            np.minimum(
                                trade_proportions * transactor.transactor_buyer_states[init_field],
                                transactor.transactor_buyer_states[rem_field],
                            ).sum(axis=0),
                            average_price,
                            out=np.full(
                                transactor.transactor_buyer_states["Remaining Goods"].shape[1],
                                np.nan,
                            ),
                            where=average_price != 0,
                        )
                        total_nominal_demand[country_name] += np.minimum(
                            trade_proportions * transactor.transactor_buyer_states[init_field],
                            transactor.transactor_buyer_states[rem_field],
                        ).sum(axis=0)

    # Calculate sums
    aggr_real_demand = np.stack(list(total_real_demand.values()), axis=0).sum(axis=0)
    aggr_nominal_demand = np.stack(list(total_nominal_demand.values()), axis=0).sum(axis=0)

    return (
        total_real_demand,
        aggr_real_demand,
        total_nominal_demand,
        aggr_nominal_demand,
    )


def clear_evenly(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
    total_real_supply: dict[str, np.ndarray],
    aggr_real_supply: np.ndarray,
    average_goods_price: np.ndarray,
    total_real_demand: dict[str, np.ndarray],
    aggr_real_demand: np.ndarray,
    sell_high_prio_only: bool = False,
    buy_high_prio_only: bool = False,
    from_country: Optional[str] = None,
    to_country: Optional[str] = None,
    trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    with_buyer_value_type: Optional[ValueType] = None,
) -> None:
    """Execute pro-rata credit market clearing across all participants.

    This function implements proportional credit market clearing where supply and
    demand are matched based on relative shares. It handles both cases where
    credit supply exceeds demand and vice versa, ensuring fair distribution.

    The clearing process:
    1. When supply > demand:
       - Each lender provides credit proportional to their capacity
       - Each borrower receives their full requested amount
       - Lenders maintain excess lending capacity
       - Interest rates tend to be more favorable to borrowers

    2. When demand > supply:
       - Each lender provides their full available credit
       - Each borrower receives proportional to their demand
       - Borrowers maintain excess demand records
       - Interest rates tend to be more favorable to lenders

    3. Priority handling:
       - High-priority lenders get first choice of borrowers
       - High-priority borrowers get first access to credit
       - Minimum credit allocations are guaranteed where possible
       - Sector-specific constraints are respected

    Args:
        goods_market_participants: Dict mapping sector names to lists of agents
        n_industries: Number of industries/sectors in the model
        total_real_supply: Dict mapping sectors to real credit supply arrays
        aggr_real_supply: Aggregate real credit supply array
        average_goods_price: Array of average interest rates by sector
        total_real_demand: Dict mapping sectors to real credit demand arrays
        aggr_real_demand: Aggregate real credit demand array
        sell_high_prio_only: Whether to only process high-priority lenders
        buy_high_prio_only: Whether to only process high-priority borrowers
        from_country: Optional specific source sector
        to_country: Optional specific destination sector
        trade_proportions: Optional array of credit proportion constraints
        exclude_row: Whether to exclude Rest of World sector
        with_buyer_value_type: Optional filter for specific borrower value type

    Example:
        For banking sector:
        1. Supply > Demand case:
           Supply: $10M
           Demand: $8M
           Result:
           - Each lender provides 80% of their capacity
           - Borrowers get 100% of requested amount
           - $2M remains available for lending

        2. Demand > Supply case:
           Supply: $8M
           Demand: $10M
           Result:
           - Lenders provide 100% of capacity
           - Each borrower gets 80% of requested amount
           - $2M of excess demand recorded
    """
    if from_country is None:
        from_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in from_country_names:
                from_country_names.remove("ROW")
    else:
        from_country_names = [from_country]
    if to_country is None:
        to_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in to_country_names:
                to_country_names.remove("ROW")
    else:
        to_country_names = [to_country]
    for g in range(n_industries):
        if trade_proportions is None:
            trade_prop = 1.0
        else:
            trade_prop = trade_proportions[g]
        if aggr_real_supply[g] == 0 or aggr_real_demand[g] == 0:
            continue
        if aggr_real_supply[g] > aggr_real_demand[g]:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if not np.all(transactor.transactor_seller_states["Remaining Goods"] == 0):
                            if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                                for rec_country in total_real_demand.keys():
                                    real_amount = (
                                        transactor.transactor_seller_states["Remaining Goods"][
                                            transactor.transactor_seller_states["Industries"] == g
                                        ]
                                        / aggr_real_supply[g]
                                        * total_real_demand[rec_country][g]
                                    )
                                    transactor.transactor_seller_states["Real Amount sold to " + rec_country][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ] += real_amount
                                total_real_amount = (
                                    transactor.transactor_seller_states["Remaining Goods"][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ]
                                    / aggr_real_supply[g]
                                    * aggr_real_demand[g]
                                )
                                transactor.transactor_seller_states["Real Amount sold"][
                                    transactor.transactor_seller_states["Industries"] == g
                                ] += total_real_amount
                                transactor.transactor_seller_states["Remaining Goods"][
                                    transactor.transactor_seller_states["Industries"] == g
                                ] -= (
                                    transactor.transactor_seller_states["Remaining Goods"][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ]
                                    / aggr_real_supply[g]
                                    * aggr_real_demand[g]
                                )
            # Buyer
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    if (
                        with_buyer_value_type is None
                        or transactor.transactor_buyer_states["Value Type"] == with_buyer_value_type
                    ):
                        if transactor.transactor_buyer_states["Priority"] == 1 or not buy_high_prio_only:
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    average_goods_price[g]
                                    * trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        (
                                            average_goods_price[g]
                                            * trade_prop
                                            * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        )
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                            elif transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / average_goods_price[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        (
                                            trade_prop
                                            * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                            / average_goods_price[g]
                                        )
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
        else:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            transactor.transactor_seller_states["Real Amount sold"][
                                transactor.transactor_seller_states["Industries"] == g
                            ] += transactor.transactor_seller_states["Remaining Goods"][
                                transactor.transactor_seller_states["Industries"] == g
                            ]
                            for buy_country in total_real_demand.keys():
                                transactor.transactor_seller_states["Real Amount sold to " + buy_country][
                                    transactor.transactor_seller_states["Industries"] == g
                                ] += (
                                    transactor.transactor_seller_states["Remaining Goods"][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ]
                                    / aggr_real_demand[g]
                                    * total_real_demand[buy_country][g]
                                )

                            # Sellers are happy
                            transactor.transactor_seller_states["Remaining Goods"][
                                transactor.transactor_seller_states["Industries"] == g
                            ] = 0.0

            # Buyer
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    if (
                        with_buyer_value_type is None
                        or transactor.transactor_buyer_states["Value Type"] == with_buyer_value_type
                    ):
                        if transactor.transactor_buyer_states["Priority"] == 1 or not buy_high_prio_only:
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    average_goods_price[g]
                                    * trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        average_goods_price[g]
                                        * trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )
                            elif transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / average_goods_price[g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / average_goods_price[g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )

                            # Remaining goods
                            transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                trade_prop
                                * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                / aggr_real_demand[g]
                                * aggr_real_supply[g]
                            )


def distribute_excess_demand_evenly(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
) -> None:
    """Distribute excess credit demand proportionally across lenders.

    This function handles unfulfilled credit demand by allocating it to lenders
    based on their initial lending capacity and interest rates. This helps track
    market pressures and potential credit constraints.

    The distribution process:
    1. Calculate initial metrics:
       - Initial lending capacity by sector
       - Current interest rates
       - Utilization rates
       - Market concentration

    2. Determine total excess demand:
       - Aggregate unfulfilled credit requests
       - Break down by sector and priority
       - Consider value type conversions
       - Track bilateral relationships

    3. Allocate excess demand:
       - Distribute proportionally to lending capacity
       - Consider interest rate differentials
       - Respect regulatory constraints
       - Update lender records

    4. Update market metrics:
       - Excess demand by sector
       - Credit rationing indicators
       - Market pressure signals
       - Policy effectiveness measures

    Args:
        goods_market_participants: Dict mapping sector names to lists of agents
        n_industries: Number of industries/sectors in the model

    Example:
        For banking sector:
        1. Initial state:
           - Total credit supply: $8M
           - Total credit demand: $10M
           - Excess demand: $2M
           - Bank A capacity: $5M (62.5%)
           - Bank B capacity: $3M (37.5%)

        2. Allocation:
           - Bank A gets $1.25M of excess demand (62.5% of $2M)
           - Bank B gets $0.75M of excess demand (37.5% of $2M)

        This information helps:
        - Guide credit policy decisions
        - Identify credit constraints
        - Signal market stress
        - Plan capacity adjustments
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

    # Distribute excess demand
    for g in range(n_industries):
        if aggr_nominal_supply[g] == 0:
            continue
        for country_name in goods_market_participants.keys():
            for transactor in goods_market_participants[country_name]:
                if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                    transactor.transactor_seller_states["Real Excess Demand"][
                        transactor.transactor_seller_states["Industries"] == g
                    ] = (
                        (
                            transactor.transactor_seller_states["Initial Goods"][
                                transactor.transactor_seller_states["Industries"] == g
                            ]
                            * transactor.transactor_seller_states["Prices"][
                                transactor.transactor_seller_states["Industries"] == g
                            ]
                        )
                        / aggr_nominal_supply[g]
                        * excess_real_demand[g]
                    )
                    transactor.transactor_seller_states["Real Excess Demand"][
                        transactor.transactor_seller_states["Industries"] == g
                    ] = np.minimum(
                        transactor.transactor_seller_states["Real Excess Demand"][
                            transactor.transactor_seller_states["Industries"] == g
                        ],
                        transactor.transactor_seller_states["Remaining Excess Goods"][
                            transactor.transactor_seller_states["Industries"] == g
                        ],
                    )
