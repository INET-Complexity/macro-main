"""Pro-rata market clearing utility functions.

This module implements a proportional (pro-rata) market clearing mechanism where
supply and demand are matched based on relative shares rather than individual
matches. This approach ensures fair distribution of goods when supply and demand
are imbalanced.

Key Features:
1. Supply Collection:
   - Aggregate real and nominal supply by country
   - Calculate average prices
   - Handle trade proportions and priorities

2. Demand Collection:
   - Aggregate real and nominal demand by country
   - Convert between value types
   - Apply trade proportion constraints

3. Even Distribution:
   - Proportional allocation when supply > demand
   - Fair rationing when demand > supply
   - Priority-based distribution

4. Excess Demand Handling:
   - Track unfulfilled demand
   - Distribute excess demand proportionally
   - Update seller capacity utilization

The pro-rata mechanism is particularly useful for:
- Ensuring fair allocation in supply-constrained markets
- Handling large numbers of buyers and sellers efficiently
- Maintaining stable trade relationships
- Managing priority-based distribution
"""

from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

from macromodel.agents.agent import Agent
from macromodel.markets.goods_market.value_type import ValueType


# @njit(float64[:](float64[:], int64[:], int64), parallel=True, cache=True)
@njit(cache=True)
def get_split_sum(val: np.ndarray, groups: np.ndarray, n_industries: int) -> np.ndarray:
    """Calculate sum of values by industry group.

    This function efficiently computes the sum of values for each industry
    group using parallel processing. It's optimized with Numba for performance
    on large datasets.

    Args:
        val: Array of values to sum
        groups: Array of industry indices for each value
        n_industries: Total number of industries

    Returns:
        np.ndarray: Array of sums for each industry

    Example:
        val = [100, 200, 300, 400]
        groups = [0, 0, 1, 1]
        n_industries = 2
        Returns: [300, 700]  # Sum for industry 0 and 1
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
    """Collect and aggregate seller information across all participants.

    This function gathers supply information from all sellers, considering
    priorities, trade proportions, and value types. It calculates both real
    and nominal supply totals and average prices.

    Args:
        goods_market_participants: Dict mapping country names to lists of trading agents
        n_industries: Number of industries
        high_prio_only: Whether to only consider high-priority sellers
        from_country: Optional specific country index to collect from
        trade_proportions: Optional array of trade proportion constraints
        exclude_row: Whether to exclude Rest of World
        use_initial: Whether to use initial rather than remaining goods

    Returns:
        Tuple containing:
        - Dict mapping countries to real supply arrays
        - Aggregate real supply array
        - Dict mapping countries to nominal supply arrays
        - Aggregate nominal supply array
        - Average price array by industry

    Example:
        For steel industry across three countries:
        1. Collect supply:
           Country A: 1000 units at $100/unit
           Country B: 500 units at $120/unit
           Country C: 300 units at $90/unit
        2. Calculate aggregates:
           Total real supply: 1800 units
           Total nominal: $189,000
           Average price: $105/unit
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
    """Collect and aggregate buyer information across all participants.

    This function gathers demand information from all buyers, handling both
    real and nominal value types. It converts between value types using
    average prices and applies trade proportion constraints.

    Args:
        goods_market_participants: Dict mapping country names to lists of trading agents
        average_price: Array of average prices by industry
        n_industries: Number of industries
        high_prio_only: Whether to only consider high-priority buyers
        to_country: Optional specific country index to collect from
        trade_proportions: Optional array of trade proportion constraints
        exclude_row: Whether to exclude Rest of World
        with_value_type: Optional filter for specific value type

    Returns:
        Tuple containing:
        - Dict mapping countries to real demand arrays
        - Aggregate real demand array
        - Dict mapping countries to nominal demand arrays
        - Aggregate nominal demand array

    Example:
        For steel industry across two countries:
        1. Collect demand:
           Country A:
           - Real buyers: 800 units
           - Nominal buyers: $100,000 (at $100/unit = 1000 units)
           Country B:
           - Real buyers: 400 units
           - Nominal buyers: $60,000 (at $100/unit = 600 units)
        2. Calculate aggregates:
           Total real demand: 2800 units
           Total nominal: $280,000
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
    """Execute pro-rata market clearing across all participants.

    This function implements proportional market clearing where supply and
    demand are matched based on relative shares. It handles both cases where
    supply exceeds demand and vice versa, ensuring fair distribution.

    The clearing process:
    1. When supply > demand:
       - Each seller provides proportional to their supply
       - Each buyer receives their full demand
       - Sellers keep excess inventory

    2. When demand > supply:
       - Each seller provides their full supply
       - Each buyer receives proportional to their demand
       - Buyers maintain excess demand records

    Args:
        goods_market_participants: Dict mapping country names to lists of trading agents
        n_industries: Number of industries
        total_real_supply: Dict mapping countries to real supply arrays
        aggr_real_supply: Aggregate real supply array
        average_goods_price: Array of average prices by industry
        total_real_demand: Dict mapping countries to real demand arrays
        aggr_real_demand: Aggregate real demand array
        sell_high_prio_only: Whether to only process high-priority sellers
        buy_high_prio_only: Whether to only process high-priority buyers
        from_country: Optional specific source country
        to_country: Optional specific destination country
        trade_proportions: Optional array of trade proportion constraints
        exclude_row: Whether to exclude Rest of World
        with_buyer_value_type: Optional filter for specific buyer value type

    Example:
        For steel industry:
        1. Supply > Demand case:
           Supply: 1000 units
           Demand: 800 units
           Result:
           - Each seller provides 80% of their supply
           - Buyers get 100% of demand
           - 200 units remain in inventory

        2. Demand > Supply case:
           Supply: 800 units
           Demand: 1000 units
           Result:
           - Sellers provide 100% of supply
           - Each buyer gets 80% of demand
           - 200 units of excess demand recorded
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
    """Distribute excess demand proportionally across sellers.

    This function handles unfulfilled demand by allocating it to sellers
    based on their initial production capacity and prices. This helps track
    market pressures and potential capacity constraints.

    The process:
    1. Calculate initial supply and price metrics
    2. Determine total excess demand
    3. Allocate excess demand to sellers proportionally
    4. Update seller excess demand records

    Args:
        goods_market_participants: Dict mapping country names to lists of trading agents
        n_industries: Number of industries

    Example:
        For steel industry:
        1. Initial state:
           - Total supply: 800 units
           - Total demand: 1000 units
           - Excess demand: 200 units
           - Seller A capacity: 500 units (62.5%)
           - Seller B capacity: 300 units (37.5%)

        2. Allocation:
           - Seller A gets 125 units of excess demand (62.5% of 200)
           - Seller B gets 75 units of excess demand (37.5% of 200)

        This information helps:
        - Plan capacity adjustments
        - Set production targets
        - Identify supply bottlenecks
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
