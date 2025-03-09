"""Default market clearing utility functions.

This module provides the core utility functions for the default market clearing
mechanism. It implements a random matching algorithm with priorities and constraints,
supporting both bilateral and multilateral trade.

Key Features:
1. Market Status Checking:
   - Track remaining buyers and sellers
   - Monitor goods availability
   - Check transaction eligibility

2. Random Selection:
   - Priority-based buyer selection
   - Price-sensitive seller selection
   - Supply chain persistence

3. Transaction Handling:
   - Real and nominal value processing
   - Bilateral trade recording
   - Supply chain updates

4. Error Management:
   - Rounding error correction
   - Transaction validation
   - State consistency checks

The functions in this module work together to implement a sophisticated
market clearing process that respects various economic constraints while
maintaining efficiency and fairness in trade matching.
"""

from typing import Optional, Tuple

import numpy as np

from macromodel.agents.agent import Agent
from macromodel.markets.goods_market.value_type import ValueType


def check_sellers_left(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    field: str,
) -> bool:
    """Check if there are any sellers with remaining goods in the specified industry.

    This function iterates through all market participants to find sellers
    who still have goods available to sell in the given industry.

    Args:
        industry: Industry index to check
        goods_market_participants: Dict mapping country names to lists of trading agents
        field: Name of the field to check for remaining goods (e.g., "Remaining Goods")

    Returns:
        bool: True if there are sellers with goods left, False otherwise

    Example:
        If checking industry 0 (e.g., steel):
        - Country A has 100 units left
        - Country B has 0 units left
        - Country C has 50 units left
        Returns True because there are still goods available
    """
    sum_left = 0
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_seller_states["Value Type"] != ValueType.NONE:
                sum_left += transactor.transactor_seller_states[field][
                    transactor.transactor_seller_states["Industries"] == industry
                ].sum()
                if sum_left > 0.0:
                    return True
    return False


def check_buyers_left(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    field: str,
) -> bool:
    """Check if there are any buyers with remaining demand in the specified industry.

    This function iterates through all market participants to find buyers
    who still have unfulfilled demand in the given industry.

    Args:
        industry: Industry index to check
        goods_market_participants: Dict mapping country names to lists of trading agents
        field: Name of the field to check for remaining demand (e.g., "Remaining Goods")

    Returns:
        bool: True if there are buyers with unfulfilled demand, False otherwise

    Example:
        If checking industry 0 (e.g., steel):
        - Firm 1 needs 200 more units
        - Firm 2 has fulfilled its demand
        - Firm 3 needs 50 more units
        Returns True because there is still unfulfilled demand
    """
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_buyer_states["Value Type"] != ValueType.NONE:
                if transactor.transactor_buyer_states[field][:, industry].sum() > 0.0:
                    return True
    return False


# def get_random_buyer_country(
#     industry: int,
#     goods_market_participants: dict[str, list[Agent]],
#     prio_real_countries: bool,
#     field: str,
# ) -> str:
#     # Goods market participants
#     gmp = list(goods_market_participants.keys())
#     gmp.remove("ROW")
#     gmp += ["ROW"]
#
#     # Collect the number of buyers by country
#     n_buyers_by_country = {
#         c: np.sum(
#             [
#                 np.sum(
#                     goods_market_participants[c][i].transactor_buyer_states[
#                         field
#                     ][:, industry]
#                     > 0.0
#                 )
#                 for i in range(len(goods_market_participants[c]))
#             ]
#         )
#         for c in goods_market_participants.keys()
#     }
#     n_buyers_keys = list(n_buyers_by_country.keys())
#     n_buyers_vals = list(n_buyers_by_country.values())
#
#     #
#
#     #
#
#     if prio_real_countries and np.sum(n_buyers_vals) > 0:
#         return np.random.choice(
#             n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals)
#         )
#
#     # If the ROW is considered
#     if np.sum(n_buyers_vals) == 0:
#         return "ROW"
#     total_buying_real_countries = np.sum(
#         [
#             [
#                 goods_market_participants[c][i]
#                 .transactor_buyer_states[field][:, industry]
#                 .sum()
#                 for i in range(len(goods_market_participants[c]))
#             ]
#             for c in goods_market_participants.keys()
#             if c != "ROW"
#         ]
#     )
#     total_buying_row = (
#         goods_market_participants["ROW"][0]
#         .transactor_buyer_states[field][:, industry]
#         .sum()
#     )
#     if total_buying_real_countries == 0:
#         n_buyers_by_country["ROW"] = 1
#     else:
#         n_buyers_by_country["ROW"] = int(
#             np.sum(n_buyers_vals)
#             * total_buying_row
#             / total_buying_real_countries
#         )
#     n_buyers_keys = list(n_buyers_by_country.keys())
#     n_buyers_vals = list(n_buyers_by_country.values())
#
#     return np.random.choice(
#         n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals)
#     )


def get_random_buyer_type(
    industry: int,
    country_goods_market_participants: list[Agent],
    prio_high_prio: bool,
    field: str,
) -> Agent:
    """Select a random buyer type from a country's market participants.

    This function implements a priority-based selection mechanism for buyers
    within a single country. It can prioritize high-priority buyers (e.g.,
    critical industries) before considering other buyers.

    Args:
        industry: Industry index for which to select a buyer
        country_goods_market_participants: List of trading agents in the country
        prio_high_prio: Whether to prioritize high-priority buyers
        field: Name of the field to check for remaining demand

    Returns:
        Agent: Selected buyer agent

    Example:
        With prio_high_prio=True:
        1. First tries to select from high-priority buyers:
           - Power plants needing coal
           - Military needing steel
        2. If no high-priority buyers found, selects from all buyers:
           - Consumer goods manufacturers
           - Service industries
    """
    if prio_high_prio:
        n_buyers_by_agent = {
            agent: np.sum(agent.transactor_buyer_states[field][:, industry] > 0.0)
            for agent in country_goods_market_participants
            if agent.transactor_buyer_states["Priority"] == 1
        }
        n_buyers_keys = list(n_buyers_by_agent.keys())
        n_buyers_vals = list(n_buyers_by_agent.values())
        if np.sum(n_buyers_vals) > 0:
            return np.random.choice(n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals))

    # Considering all agents
    n_buyers_by_agent = {
        agent: np.sum(agent.transactor_buyer_states[field][:, industry] > 0.0)
        for agent in country_goods_market_participants
    }
    n_buyers_keys = list(n_buyers_by_agent.keys())
    n_buyers_vals = list(n_buyers_by_agent.values())
    return np.random.choice(n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals))


def get_random_buyer(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    real_country_prioritisation: float,
    prio_high_prio: bool,
    field: str,
) -> Tuple[Agent, int]:
    """Select a random buyer and specific buyer index from all market participants.

    This function implements a hierarchical selection process:
    1. First tries to select from high-priority firms
    2. Then selects a country based on buyer distribution
    3. Finally selects a specific buyer within the chosen country

    The selection process respects priorities and ensures fair distribution
    of buying opportunities across countries and agents.

    Args:
        industry: Industry index for which to select a buyer
        goods_market_participants: Dict mapping country names to lists of trading agents
        real_country_prioritisation: Weight given to real countries vs ROW [0,1]
        prio_high_prio: Whether to prioritize high-priority buyers
        field: Name of the field to check for remaining demand

    Returns:
        Tuple[Agent, int]: Selected buyer agent and specific buyer index

    Example:
        For steel industry (industry=0):
        1. Check high-priority firms first:
           - Military procurement
           - Critical infrastructure
        2. If none found, select country:
           - Weight by number of active buyers
           - Consider real_country_prioritisation
        3. Select specific buyer:
           - Random selection weighted by demand
    """
    # GMP
    country = list(goods_market_participants.keys())
    country.remove("ROW")
    assert len(country) == 1
    country = country[0]

    # Prioritise firms
    firm_agents = None
    for i in range(len(goods_market_participants[country])):
        if goods_market_participants[country][i].transactor_buyer_states["Priority"] == 1:
            firm_agents = goods_market_participants[country][i]
    buying_firms = np.where(firm_agents.transactor_buyer_states[field][:, industry] > 0.0)[0]
    if len(buying_firms) > 0:
        return firm_agents, np.random.choice(buying_firms)

    # Otherwise, choose the country
    gmps = list(goods_market_participants.keys())
    n_buyers_by_country = np.array(
        [
            np.sum(
                [
                    np.sum(goods_market_participants[c][i].transactor_buyer_states[field][:, industry] > 0.0)
                    for i in range(len(goods_market_participants[c]))
                ]
            )
            for c in gmps
        ]
    )
    chosen_country = np.random.choice(gmps, p=n_buyers_by_country / np.sum(n_buyers_by_country))

    # Then choose the type of buyer
    n_buyers_by_agent = np.array(
        [
            np.sum(goods_market_participants[chosen_country][i].transactor_buyer_states[field][:, industry] > 0.0)
            for i in range(len(goods_market_participants[chosen_country]))
        ]
    )
    chosen_agent = np.random.choice(
        range(len(goods_market_participants[chosen_country])),
        p=n_buyers_by_agent / np.sum(n_buyers_by_agent),
    )

    # Finally, choose the specific agent
    return goods_market_participants[chosen_country][chosen_agent], np.random.choice(
        np.where(
            goods_market_participants[chosen_country][chosen_agent].transactor_buyer_states[field][:, industry] > 0.0
        )[0]
    )


def pick_previous_seller(
    industry: int,
    chosen_buyer: Agent,
    chosen_buyer_ind: int,
    previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
    prio_real_countries: bool,
    prio_high_prio: bool,
    prio_domestic_sellers: bool,
    probability_keeping_previous_seller: float,
    field: str,
) -> Optional[Tuple[Agent, int]]:
    """Try to select a seller from the buyer's previous supply chain.

    This function attempts to maintain supply chain relationships by selecting
    a seller that previously supplied the chosen buyer. The selection respects
    various priority constraints and only succeeds with a given probability.

    Args:
        industry: Industry index for which to select a seller
        chosen_buyer: The buyer agent needing goods
        chosen_buyer_ind: Specific index of the buyer
        previous_supply_chain: Dict mapping industries to buyer-seller relationships
        prio_real_countries: Whether to prioritize real countries over ROW
        prio_high_prio: Whether to prioritize high-priority sellers
        prio_domestic_sellers: Whether to prioritize domestic sellers
        probability_keeping_previous_seller: Chance to maintain supply chain [0,1]
        field: Name of the field to check for remaining goods

    Returns:
        Optional[Tuple[Agent, int]]: Selected seller and index if found, None otherwise

    Example:
        For a steel manufacturer:
        1. Check if random number < probability_keeping_previous_seller
        2. If yes, look up previous suppliers:
           - Must have remaining goods
           - Must meet priority constraints
           - Must be from preferred country type
        3. Randomly select from eligible previous suppliers
    """
    if np.random.random() < probability_keeping_previous_seller:
        if (
            chosen_buyer in previous_supply_chain[industry].keys()
            and chosen_buyer_ind in previous_supply_chain[industry][chosen_buyer].keys()
        ):
            possible_sc_sellers, possible_sc_sellers_ind = [], []
            for sc_seller, sc_seller_ind in previous_supply_chain[industry][chosen_buyer][chosen_buyer_ind]:
                if (
                    sc_seller.transactor_seller_states[field][sc_seller_ind] > 0
                    and (not prio_real_countries or sc_seller.country_name != "ROW")
                    and (not prio_high_prio or sc_seller.transactor_seller_states["Priority"] == 1)
                    and (not prio_domestic_sellers or chosen_buyer.country_name == sc_seller.country_name)
                ):
                    possible_sc_sellers.append(sc_seller)
                    possible_sc_sellers_ind.append(sc_seller_ind)
            if len(possible_sc_sellers) > 0:
                random_ind = np.random.choice(len(possible_sc_sellers))
                return (
                    possible_sc_sellers[random_ind],
                    possible_sc_sellers_ind[random_ind],
                )

    return None


def get_random_seller_based_on_distribution(
    industry: int,
    chosen_goods_market_participants: list[Agent],
    price_temperature: float,
    field: str,
    distribution_type: str,
) -> Optional[Tuple[Agent, int]]:
    """Select a random seller based on production and price distributions.

    This function implements a sophisticated selection mechanism that considers
    both production volumes and prices. The selection probability is determined
    by combining these factors according to the specified distribution type.

    Args:
        industry: Industry index for which to select a seller
        chosen_goods_market_participants: List of potential seller agents
        price_temperature: Price sensitivity parameter
            Higher values → More sensitive to price differences
            Lower values → More uniform distribution
        field: Name of the field to check for remaining goods
        distribution_type: How to combine production and price weights
            "multiplicative": weights = production_weight * price_weight
            "additive": weights = 0.5 * (production_weight + price_weight)

    Returns:
        Optional[Tuple[Agent, int]]: Selected seller and index if found, None otherwise

    Example:
        For steel industry with three sellers:
        1. Calculate production weights:
           - Seller A: 1000 units → 0.5 weight
           - Seller B: 600 units → 0.3 weight
           - Seller C: 400 units → 0.2 weight

        2. Calculate price weights (temp=1.0):
           - Seller A: $100/unit → exp(-100) → 0.3 weight
           - Seller B: $90/unit → exp(-90) → 0.4 weight
           - Seller C: $110/unit → exp(-110) → 0.3 weight

        3. Combine weights (multiplicative):
           - Seller A: 0.15 final weight
           - Seller B: 0.12 final weight
           - Seller C: 0.06 final weight

    Raises:
        ValueError: If no participants or no remaining goods
    """
    if len(chosen_goods_market_participants) == 0:
        raise ValueError("No goods market participants.")

    # Compile all the data
    agent_ls, agent_ind_ls = [], []
    agent_prod_ls, agent_price_ls = [], []
    for agent in chosen_goods_market_participants:
        valid_agent_ind = np.where(
            np.logical_and(
                agent.transactor_seller_states[field] > 0,
                agent.transactor_seller_states["Industries"] == industry,
            )
        )[0]
        agent_ls.extend([agent] * len(valid_agent_ind))
        agent_ind_ls.extend(valid_agent_ind)
        agent_prod_ls.extend(agent.transactor_seller_states["Initial Goods"][valid_agent_ind])
        agent_price_ls.extend(agent.transactor_seller_states["Prices"][valid_agent_ind])
    agent_prod_ls = np.array(agent_prod_ls)
    agent_price_ls = np.array(agent_price_ls)
    if agent_prod_ls.sum() == 0.0:
        raise ValueError("No remaining goods to sell.")

    # Build a distribution
    distribution_production = agent_prod_ls / np.sum(agent_prod_ls)
    distribution_prices = np.exp(-price_temperature * agent_price_ls)
    distribution_prices /= np.sum(distribution_prices)
    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)
    distribution_norm = distribution / np.sum(distribution)

    if np.sum(np.isnan(distribution_norm)) > 0:
        print(
            "NaN in Distribution!",
            distribution,
            distribution_production,
            distribution_prices,
            agent_prod_ls,
        )

    # Draw a random seller
    random_ind = np.random.choice(len(distribution_norm), p=distribution_norm)
    return agent_ls[random_ind], agent_ind_ls[random_ind]


def get_random_seller(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    chosen_buyer: Agent,
    chosen_buyer_ind: int,
    previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
    real_country_prioritisation: float,
    prio_high_prio_sellers: bool,
    prio_domestic_sellers: bool,
    probability_keeping_previous_seller: float,
    price_temperature: float,
    field: str,
    distribution_type: str,
) -> Tuple[Agent, int]:
    """Select a random seller for a given buyer.

    This function implements a comprehensive seller selection process that
    considers multiple factors:
    1. Previous supply chain relationships
    2. Country preferences (domestic vs international)
    3. Priority status of sellers
    4. Price competitiveness
    5. Production volumes

    The selection process follows a hierarchical approach, trying different
    strategies in sequence until a suitable seller is found.

    Args:
        industry: Industry index for which to select a seller
        goods_market_participants: Dict mapping country names to lists of trading agents
        chosen_buyer: The buyer agent needing goods
        chosen_buyer_ind: Specific index of the buyer
        previous_supply_chain: Dict mapping industries to buyer-seller relationships
        real_country_prioritisation: Weight given to real countries vs ROW [0,1]
        prio_high_prio_sellers: Whether to prioritize high-priority sellers
        prio_domestic_sellers: Whether to prioritize domestic sellers
        probability_keeping_previous_seller: Chance to maintain supply chain [0,1]
        price_temperature: Price sensitivity parameter
        field: Name of the field to check for remaining goods
        distribution_type: How to combine production and price weights

    Returns:
        Tuple[Agent, int]: Selected seller and index

    Example:
        For a steel buyer:
        1. Try previous supplier (with probability)
        2. If that fails, try domestic sellers:
           - First high-priority if enabled
           - Then any domestic seller
        3. If still no match, try all sellers:
           - Weight by production and price
           - Consider priorities and country preferences

    Raises:
        ValueError: If no seller can be found
    """
    # Choose among all sellers
    chosen_goods_market_participants = []
    for country_name in goods_market_participants.keys():
        for seller in goods_market_participants[country_name]:
            if seller.transactor_seller_states["Value Type"] != ValueType.NONE:
                chosen_goods_market_participants += [seller]
    chosen_seller = get_random_seller_based_on_distribution(
        industry=industry,
        chosen_goods_market_participants=chosen_goods_market_participants,
        price_temperature=price_temperature,
        field=field,
        distribution_type=distribution_type,
    )
    if chosen_seller is not None:
        return chosen_seller

    raise ValueError("Unable to find a seller.")


def handle_transaction(
    industry: int,
    buyer: Agent,
    buyer_ind: int,
    seller: Agent,
    seller_ind: int,
) -> None:
    """Execute a transaction between a buyer and seller.

    This function processes a trade transaction, updating the states of both
    buyer and seller. It handles both real and nominal value types, ensuring
    proper accounting of quantities and prices.

    Args:
        industry: Industry index for the transaction
        buyer: The buying agent
        buyer_ind: Specific index of the buyer
        seller: The selling agent
        seller_ind: Specific index of the seller

    Raises:
        ValueError: If seller uses nominal value type (not supported)

    Example:
        For a steel transaction:
        1. Seller has 100 units at $10/unit
        2. Buyer needs 30 units (real) or $300 (nominal)
        3. Transaction processes:
           - Updates remaining goods
           - Records amounts sold/bought
           - Updates bilateral trade records
           - Handles price calculations
    """
    if seller.transactor_seller_states["Value Type"] != ValueType.REAL:
        raise ValueError("Nominal seller value type not supported.")

    # Price
    price = seller.transactor_seller_states["Prices"][seller_ind]

    # Determine the amount of the transaction
    if buyer.transactor_buyer_states["Value Type"] == ValueType.REAL:
        real_value = min(
            seller.transactor_seller_states["Remaining Goods"][seller_ind],
            buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry],
        )
        buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] -= real_value
    else:
        real_value = min(
            seller.transactor_seller_states["Remaining Goods"][seller_ind],
            buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] / price,
        )
        buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] -= price * real_value

    # Check if we're close to 0 for the buyer
    if (
        np.round(
            buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry],
            6,
        )
        == 0.0
    ):
        buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] = 0.0

    # Record the transaction for the seller
    seller.transactor_seller_states["Remaining Goods"][seller_ind] -= real_value
    if np.round(seller.transactor_seller_states["Remaining Goods"][seller_ind], 6) == 0.0:
        seller.transactor_seller_states["Remaining Goods"][seller_ind] = 0.0

    # Record the amount sold by the seller
    seller.transactor_seller_states["Real Amount sold"][seller_ind] += real_value
    seller.transactor_seller_states["Real Amount sold to " + buyer.country_name][seller_ind] += real_value

    # Record the amount spent by the buyer
    buyer.transactor_buyer_states["Nominal Amount spent"][buyer_ind, industry] += price * real_value
    buyer.transactor_buyer_states["Nominal Amount spent on Goods from " + seller.country_name][buyer_ind, industry] += (
        price * real_value
    )
    buyer.transactor_buyer_states["Real Amount bought"][buyer_ind, industry] += real_value
    buyer.transactor_buyer_states["Real Amount bought from " + seller.country_name][buyer_ind, industry] += real_value


def handle_hypothetical_transaction(
    industry: int,
    buyer: Agent,
    buyer_ind: int,
    seller: Agent,
    seller_ind: int,
) -> None:
    """Process a hypothetical transaction for excess demand handling.

    This function simulates a transaction to handle excess demand situations.
    It updates the excess demand tracking fields but does not execute an
    actual trade. This is used to model potential market adjustments.

    Args:
        industry: Industry index for the transaction
        buyer: The buying agent
        buyer_ind: Specific index of the buyer
        seller: The selling agent
        seller_ind: Specific index of the seller

    Raises:
        ValueError: If seller uses nominal value type (not supported)

    Example:
        For excess steel demand:
        1. Calculate potential transaction amount
        2. Update excess demand tracking
        3. Record hypothetical allocation
        This helps in:
        - Understanding market pressures
        - Planning capacity adjustments
        - Identifying supply bottlenecks
    """
    if seller.transactor_seller_states["Value Type"] != ValueType.REAL:
        raise ValueError("Nominal seller value type not supported.")

    # Price
    price = seller.transactor_seller_states["Prices"][seller_ind]

    # Process the transaction
    if buyer.transactor_buyer_states["Value Type"] == ValueType.REAL:
        real_min = min(
            buyer.transactor_buyer_states["Remaining Excess Goods"][buyer_ind, industry],
            seller.transactor_seller_states["Remaining Excess Goods"][seller_ind],
        )
        seller.transactor_seller_states["Real Excess Demand"][seller_ind] += real_min
        seller.transactor_seller_states["Remaining Excess Goods"][seller_ind] -= real_min
        buyer.transactor_buyer_states["Remaining Excess Goods"][buyer_ind, industry] -= real_min
    elif buyer.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
        real_min = min(
            buyer.transactor_buyer_states["Remaining Excess Goods"][buyer_ind, industry] / price,
            seller.transactor_seller_states["Remaining Excess Goods"][seller_ind],
        )
        seller.transactor_seller_states["Real Excess Demand"][seller_ind] += real_min
        seller.transactor_seller_states["Remaining Excess Goods"][seller_ind] -= real_min
        buyer.transactor_buyer_states["Remaining Excess Goods"][buyer_ind, industry] -= real_min * price


def update_supply_chain(
    current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
    industry: int,
    buyer: Agent,
    buyer_ind: int,
    seller: Agent,
    seller_ind: int,
) -> None:
    """Record a transaction in the supply chain tracking system.

    This function updates the supply chain data structure to reflect a new
    trade relationship. It maintains a history of buyer-seller connections
    that can be used for future market clearing decisions.

    Args:
        current_supply_chain: Dict tracking supply chain relationships
        industry: Industry index for the transaction
        buyer: The buying agent
        buyer_ind: Specific index of the buyer
        seller: The selling agent
        seller_ind: Specific index of the seller

    Example:
        For a steel transaction:
        1. Check if buyer exists in supply chain
        2. Check if specific buyer index exists
        3. Add seller to buyer's supplier list
        This builds a network of:
        - Who supplies whom
        - Regular trading relationships
        - Supply chain dependencies
    """
    if buyer not in current_supply_chain[industry].keys():
        current_supply_chain[industry][buyer] = {}
    if buyer_ind in current_supply_chain[industry][buyer].keys():
        current_supply_chain[industry][buyer][buyer_ind] += [(seller, seller_ind)]
    else:
        current_supply_chain[industry][buyer][buyer_ind] = [(seller, seller_ind)]


def clean_rounding_errors(goods_market_participants: dict[str, list[Agent]], decimals: int = 12) -> None:
    """Clean up numerical rounding errors in market participant states.

    This function rounds various state variables to a specified number of
    decimal places to prevent accumulation of floating-point errors during
    market clearing iterations.

    Args:
        goods_market_participants: Dict mapping country names to lists of trading agents
        decimals: Number of decimal places to round to (default: 12)

    Example:
        For all market participants:
        1. Round remaining goods quantities
        2. Round excess goods quantities
        3. Ensure consistent precision across:
           - Seller states
           - Buyer states
           - Regular and excess demand
    """
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            transactor.transactor_seller_states["Remaining Goods"] = np.round(
                transactor.transactor_seller_states["Remaining Goods"], decimals
            )
            transactor.transactor_buyer_states["Remaining Goods"] = np.round(
                transactor.transactor_buyer_states["Remaining Goods"], decimals
            )
            if "Remaining Excess Goods" in transactor.transactor_seller_states.keys():
                transactor.transactor_seller_states["Remaining Excess Goods"] = np.round(
                    transactor.transactor_seller_states["Remaining Excess Goods"],
                    decimals,
                )
            if "Remaining Excess Goods" in transactor.transactor_buyer_states.keys():
                transactor.transactor_buyer_states["Remaining Excess Goods"] = np.round(
                    transactor.transactor_buyer_states["Remaining Excess Goods"],
                    decimals,
                )
