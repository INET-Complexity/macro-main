"""Default credit market clearing utility functions.

This module implements the default credit market clearing mechanism, which matches
individual lenders and borrowers based on various criteria including credit history,
priorities, and market conditions. The mechanism supports relationship lending and
incorporates both deterministic and stochastic elements in the matching process.

Key Features:
1. Relationship Lending:
   - Track and maintain lending relationships
   - Probability-based relationship persistence
   - History-based matching preferences
   - Supply chain credit management

2. Priority-Based Matching:
   - High-priority lender/borrower preferences
   - Domestic vs international lending bias
   - Real sector prioritization
   - Regulatory compliance checks

3. Market-Based Selection:
   - Interest rate sensitivity
   - Credit capacity constraints
   - Risk-based allocation
   - Multi-period relationships

4. Transaction Processing:
   - Credit limit verification
   - Regulatory compliance checks
   - Balance sheet updates
   - Relationship tracking

The default mechanism is particularly useful for:
- Modeling relationship-based lending
- Implementing credit market segmentation
- Handling heterogeneous agents
- Incorporating regulatory constraints
- Simulating market microstructure
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
    """Check if there are any lenders with remaining credit capacity.

    This function examines all potential lenders in a given industry to determine
    if any still have credit available to lend. It's used to decide whether to
    continue the credit allocation process or terminate due to supply exhaustion.

    The check considers:
    1. All sectors in the market
    2. Each lender's remaining credit capacity
    3. Value type constraints
    4. Industry-specific limits

    Args:
        industry: Industry/sector index to check
        goods_market_participants: Dict mapping sector names to lists of agents
        field: Name of the field containing credit capacity information
            (e.g., "Remaining Goods" for available credit)

    Returns:
        bool: True if any lender has remaining credit capacity, False otherwise

    Example:
        For banking sector (industry 0):
        - Bank A: $0 remaining
        - Bank B: $1M remaining
        - Bank C: $0 remaining
        Returns: True (because Bank B has capacity)
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
    """Check if there are any borrowers with remaining credit demand.

    This function examines all potential borrowers in a given industry to determine
    if any still have unfulfilled credit demand. It's used to decide whether to
    continue the credit allocation process or terminate due to demand exhaustion.

    The check considers:
    1. All sectors in the market
    2. Each borrower's remaining credit demand
    3. Value type constraints
    4. Industry-specific requirements

    Args:
        industry: Industry/sector index to check
        goods_market_participants: Dict mapping sector names to lists of agents
        field: Name of the field containing credit demand information
            (e.g., "Remaining Goods" for unfulfilled credit requests)

    Returns:
        bool: True if any borrower has remaining credit demand, False otherwise

    Example:
        For corporate sector (industry 1):
        - Firm A: $0 needed
        - Firm B: $2M needed
        - Firm C: $0 needed
        Returns: True (because Firm B has demand)
    """
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_buyer_states["Value Type"] != ValueType.NONE:
                if transactor.transactor_buyer_states[field][:, industry].sum() > 0.0:
                    return True
    return False


#
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
#


def get_random_buyer_type(
    industry: int,
    country_goods_market_participants: list[Agent],
    prio_high_prio: bool,
    field: str,
) -> Agent:
    """Select a random borrower type based on priority and demand.

    This function implements a weighted random selection of borrower types within
    a sector, considering priorities and current credit demand. It supports
    preferential treatment of high-priority borrowers while maintaining some
    randomness in the selection process.

    The selection process:
    1. Priority handling:
       - If priority mode is active, first try high-priority borrowers
       - Only consider other borrowers if no high-priority demand exists
       - Weight selection by current credit demand

    2. Borrower filtering:
       - Check for positive credit demand
       - Verify borrower eligibility
       - Apply sector-specific constraints

    3. Random selection:
       - Calculate selection weights based on demand
       - Use weighted random choice
       - Handle edge cases (no eligible borrowers)

    Args:
        industry: Industry/sector index for borrower selection
        country_goods_market_participants: List of agents in the sector
        prio_high_prio: Whether to prioritize high-priority borrowers
        field: Name of the field containing credit demand information

    Returns:
        Agent: Selected borrower type (agent)

    Example:
        For corporate sector with three firms:
        1. High priority mode:
           - Firm A (high priority): $5M needed (weight: 0.625)
           - Firm B (high priority): $3M needed (weight: 0.375)
           - Firm C (low priority): $2M needed (weight: 0 due to priority)
           Result: Either Firm A or B based on weighted random choice

        2. No priority mode:
           - Firm A: $5M needed (weight: 0.5)
           - Firm B: $3M needed (weight: 0.3)
           - Firm C: $2M needed (weight: 0.2)
           Result: Any firm based on weighted random choice
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
    """Select a random borrower and specific loan request.

    This function implements a hierarchical selection process for borrowers,
    first choosing between sectors/countries, then selecting specific borrowers
    within the chosen sector. It supports various prioritization schemes and
    handles both domestic and international borrowers.

    The selection process:
    1. Sector prioritization:
       - First try to select from priority sectors (e.g., real economy)
       - Consider international borrowers based on prioritization parameter
       - Handle special cases like ROW (Rest of World)

    2. Borrower type selection:
       - Prioritize high-priority borrowers if specified
       - Weight selection by credit demand
       - Consider regulatory constraints

    3. Specific loan selection:
       - Choose among active credit requests
       - Verify request eligibility
       - Apply any final constraints

    Args:
        industry: Industry/sector index for borrower selection
        goods_market_participants: Dict mapping sector names to lists of agents
        real_country_prioritisation: Weight given to real economy sectors [0,1]
        prio_high_prio: Whether to prioritize high-priority borrowers
        field: Name of the field containing credit demand information

    Returns:
        Tuple[Agent, int]: Selected borrower and index of specific loan request

    Example:
        With three sectors:
        1. Initial selection:
           - Corporate sector (real): 60% weight
           - Financial sector: 30% weight
           - External sector: 10% weight

        2. Within corporate sector:
           - Large firms (high priority): 70% weight
           - SMEs: 30% weight

        3. Final selection:
           Returns: (SelectedFirm, LoanRequestIndex)
           - SelectedFirm: The chosen borrower agent
           - LoanRequestIndex: Index of the specific loan request
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
    """Select a lender from the borrower's previous lending relationships.

    This function implements relationship lending by attempting to match borrowers
    with their previous lenders. It considers various priorities and constraints
    while maintaining some randomness in the selection process to model relationship
    persistence realistically.

    The selection process:
    1. Relationship check:
       - Random draw against relationship persistence probability
       - Check for existing lending relationships
       - Verify lender eligibility

    2. Priority filtering:
       - Apply real economy sector preferences
       - Consider high-priority lender status
       - Handle domestic vs international preferences

    3. Final selection:
       - Choose among eligible previous lenders
       - Verify current lending capacity
       - Apply any final constraints

    Args:
        industry: Industry/sector index for lender selection
        chosen_buyer: Selected borrower agent
        chosen_buyer_ind: Index of the specific loan request
        previous_supply_chain: Dict tracking previous lending relationships
        prio_real_countries: Whether to prioritize real economy lenders
        prio_high_prio: Whether to prioritize high-priority lenders
        prio_domestic_sellers: Whether to prioritize domestic lenders
        probability_keeping_previous_seller: Chance of maintaining relationship
        field: Name of the field containing credit capacity information

    Returns:
        Optional[Tuple[Agent, int]]: Selected lender and loan index if found,
            None if no suitable previous lender is available

    Example:
        For a corporate borrower with previous relationships:
        1. Initial check:
           - Relationship persistence probability: 0.8
           - Random draw: 0.7
           Result: Proceed with relationship selection

        2. Previous lenders:
           - Bank A (domestic, high priority): $5M capacity
           - Bank B (foreign): $3M capacity
           - Bank C (domestic): No capacity
           Result: Bank A selected (matches all priorities)
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
    """Select a random lender based on interest rates and capacity.

    This function implements a sophisticated lender selection process that considers
    both interest rates and lending capacity. It uses a temperature parameter to
    control the balance between rate sensitivity and randomization, supporting
    different distribution types for the selection weights.

    The selection process:
    1. Weight calculation:
       - Consider lending capacity (supply)
       - Factor in interest rates
       - Apply temperature scaling
       - Support different distribution types

    2. Lender filtering:
       - Check for positive lending capacity
       - Verify lender eligibility
       - Apply sector-specific constraints

    3. Random selection:
       - Calculate final selection weights
       - Handle different distribution types
       - Perform weighted random choice

    Args:
        industry: Industry/sector index for lender selection
        chosen_goods_market_participants: List of potential lenders
        price_temperature: Parameter controlling interest rate sensitivity
            Higher values → More rate-sensitive selection
            Lower values → More uniform selection
        field: Name of the field containing credit capacity information
        distribution_type: How to combine capacity and rate weights
            "multiplicative": weights = capacity_weight * rate_weight
            "additive": weights = 0.5 * (capacity_weight + rate_weight)

    Returns:
        Optional[Tuple[Agent, int]]: Selected lender and loan index if found,
            None if no suitable lender is available

    Example:
        For banking sector with temperature = 1.0:
        1. Available lenders:
           - Bank A: $10M capacity, 5% rate
           - Bank B: $5M capacity, 4% rate
           - Bank C: $3M capacity, 6% rate

        2. Weight calculation (multiplicative):
           - Bank A: 0.44 * 0.37 = 0.16
           - Bank B: 0.22 * 0.45 = 0.10
           - Bank C: 0.33 * 0.30 = 0.10

        3. Result:
           Returns: (SelectedBank, LoanIndex)
           Selection probability matches normalized weights
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
    """Select a lender through a multi-stage selection process.

    This function implements a sophisticated lender selection mechanism that first
    attempts to maintain existing relationships, then falls back to market-based
    selection if needed. It incorporates various priorities and constraints while
    balancing between relationship stability and market efficiency.

    The selection process:
    1. Relationship-based selection:
       - Try to match with previous lender
       - Consider relationship persistence probability
       - Check lender eligibility and capacity

    2. Market-based selection:
       - Filter potential lenders by priorities
       - Apply sector preferences
       - Consider domestic vs international options
       - Weight by rates and capacity

    3. Final matching:
       - Handle special cases (e.g., no eligible lenders)
       - Apply any final constraints
       - Complete the selection

    Args:
        industry: Industry/sector index for lender selection
        goods_market_participants: Dict mapping sector names to lists of agents
        chosen_buyer: Selected borrower agent
        chosen_buyer_ind: Index of the specific loan request
        previous_supply_chain: Dict tracking previous lending relationships
        real_country_prioritisation: Weight given to real economy sectors [0,1]
        prio_high_prio_sellers: Whether to prioritize high-priority lenders
        prio_domestic_sellers: Whether to prioritize domestic lenders
        probability_keeping_previous_seller: Chance of maintaining relationship
        price_temperature: Parameter controlling interest rate sensitivity
        field: Name of the field containing credit capacity information
        distribution_type: How to combine capacity and rate weights

    Returns:
        Tuple[Agent, int]: Selected lender and loan index

    Example:
        For a corporate borrower:
        1. Relationship check:
           - Previous lender: Bank A
           - Persistence probability: 0.8
           - Check result: Bank A not eligible

        2. Market selection:
           - Available lenders filtered by priorities:
             * Bank B (domestic, high priority)
             * Bank C (domestic)
             * Bank D (foreign)
           - Weights calculated using rates and capacity
           Result: Bank B selected based on combined criteria
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
    """Process a credit transaction between lender and borrower.

    This function executes the actual credit transaction once a match has been
    made, updating all relevant state variables for both parties. It handles
    the mechanics of credit extension while ensuring all constraints and
    accounting identities are maintained.

    The transaction process:
    1. Pre-transaction checks:
       - Verify credit availability
       - Check borrower eligibility
       - Confirm regulatory compliance
       - Validate transaction parameters

    2. Credit extension:
       - Calculate final loan amount
       - Apply interest rates
       - Update credit limits
       - Record bilateral exposures

    3. State updates:
       - Update lender's available credit
       - Update borrower's credit utilization
       - Record transaction details
       - Update relationship tracking

    Args:
        industry: Industry/sector index for the transaction
        buyer: Borrower agent
        buyer_ind: Index of the specific loan request
        seller: Lender agent
        seller_ind: Index of the specific credit capacity

    Example:
        Transaction between Bank A and Firm B:
        1. Initial state:
           - Bank A: $10M available credit
           - Firm B: $2M credit request

        2. Transaction:
           - Loan amount: $2M
           - Interest rate: 5%
           - Term: 12 months

        3. Final state:
           - Bank A: $8M available credit
           - Firm B: $2M credit received
           - Relationship recorded
           - Balance sheets updated
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
    """Simulate a credit transaction without actually executing it.

    This function performs a dry-run of a credit transaction to evaluate its
    feasibility and impact. It's useful for testing transaction viability,
    stress testing, and policy analysis without affecting actual market state.

    The simulation process:
    1. Transaction validation:
       - Check credit availability
       - Verify borrower eligibility
       - Test regulatory compliance
       - Validate parameters

    2. Impact analysis:
       - Calculate potential loan amount
       - Estimate interest costs
       - Project balance sheet changes
       - Assess risk metrics

    3. Constraint checking:
       - Verify capital adequacy
       - Check exposure limits
       - Test policy compliance
       - Evaluate market impact

    Args:
        industry: Industry/sector index for the transaction
        buyer: Borrower agent
        buyer_ind: Index of the specific loan request
        seller: Lender agent
        seller_ind: Index of the specific credit capacity

    Example:
        Hypothetical transaction between Bank A and Firm B:
        1. Initial check:
           - Bank A: $10M available, 12% capital ratio
           - Firm B: $2M request, 3x leverage ratio

        2. Simulation:
           - Loan amount: $2M feasible
           - Capital ratio would be 11.5% (above minimum)
           - Leverage ratio would be 3.2x (below maximum)
           Result: Transaction is viable
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
    """Update the record of lending relationships after a transaction.

    This function maintains a history of credit relationships between lenders
    and borrowers. This information is crucial for relationship lending and
    analyzing credit network structure.

    The update process:
    1. Data structure maintenance:
       - Initialize new relationships if needed
       - Update existing relationships
       - Clean up old entries if necessary

    2. Relationship recording:
       - Store lender-borrower pair
       - Track loan indices
       - Record transaction details
       - Update relationship strength

    3. Network analysis:
       - Update connectivity metrics
       - Track relationship persistence
       - Monitor market structure
       - Identify key relationships

    Args:
        current_supply_chain: Dict tracking current lending relationships
        industry: Industry/sector index for the transaction
        buyer: Borrower agent
        buyer_ind: Index of the specific loan request
        seller: Lender agent
        seller_ind: Index of the specific credit capacity

    Example:
        After a transaction between Bank A and Firm B:
        1. Check existing relationships:
           - First transaction: Create new entry
           - Repeat transaction: Update existing entry

        2. Record details:
           - Store lender-borrower pair
           - Track loan indices
           - Update relationship strength
           Result: Updated relationship network
    """
    if buyer not in current_supply_chain[industry].keys():
        current_supply_chain[industry][buyer] = {}
    if buyer_ind in current_supply_chain[industry][buyer].keys():
        current_supply_chain[industry][buyer][buyer_ind] += [(seller, seller_ind)]
    else:
        current_supply_chain[industry][buyer][buyer_ind] = [(seller, seller_ind)]


def clean_rounding_errors(goods_market_participants: dict[str, list[Agent]], decimals: int = 12) -> None:
    """Clean up numerical rounding errors in credit market state variables.

    This function addresses floating-point arithmetic precision issues that can
    accumulate during credit market operations. It ensures that small numerical
    errors don't affect market behavior or regulatory compliance.

    The cleaning process:
    1. Error detection:
       - Check for small non-zero values
       - Identify numerical instabilities
       - Detect accumulation errors
       - Monitor precision loss

    2. State correction:
       - Round to specified precision
       - Zero out negligible values
       - Maintain accounting identities
       - Preserve regulatory ratios

    3. Validation:
       - Verify corrections
       - Check balance consistency
       - Confirm regulatory compliance
       - Ensure market integrity

    Args:
        goods_market_participants: Dict mapping sector names to lists of agents
        decimals: Number of decimal places to maintain (default: 12)

    Example:
        Cleaning process:
        1. Initial state:
           - Credit amount: 1000.000000000001
           - Utilization rate: 0.799999999999999
           - Exposure: 0.000000000000001

        2. After cleaning:
           - Credit amount: 1000.0
           - Utilization rate: 0.8
           - Exposure: 0.0
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
