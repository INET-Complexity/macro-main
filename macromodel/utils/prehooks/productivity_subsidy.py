"""Productivity subsidy pre-hook for government interventions.

This module provides a factory function to create a pre-hook that implements
a one-time government subsidy to firms for productivity investment.
"""

import logging
from typing import Callable

import numpy as np

from macromodel.simulation import Simulation


def create_productivity_subsidy_hook(
    country_code: str,
    industry_code: str,
    target_year: int,
    target_month: int,
    subsidy_amount: float,
) -> Callable[[Simulation, int, int], None]:
    """Create a pre-hook that provides a one-time productivity subsidy to firms.

    This factory function creates a pre-hook callable that:
    1. Checks if the current timestep matches the target year and month
    2. Identifies firms in the specified country and industry
    3. Adds the subsidy amount to firm deposits (as government transfer)
    4. Forces the subsidy amount into executed productivity investment
    5. Records the subsidy as government expenditure

    The subsidy is directly injected as executed productivity investment, ensuring
    that firms actually invest the money in productivity improvements rather than
    using it for other purposes.

    Args:
        country_code (str): Country code (e.g., "CAN")
        industry_code (str): Industry code (e.g., "D35" for electric power)
        target_year (int): Year when subsidy should be applied
        target_month (int): Month when subsidy should be applied
        subsidy_amount (float): Total subsidy amount in local currency units (split among firms)

    Returns:
        Callable: A pre-hook function with signature (simulation, year, month) -> None

    Example:
        >>> hook = create_productivity_subsidy_hook(
        ...     country_code="CAN",
        ...     industry_code="D35",
        ...     target_year=2014,
        ...     target_month=7,
        ...     subsidy_amount=1_000_000_000  # 1 billion CAD
        ... )
        >>> simulation.prehooks.append(hook)
        >>> simulation.run()
    """
    # Track whether subsidy has been applied (to ensure one-time payment)
    applied = [False]

    def productivity_subsidy_hook(simulation: Simulation, year: int, month: int) -> None:
        """Pre-hook that applies productivity subsidy at the specified timestep."""
        # Check if we've already applied the subsidy
        if applied[0]:
            return

        # Check if this is the target timestep
        if year != target_year or month != target_month:
            return

        # Check if the country exists in the simulation
        if country_code not in simulation.countries:
            logging.warning(
                f"Productivity subsidy hook: Country '{country_code}' not found in simulation. "
                f"Available countries: {list(simulation.countries.keys())}"
            )
            return

        country = simulation.countries[country_code]
        firms = country.firms

        # Get the industry index
        industries = firms.industries
        if industry_code not in industries:
            logging.warning(
                f"Productivity subsidy hook: Industry '{industry_code}' not found in country '{country_code}'. "
                f"Available industries: {industries}"
            )
            return

        industry_index = list(industries).index(industry_code)

        # Find firms in the target industry
        firm_industries = firms.states["Industry"]
        target_firm_mask = firm_industries == industry_index
        n_target_firms = np.sum(target_firm_mask)

        if n_target_firms == 0:
            logging.warning(
                f"Productivity subsidy hook: No firms found in industry '{industry_code}' "
                f"for country '{country_code}'"
            )
            return

        # Split subsidy evenly among target firms
        subsidy_per_firm = subsidy_amount / n_target_firms

        # Add subsidy to firm deposits (as government transfer)
        # Deposits are stored in the time series, modify the current value
        current_deposits = firms.ts.current("deposits").copy()
        current_deposits[target_firm_mask] += subsidy_per_firm
        firms.ts.time_series["deposits"][-1] = current_deposits

        # Force subsidy into executed productivity investment time series
        # This ensures it will be used by compute_tfp_growth() when update_tfp() is called
        # during update_planning_metrics() later in this iteration

        # Create investment array for this timestep
        forced_investment = np.zeros(len(current_deposits))
        forced_investment[target_firm_mask] = subsidy_per_firm

        # Append to executed productivity investment time series
        # This will be picked up by compute_tfp_growth() -> update_tfp() chain
        firms.ts.executed_productivity_investment.append(forced_investment)

        logging.info(
            f"Productivity subsidy applied: {subsidy_amount:,.2f} to {n_target_firms} firm(s) "
            f"in industry '{industry_code}' of country '{country_code}' at {year}-{month}"
        )

        # Record subsidy in government time series as spending
        gov = country.central_government
        current_spending = gov.ts.current("spending")
        if isinstance(current_spending, (list, np.ndarray)):
            new_spending = current_spending[0] + subsidy_amount
        else:
            new_spending = current_spending + subsidy_amount

        # Update the current spending value
        gov.ts.time_series["spending"][-1] = new_spending

        # Mark as applied
        applied[0] = True

    return productivity_subsidy_hook
