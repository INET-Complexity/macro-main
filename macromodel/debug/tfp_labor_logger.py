"""
Debug logging for TFP growth and labor market investigation (H1: Labor Substitution).

This module provides targeted logging to test the hypothesis that TFP growth causes
unemployment by allowing firms to produce the same output with fewer workers while
demand expectations remain flat.

Key mechanism being tested:
    TFP↑ → effective_capacity↑ → BUT demand_expectations flat
    → target_production stays same → desired_labour↓ → unemployment↑ → income↓
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus


@dataclass
class TFPLaborSnapshot:
    """Snapshot of TFP and labor dynamics at a single timestep.

    Captures the key variables needed to test H1: Labor Substitution hypothesis.
    """

    # Timestep info
    t: int
    year: int
    month: int

    # TFP and productivity
    avg_tfp_multiplier: float
    tfp_by_industry: np.ndarray  # Average TFP per industry
    executed_productivity_investment: float  # Total investment in TFP

    # Production and capacity
    total_production: float
    total_target_production: float
    avg_capacity_utilization: float  # production / target_production

    # Demand expectations
    total_estimated_demand: float
    total_estimated_growth: float  # Average growth estimate

    # Labor market
    total_desired_labour: float  # What firms want
    total_actual_labour: float  # What they get
    total_employment: int  # Number of employed individuals
    unemployment_rate: float
    labor_shortage: float  # desired - actual (positive = shortage, negative = surplus)

    # Wages and income
    total_wage_bill: float
    avg_wage: float
    total_household_income: float

    # Inventory (signal of demand mismatch)
    total_inventory: float

    def labor_utilization(self) -> float:
        """Fraction of desired labor that firms actually get."""
        if self.total_desired_labour > 0:
            return self.total_actual_labour / self.total_desired_labour
        return 1.0

    def demand_supply_gap(self) -> float:
        """Gap between demand and production (positive = excess demand)."""
        return self.total_estimated_demand - self.total_production


@dataclass
class TFPLaborLog:
    """Complete log of TFP-labor dynamics across simulation."""

    snapshots: list[TFPLaborSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: TFPLaborSnapshot):
        """Add a timestep snapshot to the log."""
        self.snapshots.append(snapshot)

    def to_dict_list(self) -> list[dict]:
        """Convert to list of dictionaries for easy DataFrame creation."""
        return [
            {
                "t": s.t,
                "year": s.year,
                "month": s.month,
                "avg_tfp_multiplier": s.avg_tfp_multiplier,
                "executed_productivity_investment": s.executed_productivity_investment,
                "total_production": s.total_production,
                "total_target_production": s.total_target_production,
                "avg_capacity_utilization": s.avg_capacity_utilization,
                "total_estimated_demand": s.total_estimated_demand,
                "total_estimated_growth": s.total_estimated_growth,
                "total_desired_labour": s.total_desired_labour,
                "total_actual_labour": s.total_actual_labour,
                "total_employment": s.total_employment,
                "unemployment_rate": s.unemployment_rate,
                "labor_shortage": s.labor_shortage,
                "labor_utilization": s.labor_utilization(),
                "total_wage_bill": s.total_wage_bill,
                "avg_wage": s.avg_wage,
                "total_household_income": s.total_household_income,
                "total_inventory": s.total_inventory,
                "demand_supply_gap": s.demand_supply_gap(),
            }
            for s in self.snapshots
        ]


def capture_tfp_labor_snapshot(simulation, t: int) -> TFPLaborSnapshot:
    """
    Capture a snapshot of TFP and labor dynamics from a simulation.

    This should be called AFTER labour market clearing in each timestep
    to capture the realized employment outcomes.

    Args:
        simulation: The Simulation object
        t: Current timestep index

    Returns:
        TFPLaborSnapshot with all relevant metrics
    """
    # Assuming single country for now (CAN)
    country = simulation.countries["CAN"]
    firms = country.firms
    households = country.households
    individuals = country.individuals

    # TFP metrics
    tfp_multipliers = firms.states["tfp_multiplier"]
    avg_tfp = np.mean(tfp_multipliers)

    # Group TFP by industry (average across firms in each industry)
    n_industries = firms.n_industries
    tfp_by_industry = np.zeros(n_industries)
    industry_indices = firms.states["Industry"]  # Industry index for each firm
    for ind in range(n_industries):
        industry_mask = industry_indices == ind
        if industry_mask.sum() > 0:
            tfp_by_industry[ind] = tfp_multipliers[industry_mask].mean()

    # Productivity investment
    if len(firms.ts.executed_productivity_investment) > 0:
        executed_inv = firms.ts.current("executed_productivity_investment").sum()
    else:
        executed_inv = 0.0

    # Production metrics
    production = firms.ts.current("production")
    target_production = firms.ts.current("target_production")
    total_prod = production.sum()
    total_target = target_production.sum()
    capacity_util = total_prod / (total_target + 1e-12)

    # Demand expectations
    estimated_demand = firms.ts.current("estimated_demand")
    total_est_demand = estimated_demand.sum()

    # Growth estimates (if available)
    if hasattr(firms, "ts") and "estimated_growth" in dir(firms.ts):
        try:
            growth_estimates = firms.ts.current("estimated_growth")
            avg_growth = np.mean(growth_estimates)
        except:
            avg_growth = 0.0
    else:
        avg_growth = 0.0

    # Labor metrics
    desired_labour = firms.ts.current("desired_labour_inputs")
    actual_labour = firms.ts.current("labour_inputs")
    total_desired = desired_labour.sum()
    total_actual = actual_labour.sum()

    # Employment from individuals
    activity_status = individuals.states["Activity Status"]
    total_employed = np.sum(activity_status == ActivityStatus.EMPLOYED)
    total_unemployed = np.sum(activity_status == ActivityStatus.UNEMPLOYED)
    labor_force = total_employed + total_unemployed
    unemployment_rate = (total_unemployed / labor_force) if labor_force > 0 else 0.0

    labor_shortage = total_desired - total_actual

    # Wages and income
    total_wage = firms.ts.current("total_wage")
    total_wages = total_wage.sum()
    avg_wage = total_wage.mean()

    # Household income
    household_income = households.ts.current("income")
    total_income = household_income.sum()

    # Inventory
    inventory = firms.ts.current("inventory")
    total_inventory = inventory.sum()

    return TFPLaborSnapshot(
        t=t,
        year=simulation.timestep.year,
        month=simulation.timestep.month,
        avg_tfp_multiplier=avg_tfp,
        tfp_by_industry=tfp_by_industry,
        executed_productivity_investment=executed_inv,
        total_production=total_prod,
        total_target_production=total_target,
        avg_capacity_utilization=capacity_util,
        total_estimated_demand=total_est_demand,
        total_estimated_growth=avg_growth,
        total_desired_labour=total_desired,
        total_actual_labour=total_actual,
        total_employment=total_employed,
        unemployment_rate=unemployment_rate,
        labor_shortage=labor_shortage,
        total_wage_bill=total_wages,
        avg_wage=avg_wage,
        total_household_income=total_income,
        total_inventory=total_inventory,
    )
