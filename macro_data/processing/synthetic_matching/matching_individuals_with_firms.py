"""Module for harmonizing individual and firm employment data.

This module matches and harmonizes employment data from different sources:
1. Individual Data Source:
   - Actual wages from household surveys
   - Industry of employment
   - Activity status
   - Employment income

2. Firm Data Source:
   - Total labor expenses
   - Number of employees
   - Industry classification
   - Production data

The harmonization process involves:
1. Data Validation:
   - Checking employment numbers match across sources
   - Ensuring industry totals align
   - Validating activity status

2. Data Reconciliation:
   - Scaling individual wages to match firm labor expenses
   - Adjusting for tax effects
   - Computing per-position wages from total expenses

3. Optimal Matching:
   - Minimizing discrepancy between data sources
   - Preserving industry-specific relationships
   - Recording final assignments

Note:
    This module focuses on harmonizing employment data from different sources
    to create a consistent initial state. The actual labor market dynamics
    are implemented in the simulation package.
"""

import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa

from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


def preprocess(
    synthetic_population: SyntheticPopulation,
    synthetic_firms: SyntheticFirms,
    industry_index: int,
    income_taxes: float,
    employee_social_contribution_taxes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess and validate employment data from both sources.

    This function ensures consistency between individual and firm data by:
    1. Validating employment counts match between sources
    2. Reconciling wage totals with firm labor expenses
    3. Computing position wages from firm total expenses
    4. Adjusting for tax effects

    The preprocessing ensures that:
    - Employment numbers match across sources
    - Wage totals align with firm expenses
    - Tax effects are properly accounted for
    - Industry-specific relationships are maintained

    Args:
        synthetic_population (SyntheticPopulation): Individual employment data
        synthetic_firms (SyntheticFirms): Firm labor expense data
        industry_index (int): Industry sector being processed
        income_taxes (float): Income tax rate for reconciliation
        employee_social_contribution_taxes (float): Social security tax rate

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - wages_offered: Array of wages derived from firm expenses
            - pos_corr_firm: Array mapping positions to firm IDs

    Raises:
        ValueError: If employment counts don't match between sources
    """
    # Sanity check
    ind_employees = np.flatnonzero(
        (synthetic_population.individual_data["Employment Industry"] == industry_index)
        & (synthetic_population.individual_data["Activity Status"] == 1)
    )
    num_employees = synthetic_population.number_employees_by_industry[industry_index]
    if len(ind_employees) != num_employees:
        raise ValueError(
            "Employee Employment NACEs do not match the numbers of employees",
            len(ind_employees),
            num_employees,
        )

    # Rescale income
    firm_ids = np.flatnonzero(synthetic_firms.firm_data["Industry"] == industry_index)
    total_employee_income = synthetic_population.individual_data.loc[ind_employees, "Employee Income"].sum()
    labour_taxrate = 1 - employee_social_contribution_taxes - income_taxes * (1 - employee_social_contribution_taxes)
    if total_employee_income == 0.0:
        synthetic_population.individual_data.loc[ind_employees, "Employee Income"] = 0.0
    else:
        # total amount paid by firms
        total_paid_by_firms = synthetic_firms.firm_data.loc[
            firm_ids,
            "Total Wages",
        ].values.sum()
        # distribute the total amount paid by firms to employees,
        # taking into account labour tax
        synthetic_population.individual_data.loc[ind_employees, "Employee Income"] *= (
            labour_taxrate * total_paid_by_firms / total_employee_income
        )

    # Create positions
    if len(firm_ids) > 0:
        wages_offered = np.concatenate(
            [
                [
                    labour_taxrate
                    * synthetic_firms.firm_data.at[firm_id, "Total Wages"]
                    / synthetic_firms.firm_data.at[firm_id, "Number of Employees"]
                ]
                * synthetic_firms.firm_data.at[firm_id, "Number of Employees"]
                for firm_id in firm_ids
            ]
        )

        pos_corr_firm = np.concatenate(
            [[firm_id] * synthetic_firms.firm_data.at[firm_id, "Number of Employees"] for firm_id in firm_ids]
        )

        return wages_offered, pos_corr_firm
    else:
        return np.array([]), np.array([])


def find_optimal_matching(
    synthetic_population: SyntheticPopulation,
    synthetic_firms: SyntheticFirms,
    industry_index: int,
    income_taxes: float,
    employee_social_contribution_taxes: float,
    wages_offered: np.ndarray,
    pos_corr_firm: np.ndarray,
    normalise_employee_income: bool = True,
) -> None:
    """Find optimal matches to harmonize individual and firm data.

    This function reconciles employment data by:
    1. Creating a cost matrix from wage differences
    2. Finding optimal assignments that minimize discrepancies
    3. Recording harmonized relationships
    4. Adjusting final wages to match firm expenses

    The optimization:
    - Minimizes differences between reported wages and firm expenses
    - Maintains industry-specific relationships
    - Accounts for tax effects in reconciliation
    - Preserves firm-level employment totals

    Args:
        synthetic_population (SyntheticPopulation): Individual employment data
        synthetic_firms (SyntheticFirms): Firm labor expense data
        industry_index (int): Industry sector being processed
        income_taxes (float): Income tax rate for reconciliation
        employee_social_contribution_taxes (float): Social security tax rate
        wages_offered (np.ndarray): Wages derived from firm expenses
        pos_corr_firm (np.ndarray): Array mapping positions to firms
        normalise_employee_income (bool, optional): Whether to adjust wages
            to match firm expenses. Defaults to True.
    """
    # Create a cost matrix
    in_industry = synthetic_population.individual_data["Employment Industry"] == industry_index
    employed = synthetic_population.individual_data["Activity Status"] == 1
    employed_ind = np.flatnonzero(in_industry & employed)
    employed_wages = synthetic_population.individual_data["Employee Income"].values[employed_ind]
    cost = sp.spatial.distance_matrix(employed_wages[:, None], wages_offered[:, None])

    # Find the optimal configuration
    corr_individuals_rel, corr_positions = lsa(cost)
    corr_individuals = np.full(len(synthetic_population.individual_data), np.nan)
    corr_individuals[employed_ind] = corr_individuals_rel
    corr_firms = pos_corr_firm[corr_positions]

    # Record the results
    synthetic_population.individual_data.loc[employed_ind, "Corresponding Firm ID"] = corr_firms
    for ind, firm_id in enumerate(np.where(synthetic_firms.firm_data["Industry"] == industry_index)[0]):
        corr_ind = employed_ind[corr_firms == firm_id]
        synthetic_firms.firm_data.at[firm_id, "Employees ID"] = list(corr_ind)

        if normalise_employee_income:
            synthetic_population.individual_data.loc[corr_ind, "Employee Income"] *= (
                1 - employee_social_contribution_taxes - income_taxes * (1 - employee_social_contribution_taxes)
            ) * (
                synthetic_firms.firm_data.at[firm_id, "Total Wages"]
                / synthetic_population.individual_data.loc[corr_ind, "Employee Income"].sum()
            )
        else:
            synthetic_firms.firm_data.at[firm_id, "Total Wages"] = (
                1.0
                / (1 - employee_social_contribution_taxes - income_taxes * (1 - employee_social_contribution_taxes))
                * synthetic_population.individual_data.loc[corr_ind, "Employee Income"].sum()
            )

    # Update individual income
    synthetic_population.set_income()


def match_individuals_with_firms_country(
    industries: list[str] | np.ndarray,
    income_taxes: float,
    employee_social_contribution_taxes: float,
    firms: SyntheticFirms,
    population: SyntheticPopulation,
):
    """Harmonize employment data across all industries in a country.

    This function coordinates the complete data harmonization by:
    1. Processing each industry sector
    2. Validating employment data consistency
    3. Reconciling wages with labor expenses
    4. Computing final labor inputs

    The process ensures:
    - Industry-specific relationships are preserved
    - Tax effects are properly handled
    - Wage totals match firm expenses
    - Employment counts are consistent

    Args:
        industries (list[str] | np.ndarray): Industry sectors to process
        income_taxes (float): Income tax rate for reconciliation
        employee_social_contribution_taxes (float): Social security tax rate
        firms (SyntheticFirms): Firm labor expense data
        population (SyntheticPopulation): Individual employment data
    """
    for industry_index in range(len(industries)):
        wage_offered, pos_corr_firm = preprocess(
            synthetic_population=population,
            synthetic_firms=firms,
            industry_index=industry_index,
            income_taxes=income_taxes,
            employee_social_contribution_taxes=employee_social_contribution_taxes,
        )

        find_optimal_matching(
            synthetic_population=population,
            synthetic_firms=firms,
            industry_index=industry_index,
            income_taxes=income_taxes,
            employee_social_contribution_taxes=employee_social_contribution_taxes,
            wages_offered=wage_offered,
            pos_corr_firm=pos_corr_firm,
        )

    population.set_individual_labour_inputs(
        firm_production=firms.firm_data["Production"], firm_employees=firms.firm_data["Employees ID"]
    )
