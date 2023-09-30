import scipy as sp
import numpy as np

from tqdm import trange
from scipy.optimize import linear_sum_assignment as lsa

from inet_data.processing.synthetic_firms.synthetic_firms import (
    SyntheticFirms,
)
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


def match_individuals_with_firms(
    synthetic_population: SyntheticPopulation,
    synthetic_firms: SyntheticFirms,
    employee_social_contribution_taxes: float,
    income_taxes: float,
) -> None:
    synthetic_firms.firm_data["Employees ID"] = [[] for _ in range(len(synthetic_firms.firm_data))]
    synthetic_population.individual_data["Corresponding Firm ID"] = np.nan

    # Iterate over industries
    for industry in trange(
        len(synthetic_firms.industries),
        desc="Matching Firms with Individuals for " + synthetic_population.country_name,
    ):
        # Preprocess inet_data
        wages_offered, pos_corr_firm = preprocess(
            synthetic_population=synthetic_population,
            synthetic_firms=synthetic_firms,
            industry=industry,
            income_taxes=income_taxes,
            employee_social_contribution_taxes=employee_social_contribution_taxes,
        )

        # Optimize
        find_optimal_matching(
            synthetic_population=synthetic_population,
            synthetic_firms=synthetic_firms,
            industry=industry,
            income_taxes=income_taxes,
            employee_social_contribution_taxes=employee_social_contribution_taxes,
            wages_offered=wages_offered,
            pos_corr_firm=pos_corr_firm,
        )


def preprocess(
    synthetic_population: SyntheticPopulation,
    synthetic_firms: SyntheticFirms,
    industry: int,
    income_taxes: float,
    employee_social_contribution_taxes: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Sanity check
    ind_employees = np.where(
        np.logical_and(
            synthetic_population.individual_data["Employment Industry"] == industry,
            synthetic_population.individual_data["Activity Status"] == 1,
        )
    )[0]
    num_employees = synthetic_population.number_employees_by_industry[industry]
    if len(ind_employees) != num_employees:
        raise ValueError(
            "Employee Employment NACEs do not match the numbers of employees",
            len(ind_employees),
            num_employees,
        )

    # Rescale income
    firm_ids = np.where(synthetic_firms.firm_data["Industry"] == industry)[0]
    total_employee_income = synthetic_population.individual_data.loc[ind_employees, "Employee Income"].sum()
    if total_employee_income == 0.0:
        synthetic_population.individual_data.loc[ind_employees, "Employee Income"] = 0.0
    else:
        total_paid_by_firms = synthetic_firms.firm_data.loc[
            firm_ids,
            "Total Wages",
        ].values.sum()
        synthetic_population.individual_data.loc[ind_employees, "Employee Income"] *= (
            (1 - employee_social_contribution_taxes - income_taxes * (1 - employee_social_contribution_taxes))
            * total_paid_by_firms
            / total_employee_income
        )

    # Create positions
    wages_offered = np.concatenate(
        [
            [
                (1 - employee_social_contribution_taxes - income_taxes * (1 - employee_social_contribution_taxes))
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


def find_optimal_matching(
    synthetic_population: SyntheticPopulation,
    synthetic_firms: SyntheticFirms,
    industry: int,
    income_taxes: float,
    employee_social_contribution_taxes: float,
    wages_offered: np.ndarray,
    pos_corr_firm: np.ndarray,
    normalise_employee_income: bool = True,
) -> None:
    # Create a cost matrix
    employed_ind = np.where(
        np.logical_and(
            synthetic_population.individual_data["Employment Industry"] == industry,
            synthetic_population.individual_data["Activity Status"] == 1,
        )
    )[0]
    inc_rec = synthetic_population.individual_data["Employee Income"].values[employed_ind]
    cost = sp.spatial.distance_matrix(inc_rec[:, None], wages_offered[:, None])

    # Find the optimal configuration
    corr_individuals_rel, corr_positions = lsa(cost)
    corr_individuals = np.full(len(synthetic_population.individual_data), np.nan)
    corr_individuals[employed_ind] = corr_individuals_rel
    corr_firms = pos_corr_firm[corr_positions]

    # Record the results
    synthetic_population.individual_data.loc[employed_ind, "Corresponding Firm ID"] = corr_firms
    for ind, firm_id in enumerate(np.where(synthetic_firms.firm_data["Industry"] == industry)[0]):
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
