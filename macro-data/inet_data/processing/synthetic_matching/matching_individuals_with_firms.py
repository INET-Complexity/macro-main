import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa
from tqdm import trange

from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from inet_data.readers.default_readers import DataReaders


def preprocess(
    synthetic_population: SyntheticPopulation,
    synthetic_firms: SyntheticFirms,
    industry: int,
    income_taxes: float,
    employee_social_contribution_taxes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the data for matching individuals with firms.

    This function performs a sanity check to ensure the number of employees in the industry matches the expected number.
    It then rescales the income of the employees in the industry based on the total wages paid by the firms.
    Finally, it creates two arrays representing the wages offered by each firm for each position
    and the corresponding firm IDs.

    Args:
        synthetic_population (SyntheticPopulation): An instance of the SyntheticPopulation class.
        synthetic_firms (SyntheticFirms): An instance of the SyntheticFirms class.
        industry (int): The industry to be processed.
        income_taxes (float): The income tax rate.
        employee_social_contribution_taxes (float): The employee social contribution tax rate.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays. The first array represents the wages offered by each firm for each position.
        The second array contains the firm IDs corresponding to each position.

    Raises:
        ValueError: If the number of employees in the industry does not match the expected number.
    """
    # Sanity check
    ind_employees = np.flatnonzero(
        (synthetic_population.individual_data["Employment Industry"] == industry)
        & (synthetic_population.individual_data["Activity Status"] == 1)
    )
    num_employees = synthetic_population.number_employees_by_industry[industry]
    if len(ind_employees) != num_employees:
        raise ValueError(
            "Employee Employment NACEs do not match the numbers of employees",
            len(ind_employees),
            num_employees,
        )

    # Rescale income
    firm_ids = np.flatnonzero(synthetic_firms.firm_data["Industry"] == industry)
    total_employee_income = synthetic_population.individual_data.loc[ind_employees, "Employee Income"].sum()
    labour_taxes = 1 - employee_social_contribution_taxes - income_taxes * (1 - employee_social_contribution_taxes)
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
            labour_taxes * total_paid_by_firms / total_employee_income
        )

    # Create positions
    wages_offered = np.concatenate(
        [
            [
                labour_taxes
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
    """
    Finds the optimal matching of individuals to firms in a given industry.

    This function creates a cost matrix based on the distance between the income of each individual and the wages offered by each firm.
    It then uses a linear sum assignemnt algorithm to find the optimal matching of individuals to firms, minimising the difference between
    asked and offered wages.
    The results are recorded in the data of the synthetic population and firms.

    Args:
        synthetic_population (SyntheticPopulation): An instance of the SyntheticPopulation class.
        synthetic_firms (SyntheticFirms): An instance of the SyntheticFirms class.
        industry (int): The industry to be processed.
        income_taxes (float): The income tax rate.
        employee_social_contribution_taxes (float): The employee social contribution tax rate.
        wages_offered (np.ndarray): An array representing the wages offered by each firm for each position.
        pos_corr_firm (np.ndarray): An array containing the firm IDs corresponding to each position.
        normalise_employee_income (bool, optional): Whether to normalise the income of the employees. Defaults to True.

    Returns:
        None
    """
    # Create a cost matrix
    in_industry = synthetic_population.individual_data["Employment Industry"] == industry
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


def match_individuals_with_firms_country(
    country: str,
    industries: list[int] | np.ndarray,
    readers: DataReaders,
    synthetic_firms: SyntheticFirms,
    synthetic_population: SyntheticPopulation,
    year: int,
):
    """
    Matches individuals with firms based on country, industries, and other parameters.

    Args:
        country (str): The country for which the matching is performed.
        industries (list[int] | np.ndarray): The industries to consider for matching.
        readers (DataReaders): An object that provides access to data readers.
        synthetic_firms (SyntheticFirms): An object that represents synthetic firms data.
        synthetic_population (SyntheticPopulation): An object that represents synthetic population data.
        year (int): The year for which the matching is performed.

    Returns:
        None
    """
    for industry in range(len(industries)):
        wage_offered, pos_corr_firm = preprocess(
            synthetic_population=synthetic_population,
            synthetic_firms=synthetic_firms,
            industry=industry,
            income_taxes=readers.oecd_econ.read_tau_income(country=country, year=year),
            employee_social_contribution_taxes=readers.oecd_econ.read_tau_siw(country=country, year=year),
        )

        find_optimal_matching(
            synthetic_population=synthetic_population,
            synthetic_firms=synthetic_firms,
            industry=industry,
            income_taxes=readers.oecd_econ.read_tau_income(country=country, year=year),
            employee_social_contribution_taxes=readers.oecd_econ.read_tau_siw(country=country, year=year),
            wages_offered=wage_offered,
            pos_corr_firm=pos_corr_firm,
        )
