"""Module for preprocessing firm-specific data utilities.

This module provides utility functions for preprocessing firm-level data that will
be used to initialize behavioral models. Key preprocessing includes:

1. Firm Size Processing:
   - Power-law distribution fitting
   - Employee allocation
   - Size distribution estimation

2. Financial Data Processing:
   - Wage calculations
   - Production allocation
   - Balance sheet construction

3. Parameter Processing:
   - Input requirements
   - Productivity metrics
   - Initial state parameters

Note:
    This module is NOT used for simulating firm behavior. It only handles
    the preprocessing and organization of firm-specific data that will later
    be used to initialize behavioral models in the simulation package.
"""

import logging
from functools import reduce

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

NDArrayInt = npt.NDArray[np.int_]


def draw_industry_firm_sizes(
    n_firms_in_industry: int,
    number_employees: int,
    firm_size_zeta_shape: float,
) -> np.ndarray:
    """Preprocess firm size distribution for an industry.

    This function generates a realistic firm size distribution using a power-law
    model, which is commonly observed in empirical firm size data. The distribution
    is used to initialize the firm size structure within each industry.

    The size distribution follows a discrete power-law with probability:
    ..math::
        p(n) = \frac{1}{n^{\zeta(s)} \zeta(s)}

    where s is the shape parameter controlling the distribution's tail.

    Note:
        This is a preprocessing function. The actual firm growth dynamics are
        implemented in the simulation package.

    Args:
        n_firms_in_industry (int): Number of firms to generate sizes for
        number_employees (int): Total employees to distribute
        firm_size_zeta_shape (float): Power-law shape parameter

    Returns:
        np.ndarray: Array of preprocessed firm sizes
    """
    employees_by_industry_range = np.arange(1, number_employees + 1)
    if len(employees_by_industry_range) == 0:
        return np.zeros(n_firms_in_industry)
    probs = 1 / (employees_by_industry_range**firm_size_zeta_shape * special.zetac(firm_size_zeta_shape))
    sizes_raw = np.random.choice(
        employees_by_industry_range,
        p=probs / sum(probs),
        size=n_firms_in_industry,
        replace=True,
    )
    n_emp_dist = number_employees - n_firms_in_industry

    sizes_raw = sizes_raw / sizes_raw.sum() * n_emp_dist
    sizes = 1 + np.rint(sizes_raw).astype("int")

    while sum(sizes) > number_employees:
        idx = np.random.randint(0, len(sizes))
        if sizes[idx] > 1:
            sizes[idx] = sizes[idx] - 1

    while sum(sizes) < number_employees:
        idx = np.random.randint(0, len(sizes))
        sizes[idx] = sizes[idx] + 1

    assert number_employees == sum(sizes)

    return sizes


def distribute_industry_employee_remainder(
    sizes: np.ndarray, number_employees: int, n_firms_in_industry: int
) -> np.ndarray:
    """
    Distributes the remainder of employees among firms in an industry. This remainder is the difference between the
    number of employees in the industry and the sum of the firm sizes (computed a the power-law distribution).

    Args:
        sizes (np.ndarray): An array of firm sizes.
        number_employees (int): The total number of employees in the industry.
        n_firms_in_industry (int): The number of firms in the industry.

    Returns:
        np.ndarray: An updated array of firm sizes after distributing the remainder of employees.
    """
    remainder = number_employees - sizes.sum()
    abs_rem = np.abs(remainder)
    f_choices = np.random.choice(n_firms_in_industry, size=int(abs_rem))
    for f in f_choices:
        new_size = sizes[f] + np.sign(remainder)
        while new_size <= 1:
            f = np.random.choice(n_firms_in_industry)
            new_size = sizes[f] + np.sign(remainder)
        sizes[f] = new_size
    return sizes


def add_number_employees_compustat(
    firm_data: pd.DataFrame,
    compustat_data: pd.DataFrame,
    n_emp_per_industry: np.ndarray | list,
    n_firms_per_industry: np.ndarray | list,
    n_industries: int,
):
    # n_firms = n_firms_per_industry.sum()
    # firms_inds = np.random.choice(range(len(compustat_data)), n_firms, replace=True)
    #
    # # select firms with those indices
    # compustat_data = compustat_data.iloc[firms_inds]

    firm_data["Number of Employees"] = 0
    current_firm_ind = 0
    for industry in range(n_industries):
        if n_emp_per_industry[industry] < n_firms_per_industry[industry]:
            logging.warning(
                f"Fewer Firms than Employees in Sector {industry},\n \
                             Employees by industry: {n_emp_per_industry[industry]},\n \
                            Firms by industry: {n_firms_per_industry[industry]}"
            )

        compustat_subset = compustat_data.iloc[current_firm_ind : current_firm_ind + n_firms_per_industry[industry]]
        sizes = compustat_subset["Number of Employees"].values

        sizes_norm = np.ones_like(sizes) if sizes.sum() == 0 else sizes / sizes.sum()

        sizes_red = np.ones_like(sizes_norm)

        # compute the difference between the number of employees and the sum of the firm sizes
        remainder = n_emp_per_industry[industry] - sizes_red.sum()

        # if the remainder is positive, allocate proportionally to the firm sizes
        if remainder > 0:
            redistributed = np.floor(sizes_norm * remainder).astype(int)
            sizes_red += redistributed

        # offset = 0
        #
        # sizes_red = np.maximum(1, np.floor(sizes_norm * n_emp_per_industry[industry]) - offset).astype(int)
        # while sum(sizes_red) > n_emp_per_industry[industry]:
        #     sizes_red = np.maximum(1, np.floor(sizes_norm * n_emp_per_industry[industry]) - offset).astype(int)
        #     offset += 1

        sizes = distribute_industry_employee_remainder(
            sizes_red,
            number_employees=n_emp_per_industry[industry],
            n_firms_in_industry=n_firms_per_industry[industry],
        )

        firm_data.loc[firm_data["Industry"] == industry, "Number of Employees"] = sizes

        current_firm_ind += n_firms_per_industry[industry]
    return firm_data


def add_number_employees_random(
    firm_data: pd.DataFrame,
    firm_size_zetas: np.ndarray | list | dict[int, float],
    n_employees_per_industry: np.ndarray | list,
    n_firms_per_industry: np.ndarray | list,
    n_industries: int,
):
    """Preprocess employee allocation across firms.

    This function distributes employees across firms within each industry using
    power-law size distributions. The preprocessing steps include:
    1. Computing initial firm sizes from power-law distributions
    2. Distributing remaining employees to maintain industry totals
    3. Updating the firm data with allocated employees

    Note:
        This is a preprocessing function. The actual labor market dynamics are
        implemented in the simulation package.

    Args:
        firm_data (pd.DataFrame): Firm data container to update
        firm_size_zetas (np.ndarray | list | dict[int, float]): Power-law parameters
        n_employees_per_industry (np.ndarray | list): Industry employment totals
        n_firms_per_industry (np.ndarray | list): Industry firm counts
        n_industries (int): Number of industries

    Returns:
        pd.DataFrame: Updated firm data with employee allocations
    """
    firm_data["Number of Employees"] = 0
    for industry in range(n_industries):
        # Sanity check
        if n_employees_per_industry[industry] < n_firms_per_industry[industry]:
            logging.warning(
                f"Fewer Firms than Employees in Sector {industry},\n \
                             Employees by industry: {n_employees_per_industry[industry]},\n \
                            Firms by industry: {n_firms_per_industry[industry]}"
            )

        # Draw firm sizes
        sizes = draw_industry_firm_sizes(
            n_firms_in_industry=int(n_firms_per_industry[industry]),
            number_employees=int(n_employees_per_industry[industry]),
            firm_size_zeta_shape=firm_size_zetas[industry],
        )

        # Distribute the remainder
        sizes = distribute_industry_employee_remainder(
            sizes,
            number_employees=int(n_employees_per_industry[industry]),
            n_firms_in_industry=int(n_firms_per_industry[industry]),
        )

        # Update the field
        firm_data.loc[firm_data["Industry"] == industry, "Number of Employees"] = sizes
        firm_data["Number of Employees"] = firm_data["Number of Employees"].astype(int)
    return firm_data


def add_wages(
    firm_data: pd.DataFrame,
    n_employees_per_industry: list | np.ndarray,
    n_firms: int,
    n_industries: int,
    labour_compensation: np.ndarray,
    tau_sif: float,
) -> pd.DataFrame:
    """Preprocess wage data for firms.

    This function initializes wage-related data for each firm based on industry
    labor compensation and tax rates. The preprocessing includes:
    1. Computing base wages from industry compensation
    2. Calculating gross wages with employer taxes
    3. Distributing wages proportionally to firm size

    Note:
        This is a preprocessing function. The actual wage setting behavior is
        implemented in the simulation package.

    Args:
        firm_data (pd.DataFrame): Firm data container to update
        n_employees_per_industry (list | np.ndarray): Industry employment totals
        n_firms (int): Total number of firms
        n_industries (int): Number of industries
        labour_compensation (np.ndarray): Industry labor compensation
        tau_sif (float): Employer social insurance tax rate

    Returns:
        pd.DataFrame: Updated firm data with wage information
    """
    firm_data["Total Wages"] = 0
    firm_data["Total Wages Paid"] = 0
    firm_wages = np.zeros(n_firms)
    for industry in range(n_industries):
        if n_employees_per_industry[industry] > 0:
            firm_wages[firm_data["Industry"] == industry] = (
                firm_data.loc[
                    firm_data["Industry"] == industry,
                    "Number of Employees",
                ]
                / n_employees_per_industry[industry]
                * (labour_compensation[industry] / (1 + tau_sif))
            )
    firm_data["Total Wages"] = firm_wages
    firm_data["Total Wages Paid"] = (1 + tau_sif) * firm_wages
    return firm_data


def add_production(
    firm_data: pd.DataFrame,
    n_employees_per_industry: list | np.ndarray,
    n_industries: int,
    output: np.ndarray,
) -> pd.DataFrame:
    """Preprocess production data for firms.

    This function initializes production-related data for each firm based on
    industry output and employment. The preprocessing includes:
    1. Allocating industry output to firms
    2. Setting initial production levels
    3. Converting to real terms (USD)

    Note:
        This is a preprocessing function. The actual production decisions are
        implemented in the simulation package.

    Args:
        firm_data (pd.DataFrame): Firm data container to update
        n_employees_per_industry (list | np.ndarray): Industry employment totals
        n_industries (int): Number of industries
        output (np.ndarray): Industry output values

    Returns:
        pd.DataFrame: Updated firm data with production values
    """
    firm_data["Production"] = np.nan
    for industry in range(n_industries):
        industry_mask = firm_data["Industry"] == industry
        # attribute production to firms, proportionally by number of employees
        firm_data.loc[industry_mask, "Production"] = (
            firm_data.loc[industry_mask, "Number of Employees"] * output[industry]
        ) / n_employees_per_industry[industry]
        firm_data.loc[industry_mask & (firm_data["Production"] == np.inf), "Production"] = 0
    return firm_data


def initialise_basic_firm_fields_compustat(
    firm_data: pd.DataFrame,
    industry_data: dict[str, pd.DataFrame],
    compustat_data: pd.DataFrame,
    n_employees_per_industry: np.ndarray | list,
    n_firms_per_industry: NDArrayInt | list[int],
    exchange_rate: float,
    tau_sif: float,
    assume_initial_unit: bool = False,
):
    n_industries = len(n_employees_per_industry)
    n_firms = sum(n_firms_per_industry)
    firm_data["Industry"] = np.array(
        reduce(
            lambda a, b: a + b,
            ([industry] * s for industry, s in enumerate(n_firms_per_industry)),
        )
    )

    firms_inds = np.random.choice(range(len(compustat_data)), n_firms, replace=True)

    # select firms with those indices
    compustat_data = compustat_data.iloc[firms_inds]

    firm_data = add_number_employees_compustat(
        firm_data, compustat_data, n_employees_per_industry, n_firms_per_industry, n_industries
    )

    labour_compensation = industry_data["industry_vectors"]["Labour Compensation in LCU"].values
    firm_data = add_wages(firm_data, n_employees_per_industry, n_firms, n_industries, labour_compensation, tau_sif)
    output = industry_data["industry_vectors"]["Output in USD"].values
    firm_data = add_production(firm_data, n_employees_per_industry, n_industries, output)
    firm_data["Price in USD"] = 1.0
    firm_data["Price"] = firm_data["Price in USD"] * exchange_rate
    firm_data["Labour Inputs"] = firm_data["Production"].copy()

    if assume_initial_unit:
        firm_data["Labour Productivity"] = 1.0
    else:
        labour_prod_by_industry = output / n_employees_per_industry
        for industry in range(n_industries):
            firm_data.loc[firm_data["Industry"] == industry, "Labour Productivity"] = labour_prod_by_industry[industry]

    firm_data["Deposits"] = compustat_data["Deposits"].values

    return firm_data


def initialise_basic_firm_fields(
    firm_data: pd.DataFrame,
    industry_data: dict[str, pd.DataFrame],
    n_employees_per_industry: np.ndarray | list,
    n_firms_per_industry: NDArrayInt | list[int],
    firm_size_zetas: dict[int, float],
    exchange_rate: float,
    tau_sif: float,
    assume_initial_unit: bool = False,
):
    """Preprocess basic firm data using industry statistics.

    This function initializes fundamental firm-level data using industry-level
    statistics. The preprocessing includes:
    1. Industry assignment
    2. Employee allocation using power-law distributions
    3. Wage computation from labor compensation
    4. Production allocation from industry output
    5. Price initialization and currency conversion

    Note:
        This is a preprocessing function. The actual firm behavior is
        implemented in the simulation package.

    Args:
        firm_data (pd.DataFrame): Firm data container to update
        industry_data (dict[str, pd.DataFrame]): Industry-level statistics
        n_employees_per_industry (np.ndarray | list): Industry employment totals
        n_firms_per_industry (NDArrayInt | list[int]): Industry firm counts
        firm_size_zetas (dict[int, float]): Power-law parameters
        exchange_rate (float): USD to local currency rate
        tau_sif (float): Employer social insurance tax rate
        assume_initial_unit (bool): Whether to use unit labor productivity

    Returns:
        pd.DataFrame: Preprocessed firm data
    """
    n_industries = len(n_employees_per_industry)
    n_firms = sum(n_firms_per_industry)
    firm_data["Industry"] = np.array(
        reduce(
            lambda a, b: a + b,
            ([industry] * s for industry, s in enumerate(n_firms_per_industry)),
        )
    )

    firm_data = add_number_employees_random(
        firm_data, firm_size_zetas, n_employees_per_industry, n_firms_per_industry, n_industries
    )
    labour_compensation = industry_data["industry_vectors"]["Labour Compensation in LCU"].values
    firm_data = add_wages(firm_data, n_employees_per_industry, n_firms, n_industries, labour_compensation, tau_sif)
    output = industry_data["industry_vectors"]["Output in USD"].values
    firm_data = add_production(firm_data, n_employees_per_industry, n_industries, output)
    firm_data["Price in USD"] = 1.0
    firm_data["Price"] = firm_data["Price in USD"] * exchange_rate
    firm_data["Labour Inputs"] = firm_data["Production"].copy()

    if assume_initial_unit:
        firm_data["Labour Productivity"] = 1.0
    else:
        labour_prod_by_industry = output / n_employees_per_industry
        for industry in range(n_industries):
            firm_data.loc[firm_data["Industry"] == industry, "Labour Productivity"] = labour_prod_by_industry[industry]

    return firm_data


def function_parameters_dependent_initialisation(
    firm_data: pd.DataFrame,
    intermediate_inputs_productivity_matrix: np.ndarray,
    capital_inputs_depreciation_matrix: np.ndarray,
    capital_inputs_productivity_matrix: np.ndarray,
    total_firm_deposits: float,
    total_firm_debt: float,
    assume_zero_initial_debt: bool,
    assume_zero_initial_deposits: bool,
    capital_inputs_utilisation_rate: float,
    initial_inventory_to_input_fraction: float,
    intermediate_inputs_utilisation_rate: float,
):
    """Preprocess firm data based on functional parameters.

    This function initializes firm-level data that depends on various functional
    parameters. The preprocessing includes:
    1. Inventory level initialization
    2. Input stock calculations
    3. Capital stock computations
    4. Financial position setup

    Note:
        This is a preprocessing function. The actual firm behavior and parameter
        evolution are implemented in the simulation package.

    Args:
        firm_data (pd.DataFrame): Firm data container to update
        intermediate_inputs_productivity_matrix (np.ndarray): Input productivity
        capital_inputs_depreciation_matrix (np.ndarray): Capital depreciation
        capital_inputs_productivity_matrix (np.ndarray): Capital productivity
        total_firm_deposits (float): Aggregate firm deposits
        total_firm_debt (float): Aggregate firm debt
        assume_zero_initial_debt (bool): Whether to start with zero debt
        assume_zero_initial_deposits (bool): Whether to start with zero deposits
        capital_inputs_utilisation_rate (float): Initial capital utilization
        initial_inventory_to_input_fraction (float): Initial inventory ratio
        intermediate_inputs_utilisation_rate (float): Initial input utilization

    Returns:
        tuple: Preprocessed capital stock, input stock, and utilization data
    """
    # This needs to be moved to the macromodel package
    # note that firm_data, intermediate_inputs_productivity_matrix, capital_inputs_depreciation_matrix,
    # firm deposits, and firm debt will be attributes of the synthetic firm class
    firm_data["Inventory"] = initial_inventory_to_input_fraction * firm_data["Production"]
    firm_data["Demand"] = firm_data["Production"] + firm_data["Inventory"]  # I am not sure about this
    intermediate_inputs_stock = (
        1.0
        / intermediate_inputs_utilisation_rate
        * (firm_data["Production"].values / intermediate_inputs_productivity_matrix[:, firm_data["Industry"].values]).T
    )
    used_intermediate_inputs = (
        firm_data["Production"].values / intermediate_inputs_productivity_matrix[:, firm_data["Industry"].values]
    ).T.astype(float)
    capital_inputs_stock = (
        1.0
        / capital_inputs_utilisation_rate
        * (firm_data["Production"].values / capital_inputs_productivity_matrix[:, firm_data["Industry"].values]).T
    )
    used_capital_inputs = (
        firm_data["Production"].values * capital_inputs_depreciation_matrix[:, firm_data["Industry"].values]
    ).T.astype(float)
    # TODO : Sam's version with compustat data can have firms with negative deposits, and this doesn't work

    deposits_in_data = "Deposits" in firm_data.columns

    if assume_zero_initial_deposits:
        firm_data["Deposits"] = 0.0
    else:
        if not deposits_in_data or firm_data["Deposits"].values.sum() == 0.0:
            firm_data["Deposits"] = np.full(
                len(firm_data),
                1.0 / len(firm_data) * total_firm_deposits,
            )
        else:
            if deposits_in_data:
                firm_data["Deposits"] = np.clip(firm_data["Deposits"], 0, None)
                weights = firm_data["Deposits"].values / firm_data["Deposits"].values.sum()
            else:
                weights = firm_data["Production"].values / firm_data["Production"].values.sum()
            firm_data["Deposits"] = weights * total_firm_deposits
    firm_data["Deposits"] = np.maximum(0.0, firm_data["Deposits"])
    if assume_zero_initial_debt:
        firm_data["Debt"] = 0.0
    else:
        firm_data["Debt"] = capital_inputs_stock.sum(axis=1) / capital_inputs_stock.sum() * total_firm_debt
    firm_data["Equity"] = (
        firm_data["Deposits"]
        + firm_data["Price"] * firm_data["Inventory"]
        + firm_data["Price"] * intermediate_inputs_stock.sum(axis=1)
        + firm_data["Price"] * capital_inputs_stock.sum(axis=1)
        - firm_data["Debt"]
    )
    return capital_inputs_stock, intermediate_inputs_stock, used_capital_inputs, used_intermediate_inputs
