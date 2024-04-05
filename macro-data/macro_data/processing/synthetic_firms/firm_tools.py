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
    """
    Draw firm sizes for an industry based on the number of employees. The size distribution within each industry is
    assumed to be a discrete power-law distribution, with a normalisation constant that is the Riemann zeta function.

    In other words, the probability of a firm having n employees is
    ..math::
        p(n) = \frac{1}{n^{\zeta(s)} \zeta(s)}

    where the exponent is s.

    Args:
        n_firms_in_industry (int): The number of firms in the industry.
        number_employees (int): The total number of employees in the industry.
        firm_size_zeta_shape (float): The shape parameter of the zeta distribution for firm sizes.

    Returns:
        np.ndarray: An array of firm sizes, where each element represents the number of employees in a firm.

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


def add_number_employees(
    firm_data: pd.DataFrame,
    firm_size_zetas: np.ndarray | list | dict[int, float],
    n_employees_per_industry: np.ndarray | list,
    n_firms_per_industry: np.ndarray | list,
    n_industries: int,
):
    """
    Adds the number of employees to the firm_data DataFrame based on the given firm size and industry data.
    This first computes a priori firm sizes from power-law distributions with parameters given by the firm size zetas, then
    distributes the remainder of employees (unmatched employees) among firms in each industry.

    Finally, the "Number of Employees" column in the firm_data DataFrame is updated with the computed firm sizes.

    Args:
        firm_data (pd.DataFrame): The DataFrame containing firm data.
        firm_size_zetas (np.ndarray | list | dict[int, float]): The firm size zetas for each industry.
        n_employees_per_industry (np.ndarray | list): The number of employees per industry.
        n_firms_per_industry (np.ndarray | list): The number of firms per industry.
        n_industries (int): The total number of industries.

    Returns:
        pd.DataFrame: The firm_data DataFrame with the "Number of Employees" column updated.
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
    """
    Add wages information to the firm_data DataFrame based on the given parameters.
    Wages are computed as the total labour compensation for each industry divided by the number of employees in that industry.

    Wages paid by firms are computed according to their total number of employees, and include the tax rate tau_sif.

    Wages received by employees do not include the employer tax rate tau_sif, but are taxed later.

    Parameters:
        firm_data (pd.DataFrame): The DataFrame containing firm data.
        n_employees_per_industry (list | np.ndarray): The number of employees per industry.
        n_firms (int): The total number of firms.
        n_industries (int): The total number of industries.
        labour_compensation (np.ndarray): The compensation for each industry.
        tau_sif (float): The tax rate.

    Returns:
        pd.DataFrame: The updated firm_data DataFrame with added wage information.
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
    firm_data: pd.DataFrame, n_employees_per_industry: list | np.ndarray, n_industries: int, output: np.ndarray
) -> pd.DataFrame:
    """
    Allocate production values to firms based on the total industry output and proportionally to the number of employees.

    Production is in real terms, so we use the production value in USD and set the price to 1USD.

    Parameters:
        firm_data (pd.DataFrame): The DataFrame containing firm data.
        n_employees_per_industry (list | np.ndarray): The number of employees per industry.
        n_industries (int): The number of industries.
        output (np.ndarray): The industry output values.

    Returns:
        pd.DataFrame: The updated firm_data DataFrame with production values added.
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


def initialise_basic_firm_fields(
    firm_data: pd.DataFrame,
    industry_data: dict[str, pd.DataFrame],
    n_employees_per_industry: np.ndarray | list,
    n_firms_per_industry: NDArrayInt | list[int],
    firm_size_zetas: dict[int, float],
    exchange_rate: float,
    tau_sif: float,
):
    """
    Initializes basic fields for each firm in the firm_data DataFrame.
    First, firms are assigned to industries based on the number of firms per industry. Then, the number of employees of each firm
    is computed based on the number of employees per industry and the firm size zetas. Wages are computed based on the number of
    employees and the labour compensation for each industry. Finally, production values are computed based on the number of
    employees and the industry output.

    Prices are initialised to 1 in USD and then converted to the local currency using the exchange rate.

    The corresponding firm data is returned.

    Parameters:
        firm_data (pd.DataFrame): The DataFrame containing firm data.
        industry_data (dict[str, pd.DataFrame]): A dictionary mapping industry names to industry data DataFrames.
        n_employees_per_industry (np.ndarray | list): An array or list containing the number of employees per industry.
        n_firms_per_industry (NDArrayInt | list[int]): An array or list containing the number of firms per industry.
        firm_size_zetas (dict[int, float]): A dictionary mapping firm sizes to zeta values.
        exchange_rate (float): The exchange rate.
        tau_sif (float): The tau_sif value.

    Returns:
        pd.DataFrame: The firm_data DataFrame with the initialized fields.
    """
    n_industries = len(n_employees_per_industry)
    n_firms = sum(n_firms_per_industry)
    firm_data["Industry"] = np.array(
        reduce(
            lambda a, b: a + b,
            ([industry] * s for industry, s in enumerate(n_firms_per_industry)),
        )
    )

    firm_data = add_number_employees(
        firm_data, firm_size_zetas, n_employees_per_industry, n_firms_per_industry, n_industries
    )
    labour_compensation = industry_data["industry_vectors"]["Labour Compensation in LCU"].values
    firm_data = add_wages(firm_data, n_employees_per_industry, n_firms, n_industries, labour_compensation, tau_sif)
    output = industry_data["industry_vectors"]["Output in USD"].values
    firm_data = add_production(firm_data, n_employees_per_industry, n_industries, output)
    firm_data["Price in USD"] = 1.0
    firm_data["Price"] = firm_data["Price in USD"] * exchange_rate
    firm_data["Labour Inputs"] = firm_data["Production"].copy()
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
    """
    Perform parameter-dependent initialization of firm data.

    This depends on parameters that depend on the functions used to run the simulation, as they are used to compute
    the initial values of inventories and input usage.

    Args:
        firm_data (pd.DataFrame): DataFrame containing firm data.
        intermediate_inputs_productivity_matrix (np.ndarray): Matrix of intermediate inputs productivity.
        capital_inputs_depreciation_matrix (np.ndarray): Matrix of capital inputs depreciation.
        capital_inputs_productivity_matrix (np.ndarray): Matrix of capital inputs productivity.
        total_firm_deposits (float): Total firm deposits.
        total_firm_debt (float): Total firm debt.
        assume_zero_initial_debt (bool): Flag indicating whether to assume zero initial debt.
        assume_zero_initial_deposits (bool): Flag indicating whether to assume zero initial deposits.
        capital_inputs_utilisation_rate (float): Capital inputs utilization rate.
        initial_inventory_to_input_fraction (float): Fraction of initial inventory to input.
        intermediate_inputs_utilisation_rate (float): Intermediate inputs utilization rate.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing capital inputs stock,
        intermediate inputs stock, used capital inputs, and used intermediate inputs.
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
    if assume_zero_initial_deposits:
        firm_data["Deposits"] = 0.0
    else:
        firm_data["Deposits"] = firm_data["Production"] / firm_data["Production"].sum() * total_firm_deposits
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
