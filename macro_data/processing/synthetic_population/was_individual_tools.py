import logging

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.default_readers import DataReaders
from macro_data.util.clean_data import remove_outliers
from macro_data.util.imputation import apply_iterative_imputer


def process_individual_data(
    individual_data: pd.DataFrame,
    industries: list[str],
    scale: int,
    total_unemployment_benefits: float,
    unemployment_rate: float,
    participation_rate: float,
    n_firms_by_industry: list[int] | np.ndarray,
) -> pd.DataFrame:
    """
    Process individual data by performing various data cleaning and transformation steps.

    Args:
        individual_data (pd.DataFrame): The individual data to be processed.
        industries (list[str]): The list of industries.
        scale (int): The scale factor.
        total_unemployment_benefits (float): The total unemployment benefits.
        unemployment_rate (float): The unemployment rate.
        participation_rate (float): The participation rate.
        n_firms_by_industry (list[int] | np.ndarray): The number of firms by industry.

    Returns:
        pd.DataFrame: The processed individual data.
    """

    n_industries = len(industries)

    # Temporarily set age
    no_age = individual_data["Age"].isna()
    individual_data.loc[no_age, "Age"] = np.random.choice(range(0, 100), no_age.sum(), replace=True)

    # Only remove outliers if we have enough data (more than 10 rows)
    # For small test datasets, skip outlier removal to avoid filtering out all data
    if len(individual_data) > 10:
        individual_data = remove_outliers(
            data=individual_data,
            cols=["Employee Income", "Gender", "Age", "Education", "Labour Status"],
            use_logpdf=False,
        )
    individual_data = fill_missing_gender(individual_data)
    individual_data = fill_individual_age(individual_data)
    individual_data = fill_individual_education(individual_data)
    individual_data = fill_individual_labour_status(individual_data)
    individual_data = set_individual_activity_status(
        individual_data=individual_data,
        unemployment_rate=unemployment_rate,
        participation_rate=participation_rate,
    )
    individual_data = fill_individual_nace(individual_data, industries, n_firms_by_industry, n_industries)
    n_unemployed = np.sum(individual_data["Activity Status"] == 2)

    # DANGER: if we don't have total unemployment benefits
    # we set them to 0; must be checked or filled in another way
    if total_unemployment_benefits is None:
        total_unemployment_benefits = 0.0
        logging.warning("Total unemployment benefits not found, setting to 0.0")

    individual_data = fill_individual_employee_income(
        individual_data, unemployment_benefits_by_individual=total_unemployment_benefits / n_unemployed, scale=scale
    )
    individual_data = set_individual_unemployed_income(
        individual_data, unemployment_benefits_by_individual=total_unemployment_benefits / n_unemployed
    )
    individual_data["Income"] = (
        individual_data["Employee Income"].fillna(0.0).values
        + individual_data["Income from Unemployment Benefits"].fillna(0.0).values
    )
    
    # Create Corresponding Household ID from HID column
    individual_data["Corresponding Household ID"] = individual_data["HID"]
    
    individual_data = individual_data[
        [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
            "Corresponding Household ID",
        ]
    ]

    individual_data.loc[:, "Corresponding Invested Firm"] = np.nan
    individual_data.loc[:, "Corresponding Invested Bank"] = np.nan

    return individual_data


def fill_missing_gender(individual_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing gender values in the individual_data DataFrame using a probabilistic approach.

    Parameters:
        individual_data (pd.DataFrame): DataFrame containing individual data.

    Returns:
        pd.DataFrame: DataFrame with missing gender values filled.

    """
    missing_genders = individual_data["Gender"].isna()
    p_male = (individual_data["Gender"] == 1).mean()
    p_female = (individual_data["Gender"] == 2).mean()
    p_total = p_male + p_female
    p_male /= p_total
    p_female /= p_total
    individual_data.loc[missing_genders, "Gender"] = np.random.choice(
        [1, 2],
        missing_genders.sum(),
        p=[
            p_male,
            p_female,
        ],
        replace=True,
    )
    return individual_data


def fill_individual_age(individual_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the 'Age' column of the individual_data DataFrame.

    If the individual is a student, the missing values are filled within the range of 6 to 18.
    If the individual is not a student, the missing values are filled with a minimum value of 0.

    Parameters:
        individual_data (pd.DataFrame): The DataFrame containing individual data.

    Returns:
        pd.DataFrame: The DataFrame with missing values in the 'Age' column filled.
    """
    is_student = individual_data["Labour Status"] == 4
    # Only apply imputation for students if there are any students
    if is_student.any():
        individual_data = apply_iterative_imputer(
            individual_data, columns=["Gender", "Age"], selection=is_student, min_value=6, max_value=18
        )
    individual_data = apply_iterative_imputer(individual_data, columns=["Gender", "Age"], min_value=0)
    return individual_data


def fill_individual_education(individual_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the 'Education' column of the individual_data DataFrame using iterative imputation.

    Parameters:
        individual_data (pd.DataFrame): The DataFrame containing individual data.

    Returns:
        pd.DataFrame: The DataFrame with missing values in the 'Education' column imputed.
    """
    individual_data = apply_iterative_imputer(individual_data, columns=["Gender", "Age", "Education"])
    individual_data["Education"] = individual_data["Education"].astype(int)
    return individual_data


def fill_individual_labour_status(individual_data: pd.DataFrame) -> pd.DataFrame:
    individual_data.loc[individual_data["Age"] < 16, "Labour Status"] = 4  # student
    individual_data.loc[individual_data["Labour Status"].isna()] = 2  # unemployed
    return individual_data


def convert_labour_status_to_activity_status(ls: int) -> int:
    individual_activity_status_map = {
        1: 1,  # REGULAR_WORK -> EMPLOYED
        2: 2,  # ON_LEAVE -> UNEMPLOYED
        3: 2,  # UNEMPLOYED -> UNEMPLOYED
        4: 3,  # STUDENT -> NOT_ECONOMICALLY_ACTIVE
        5: 3,  # RETIREE -> NOT_ECONOMICALLY_ACTIVE
        6: 3,  # DISABLED -> NOT_ECONOMICALLY_ACTIVE
        7: 1,  # MILITARY_SOCIAL_SERVICE -> EMPLOYED
        8: 2,  # DOMESTIC_TASKS -> UNEMPLOYED
        9: 2,  # OTHER_NOT_FOR_PAY -> UNEMPLOYED
        # WAS-specific negative values (missing/special categories)
        -6: 2,  # Missing/Unknown -> UNEMPLOYED
        -8: 2,  # Missing/Unknown -> UNEMPLOYED  
        -9: 2,  # Missing/Unknown -> UNEMPLOYED
    }
    return individual_activity_status_map[ls]


def set_individual_activity_status(
    individual_data: pd.DataFrame,
    unemployment_rate: float,
    participation_rate: float,
) -> pd.DataFrame:
    """
    Sets the activity status of individuals in the given DataFrame based on the provided unemployment rate
    and participation rate.

    Parameters:
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        unemployment_rate (float): The desired unemployment rate.
        participation_rate (float): The desired participation rate.

    Returns:
        pd.DataFrame: The updated DataFrame with activity status set for each individual.
    """
    # Turn the labour status into an activity status
    individual_data["Activity Status"] = individual_data["Labour Status"].map(
        convert_labour_status_to_activity_status,
    )

    # Adjust according to the participation rate
    current_participation_rate = np.sum(
        np.logical_and(
            individual_data["Activity Status"] != 3,
            individual_data["Age"] >= 16,
        )
    ) / np.sum(individual_data["Age"] >= 16)

    if participation_rate >= current_participation_rate:
        increase_participation_rate(individual_data, participation_rate)
    else:
        decrease_participation_rate(individual_data, participation_rate)

    # Adjust according to the unemployment rate
    n_unemployed = np.sum(individual_data["Activity Status"] == 2)
    n_employed = np.sum(individual_data["Activity Status"] == 1)
    current_unemployment_rate = n_unemployed / (n_unemployed + n_employed)

    if unemployment_rate >= current_unemployment_rate:
        increase_unemployment_rate(individual_data, n_employed, n_unemployed, unemployment_rate)
    else:
        decrease_unemployment_rate(individual_data, n_employed, n_unemployed, unemployment_rate)

    return individual_data


def decrease_unemployment_rate(
    individual_data: pd.DataFrame, n_employed: int, n_unemployed: int, unemployment_rate: float
) -> None:
    """
    Decreases the unemployment rate by transitioning individuals from unemployed to employed.

    Args:
        individual_data (pd.DataFrame): DataFrame containing individual data.
        n_employed (int): Number of currently employed individuals.
        n_unemployed (int): Number of currently unemployed individuals.
        unemployment_rate (float): Current unemployment rate.

    Returns:
        None
    """
    n_additional_employed = int(n_unemployed - unemployment_rate * (n_employed + n_unemployed))
    rnd_ind = np.random.choice(
        np.flatnonzero(
            individual_data["Activity Status"] == 2,
        ),
        n_additional_employed,
        replace=False,
    )
    individual_data.loc[rnd_ind, "Activity Status"] = 1
    individual_data.loc[
        rnd_ind,
        [
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
        ],
    ] = np.nan


def increase_unemployment_rate(
    individual_data: pd.DataFrame, n_employed: int, n_unemployed: int, unemployment_rate: float
) -> None:
    """
    Increase the unemployment rate in the individual data by updating the activity status and income information.

    Parameters:
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        n_employed (int): The number of individuals currently employed.
        n_unemployed (int): The number of individuals currently unemployed.
        unemployment_rate (float): The desired unemployment rate.

    Returns:
        None
    """
    n_additional_unemployed = int(unemployment_rate * (n_employed + n_unemployed) - n_unemployed)
    rnd_ind = np.random.choice(
        np.flatnonzero(individual_data["Activity Status"] == 1),
        n_additional_unemployed,
        replace=False,
    )
    individual_data.loc[rnd_ind, "Activity Status"] = 2
    individual_data.loc[
        rnd_ind,
        [
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
        ],
    ] = np.nan


def decrease_participation_rate(individual_data: pd.DataFrame, participation_rate: float) -> None:
    """
    Decreases the participation rate of economically active individuals in the given DataFrame.

    Args:
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        participation_rate (float): The desired participation rate.

    Returns:
        None
    """
    economically_active = np.logical_and(
        individual_data["Activity Status"] != 3,
        individual_data["Age"] >= 16,
    )

    n_additional_nea = int(economically_active.sum() - participation_rate * np.sum(individual_data["Age"] >= 16))
    unemployed = np.logical_and(
        individual_data["Activity Status"] == 2,
        individual_data["Age"] >= 16,
    )

    rnd_ind = np.random.choice(
        np.flatnonzero(unemployed),
        min(n_additional_nea, np.count_nonzero(unemployed)),
        replace=False,
    )
    individual_data.loc[rnd_ind, "Activity Status"] = 3
    individual_data.loc[
        rnd_ind,
        [
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
        ],
    ] = np.nan


def increase_participation_rate(individual_data: pd.DataFrame, participation_rate: float) -> None:
    """
    Increase the participation rate in the individual data by modifying the activity status of individuals.

    Args:
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        participation_rate (float): The desired participation rate.

    Returns:
        None
    """
    economically_active = np.logical_and(
        individual_data["Activity Status"] != 3,
        individual_data["Age"] >= 16,
    )
    n_additional_unemployed = int(participation_rate * np.sum(individual_data["Age"] >= 16) - economically_active.sum())
    not_economically_active = np.logical_and(
        individual_data["Activity Status"] == 3,
        individual_data["Age"] >= 16,
    )
    rnd_ind = np.random.choice(
        np.flatnonzero(not_economically_active),
        np.min([n_additional_unemployed, np.count_nonzero(not_economically_active)]),
        replace=False,
    )
    individual_data.loc[rnd_ind, "Activity Status"] = 2
    individual_data.loc[
        rnd_ind,
        [
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
        ],
    ] = np.nan


def fill_individual_nace(
    individual_data: pd.DataFrame,
    industries: list[str],
    n_firms_by_industry: list[int] | np.ndarray,
    n_industries: int = 43,
) -> pd.DataFrame:
    """
    Fill in missing values in the 'Employment Industry' column of the individual_data DataFrame
    based on the provided industries list.

    Args:
        individual_data (pd.DataFrame): DataFrame containing individual data.
        industries (list[str]): List of industries.
        n_firms_by_industry (list[int] | np.ndarray): Number of firms by industry.

    Returns:
        pd.DataFrame: DataFrame with missing values in the 'Employment Industry' column filled.

    """
    # For WAS data, we need to map SIC codes to industry categories
    # First, handle missing values (-9) and convert to NaN
    individual_data.loc[
        individual_data["Employment Industry"] == -9,
        "Employment Industry",
    ] = np.nan
    
    # For WAS data, we'll assign industries randomly based on the available industries
    # since we don't have a direct mapping from SIC codes to the model's industry categories
    individual_data.loc[
        individual_data["Employment Industry"].isin(["R", "S"]),
        "Employment Industry",
    ] = "R_S"

    # Convert to numbers
    industry_map = dict(zip(industries, range(len(industries))))
    individual_data["Employment Industry"] = individual_data["Employment Industry"].map(industry_map)

    # Clean other missing value codes
    individual_data.loc[
        individual_data["Employment Industry"] == "-1",
        "Employment Industry",
    ] = np.nan
    individual_data.loc[
        individual_data["Employment Industry"] == "-2",
        "Employment Industry",
    ] = np.nan

    # Assumption
    individual_data.loc[individual_data["Activity Status"] == 3, "Employment Industry"] = np.nan

    # Count employees by sector - ensure we get a Series, not a DataFrame
    n_employees_by_sector_series = individual_data.groupby("Employment Industry")["Activity Status"].apply(
        lambda x: (x == 1).sum()
    )
    # Use the actual number of industries, not the default n_industries
    actual_n_industries = len(industries)
    n_employees_by_sector_series = n_employees_by_sector_series.reindex(range(actual_n_industries)).fillna(0).astype("int")

    n_employees_by_sector = n_employees_by_sector_series.values

    for ind in range(len(industries)):
        individual_data, n_employees_by_sector = reassign_industry(
            ind,
            n_firms_by_industry,
            n_employees_by_sector,
            individual_data,
        )

    # frequencies = np.array(
    #     [
    #         np.sum(
    #             np.logical_and(
    #                 individual_data["Activity Status"] < 3,
    #                 individual_data["Employment Industry"] == ind,
    #             )
    #         )
    #         for ind in range(len(industries))
    #     ]
    # ).astype(float)
    # frequencies /= np.sum(frequencies)
    frequencies = (
        individual_data.groupby("Employment Industry").apply(lambda x: (x["Activity Status"] < 3).sum()).values
    )
    frequencies = frequencies / np.sum(frequencies)

    # fill in missing sector of unemployed people
    mask = np.logical_and(individual_data["Activity Status"] < 3, pd.isnull(individual_data["Employment Industry"]))
    individuals_without_industry = np.flatnonzero(mask)

    new_industries = np.random.choice(range(len(industries)), len(individuals_without_industry), replace=True)

    individual_data.loc[individuals_without_industry, "Employment Industry"] = new_industries

    # # Fill-in missing sectors
    # sectors_missing = np.where(frequencies == 0)[0]
    # individuals_missing_industry = np.where(
    #     np.logical_and(
    #         individual_data["Activity Status"] < 3,
    #         pd.isnull(individual_data["Employment Industry"]).values,
    #     )
    # )[0]
    # individual_data.loc[individuals_missing_industry, "Employment Industry"] = np.random.choice(
    #     sectors_missing,
    #     len(individuals_missing_industry),
    #     replace=True,
    # )
    #
    # individual_data.loc[individual_data["Activity Status"] == 3, "Employment Industry"] = np.nan
    return individual_data


def reassign_industry(
    ind: int,
    n_firms_by_industry: list[int] | np.ndarray,
    n_employees_by_industry: list[int] | np.ndarray,
    individual_data: pd.DataFrame,
):
    # we need to have at least one employee per firm
    # so we first compute the number of extra employees needed that we need to reassign
    n_extra = n_firms_by_industry[ind] - n_employees_by_industry[ind]
    if n_extra > 0:
        # boolean mask for individual data with Activity Status == 1 and Employment Industry nan
        mask = np.logical_and(
            individual_data["Activity Status"] == 1,
            pd.isnull(individual_data["Employment Industry"]),
        )
        # get the indices of the individuals that satisfy the mask
        individuals_without_industry = individual_data.index[mask]

        # if we have less individuals that can be reassigned than the number of extra employees needed
        # we will reassign them
        if len(individuals_without_industry) < n_extra:
            # the number of individuals that must be reassigned
            reset_ind = n_extra - len(individuals_without_industry)

            # iterate over industry and identify employees within industry that can be reassigned
            individuals_having_industry = []
            for i in range(len(n_employees_by_industry)):
                list_employed_in_industry = select_employed_in_industry(individual_data, i)
                n_select = len(list_employed_in_industry) - n_firms_by_industry[i]
                # Only select if we have individuals and need to select some
                if len(list_employed_in_industry) > 0 and n_select > 0:
                    individuals_having_industry += list(
                        np.random.choice(list_employed_in_industry, n_select, replace=False)
                    )

            individuals_having_industry = np.array(individuals_having_industry)

            # then we select randomly from this index a number reset_ind of them
            # and set their industry to nan
            if len(individuals_having_industry) > 0:
                # Make sure we don't try to select more than available
                n_to_select = min(reset_ind, len(individuals_having_industry))
                individuals_to_reassign = np.random.choice(individuals_having_industry, n_to_select, replace=False)
                individual_data.loc[individuals_to_reassign, "Employment Industry"] = np.nan

            # now get the individuals without industry again
            mask = np.logical_and(
                individual_data["Activity Status"] == 1,
                pd.isnull(individual_data["Employment Industry"]),
            )
            individuals_without_industry = individual_data.index[mask]

            # if we still have individuals to reassign, pick n_extra of them and assign them to the industry
            if len(individuals_without_industry) > 0:
                # Make sure we don't try to select more than available
                n_to_select = min(n_extra, len(individuals_without_industry))
                individuals_to_reassign = np.random.choice(individuals_without_industry, n_to_select, replace=False)
                individual_data.loc[individuals_to_reassign, "Employment Industry"] = ind

            # update the number of employees by industry
            n_employees_by_industry_series = individual_data.groupby("Employment Industry").apply(lambda x: (x["Activity Status"] == 1).sum())
            # Ensure we have the correct size by reindexing to include all industries
            n_employees_by_industry = n_employees_by_industry_series.reindex(range(len(n_firms_by_industry))).fillna(0).astype(int).values

    return individual_data, n_employees_by_industry


def select_employed_in_industry(individual_data: pd.DataFrame, industry: int) -> np.ndarray:
    """
    Selects the employed individuals in the given industry from the individual_data DataFrame.

    Parameters:
        individual_data (pd.DataFrame): DataFrame containing individual data.
        industry (int): The industry to select employed individuals from.

    Returns:
        np.ndarray: The indices of employed individuals in the given industry.
    """
    mask = np.logical_and(
        individual_data["Activity Status"] == 1,
        individual_data["Employment Industry"] == industry,
    )
    return individual_data.index[mask].values


def fill_individual_employee_income(
    individual_data: pd.DataFrame, unemployment_benefits_by_individual: float, scale: int
) -> pd.DataFrame:
    """
    Fills the 'Employee Income' column in the individual_data DataFrame for employed individuals.
    The function applies iterative imputation to estimate missing values based on 'Age' and 'Education'.
    It also performs several adjustments and rescaling of the 'Employee Income' values.

    Parameters:
        individual_data (pd.DataFrame): DataFrame containing individual data.
        unemployment_benefits_by_individual (float): Unemployment benefits received by each individual.
        scale (int): Scaling factor for the 'Employee Income' values.

    Returns:
        pd.DataFrame: DataFrame with the 'Employee Income' column filled and adjusted.
    """
    is_employed = individual_data["Activity Status"] == 1

    # We're not explicitly modelling this
    individual_data["Employee Income"] += individual_data["Self-Employment Income Total"].fillna(0.0).values
    individual_data.loc[
        individual_data["Employee Income"] < 0,
        "Employee Income",
    ] = 0.0
    no_income = individual_data["Employee Income"] == 0.0
    individual_data.loc[is_employed & no_income, "Employee Income"] = np.nan
    individual_data = apply_iterative_imputer(
        individual_data, columns=["Employee Income", "Age", "Education"], selection=is_employed, min_value=0
    )

    # Only employed individuals receive employee income
    individual_data.loc[individual_data["Activity Status"] != 1, "Employee Income"] = 0.0

    # Rescale that
    individual_data.loc[:, "Employee Income"] *= scale

    # Monthly!
    individual_data.loc[:, "Employee Income"] /= 4.0

    # Employee income is at least the unemployment rate
    is_employed = individual_data["Activity Status"] == 1
    individual_data.loc[
        np.logical_and(
            is_employed,
            individual_data["Employee Income"] < unemployment_benefits_by_individual,
        ),
        "Employee Income",
    ] = unemployment_benefits_by_individual
    return individual_data


def set_individual_unemployed_income(
    individual_data: pd.DataFrame,
    unemployment_benefits_by_individual: float,
) -> pd.DataFrame:
    """
    Sets the income from unemployment benefits for each individual in the given DataFrame.

    Parameters:
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        unemployment_benefits_by_individual (float): The amount of unemployment benefits received by each individual.

    Returns:
        pd.DataFrame: The updated DataFrame with the income from unemployment benefits added.
    """
    is_unemployed = individual_data["Activity Status"] == 2
    not_unemployed = individual_data["Activity Status"] != 2

    individual_data["Income from Unemployment Benefits"] = 0.0
    individual_data.loc[is_unemployed, "Income from Unemployment Benefits"] = unemployment_benefits_by_individual

    # Only unemployed individuals receive unemployment income
    individual_data.loc[not_unemployed, "Income from Unemployment Benefits"] = 0.0
    return individual_data
