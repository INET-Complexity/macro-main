import numpy as np
import pandas as pd

from inet_data.readers.default_readers import DataReaders
from inet_data.util.clean_data import remove_outliers
from inet_data.util.imputation import apply_iterative_imputer


def process_individual_data(
    country_name: str,
    individual_data: pd.DataFrame,
    industries: list[str],
    readers: DataReaders,
    scale: int,
    total_unemployment_benefits: float,
    year: int,
) -> pd.DataFrame:
    individual_data = remove_outliers(
        data=individual_data,
        cols=["Employee Income", "Gender", "Age", "Education", "Labour Status"],
    )
    individual_data = fill_missing_gender(individual_data)
    individual_data = fill_individual_age(individual_data)
    individual_data = fill_individual_education(individual_data)
    individual_data = fill_individual_labour_status(individual_data)
    individual_data = set_individual_activity_status(
        individual_data=individual_data,
        unemployment_rate=readers.world_bank.get_unemployment_rate(
            country=country_name,
            year=year,
        ),
        participation_rate=readers.world_bank.get_participation_rate(
            country=country_name,
            year=year,
        ),
    )
    individual_data = fill_individual_nace(individual_data, industries)
    n_unemployed = np.sum(individual_data["Activity Status"] == 2)
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

    return individual_data


def fill_missing_gender(individual_data: pd.DataFrame) -> pd.DataFrame:
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
    is_student = individual_data["Labour Status"] == 4
    individual_data = apply_iterative_imputer(
        individual_data, columns=["Gender", "Age"], selection=is_student, min_value=6, max_value=18
    )
    individual_data = apply_iterative_imputer(individual_data, columns=["Gender", "Age"], min_value=0)
    return individual_data


def fill_individual_education(individual_data: pd.DataFrame) -> pd.DataFrame:
    # Impute education
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
    }
    return individual_activity_status_map[ls]


def set_individual_activity_status(
    individual_data: pd.DataFrame,
    unemployment_rate: float,
    participation_rate: float,
) -> pd.DataFrame:
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


def decrease_unemployment_rate(individual_data, n_employed, n_unemployed, unemployment_rate):
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


def increase_unemployment_rate(individual_data, n_employed, n_unemployed, unemployment_rate):
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


def decrease_participation_rate(individual_data: pd.DataFrame, participation_rate: float):
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


def increase_participation_rate(individual_data: pd.DataFrame, participation_rate: float):
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


def fill_individual_nace(individual_data: pd.DataFrame, industries: list[str]) -> pd.DataFrame:
    individual_data.loc[
        individual_data["Employment Industry"].isin(["R", "S"]),
        "Employment Industry",
    ] = "R_S"

    # Convert to numbers
    industry_map = dict(zip(industries, range(len(industries))))
    individual_data["Employment Industry"] = individual_data["Employment Industry"].map(industry_map)

    # Clean
    individual_data.loc[
        individual_data["Employment Industry"] == "-1",
        "Employment Industry",
    ] = np.nan
    individual_data.loc[
        individual_data["Employment Industry"] == "-2",
        "Employment Industry",
    ] = np.nan

    # Get current frequency of NACE employments
    frequencies = np.array(
        [np.sum(individual_data["Employment Industry"] == ind) for ind in range(len(industries))]
    ).astype(float)
    frequencies /= np.sum(frequencies)

    # Fill-in missing sectors
    sectors_missing = np.where(frequencies == 0)[0]
    individuals_missing_industry = np.where(
        np.logical_and(
            individual_data["Activity Status"] < 3,
            pd.isnull(individual_data["Employment Industry"]).values,
        )
    )[0]
    individual_data.loc[individuals_missing_industry, "Employment Industry"] = np.random.choice(
        sectors_missing,
        len(individuals_missing_industry),
        replace=True,
    )
    # # Distribute proportionally
    # individuals_missing_industry = np.where(
    #     np.logical_and(
    #         individual_data["Activity Status"] < 3,
    #         pd.isnull(individual_data["Employment Industry"]).values,
    #     )
    # )[0]
    # individual_data.loc[individuals_missing_industry, "Employment Industry"] = np.random.choice(
    #     np.arange(len(industries)),
    #     len(individuals_missing_industry),
    #     p=frequencies,
    #     replace=True,
    # )

    # Assumption
    individual_data.loc[individual_data["Activity Status"] == 3, "Employment Industry"] = np.nan
    return individual_data


def fill_individual_employee_income(
    individual_data: pd.DataFrame, unemployment_benefits_by_individual: float, scale: int
) -> pd.DataFrame:
    is_employed = individual_data["Activity Status"] == 1

    # We're not explicitly modelling this
    individual_data["Employee Income"] += individual_data["Self-Employment Income"].fillna(0.0).values
    individual_data.loc[
        individual_data["Employee Income"] < 0,
        "Employee Income",
    ] = 0.0
    no_income = individual_data["Employee Income"] == 0.0
    individual_data.loc[is_employed & no_income, "Employee Income"] = np.nan
    individual_data = apply_iterative_imputer(
        individual_data, columns=["Employee Income", "Age", "Education"], selection=is_employed
    )

    # Only employed individuals receive employee income
    individual_data.loc[individual_data["Activity Status"] != 1, "Employee Income"] = 0.0

    # Rescale that
    individual_data.loc[:, "Employee Income"] *= scale

    # Monthly!
    individual_data.loc[:, "Employee Income"] /= 12.0

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
    is_unemployed = individual_data["Activity Status"] == 2
    not_unemployed = individual_data["Activity Status"] != 2

    individual_data["Income from Unemployment Benefits"] = 0.0
    individual_data.loc[is_unemployed, "Income from Unemployment Benefits"] = unemployment_benefits_by_individual

    # Only unemployed individuals receive unemployment income
    individual_data.loc[not_unemployed, "Income from Unemployment Benefits"] = 0.0
    return individual_data
