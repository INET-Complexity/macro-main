import numpy as np
import pandas as pd

from macro_data.util.imputation import apply_iterative_imputer


def get_household_type(ages: np.ndarray) -> int:
    """
    Determines the household type based on the set of ages in the household.
    Age types are defined in the HFCS documentation as follows:
    - 51 for one adult younger than 65
    - 52 for one adult older than 65
    - 6 for two adults younger than 65
    - 7 for two adults, one at least 65
    - 8 for three or more adults
    - 9 for single parent with children
    - 10 for two adults with one child
    - 11 for two adults with two children
    - 12 for two adults with at least three children
    - 13 for three or more adults with children

    Args:
        ages (np.ndarray): An array of ages representing the individuals in the household.

    Returns:
        int: The household type code.

    Raises:
        ValueError: If there are children living together.

    """
    ages = np.sort(ages)
    match len(ages):
        case 1:
            if ages[0] < 64:
                return 51  # ONE_ADULT_YOUNGER_THAN_64
            else:
                return 52  # ONE_ADULT_OLDER_THAN_65
        case 2:
            if 18 <= ages[0] < 64 and 18 <= ages[1] < 64:
                return 6  # TWO_ADULTS_YOUNGER_THAN_65
            elif (18 <= ages[0] < 64 < ages[1]) or (ages[0] > 64 > ages[1] >= 18):
                return 7  # TWO_ADULTS_ONE_AT_LEAST_65
            elif ages[0] < 18 and ages[1] < 18:
                raise ValueError("Children living together?")
            else:
                return 9  # SINGLE_PARENT_WITH_CHILDREN
        case 3:
            if ages[0] < 18 <= ages[1] and ages[2] >= 18:
                return 10  # TWO_ADULTS_WITH_ONE_CHILD
            elif np.all(ages) >= 18:
                return 8  # THREE_OR_MORE_ADULTS
            else:
                return 9  # SINGLE_PARENT_WITH_CHILDREN
        case _:
            if len(ages) == 4:
                if ages[0] < 18 <= ages[2] and ages[1] < 18 <= ages[3]:
                    return 11  # TWO_ADULTS_WITH_TWO_CHILDREN
            else:
                if ages[-1] >= 18 and ages[-2] >= 18 and np.all(ages[0:-2] < 18):
                    return 12  # TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN

            return 13  # THREE_OR_MORE_ADULTS_WITH_CHILDREN


def set_household_types(household_data: pd.DataFrame, individual_data: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the household types for the missing values in the 'Type' column of the household_data DataFrame.

    Parameters:
    - household_data (pd.DataFrame): DataFrame containing household data.
    - individual_data (pd.DataFrame): DataFrame containing individual data.

    Returns:
    - pd.DataFrame: DataFrame with updated 'Type' column.

    """
    i_natypes = np.flatnonzero(household_data["Type"].isna())
    for idx in i_natypes:
        corr_inds = np.array(household_data["Corresponding Individuals ID"][idx])
        household_data.loc[idx, "Type"] = get_household_type(individual_data.loc[corr_inds, "Age"].values)

    return household_data


def set_household_housing_data(
    household_data: pd.DataFrame,
    scale: int,
    rent_as_fraction_of_unemployment_rate: float,
    unemployment_benefits_by_capita: float,
) -> pd.DataFrame:
    """
    Sets the housing data for each household in the given DataFrame.
    It maps the 'Tenure Status of the Main Residence' column to a binary column indicating whether the household owns or
    rents the main residence.
    It also maps the 'Number of Properties other than Household Main Residence' column to a binary column indicating
    whether the household owns additional properties.
    The value of the main residence is scaled by the given scale factor.
    The value of other properties is scaled by the given scale factor.
    Rent and property values are set used the given functions.

    Args:
        household_data (pd.DataFrame): The DataFrame containing the household data.
        scale (int): The scaling factor for the value of other properties.
        rent_as_fraction_of_unemployment_rate (float): The fraction of unemployment benefits used as rent for social housing.
        unemployment_benefits_by_capita (float): The unemployment benefits per capita.

    Returns:
        pd.DataFrame: The updated DataFrame with the housing data set for each household.
    """
    # Whether the household owns or rents
    household_data["Tenure Status of the Main Residence"].replace({2: 1, 4: 1, 3: 0}, inplace=True)
    households_renting = household_data["Tenure Status of the Main Residence"] == 0
    households_owning = household_data["Tenure Status of the Main Residence"] == 1

    # Rent paid and value of the household main residence
    social_housing_rent = rent_as_fraction_of_unemployment_rate * unemployment_benefits_by_capita
    household_data = fill_rent(household_data, households_owning, households_renting, scale, social_housing_rent)

    # Number of additional properties
    household_data["Number of Properties other than Household Main Residence"].fillna(0, inplace=True)
    household_data["Number of Properties other than Household Main Residence"] = household_data[
        "Number of Properties other than Household Main Residence"
    ].astype(int)
    # Value of other properties
    household_data.loc[:, "Value of other Properties"] *= scale
    household_without_additional_properties = (
        household_data["Number of Properties other than Household Main Residence"] == 0
    )
    household_data = fix_property_values(household_data, household_without_additional_properties)

    # Rent received
    household_data = fix_rent(household_data, household_without_additional_properties, scale, social_housing_rent)
    return household_data


def fix_rent(
    household_data: pd.DataFrame,
    household_without_additional_properties: pd.Series,
    scale: int,
    social_housing_rent: float,
) -> pd.DataFrame:
    """
    Adjusts the rental income of households based on specified parameters.

    Args:
        household_data (pd.DataFrame): The dataframe containing household data.
        household_without_additional_properties (pd.Series): A series indicating whether a household has additional properties.
        scale (int): The scaling factor for rental income.
        social_housing_rent (float): The minimum rental income for social housing.

    Returns:
        pd.DataFrame: The updated dataframe with adjusted rental income.
    """
    household_data.loc[:, "Rental Income from Real Estate"] *= scale
    household_data.loc[:, "Rental Income from Real Estate"] /= 12.0
    household_data.loc[
        household_data["Rental Income from Real Estate"] < social_housing_rent,
        "Rental Income from Real Estate",
    ] = social_housing_rent
    household_data.loc[household_without_additional_properties, "Rental Income from Real Estate"] = 0.0
    mask = ~household_without_additional_properties & (household_data["Rental Income from Real Estate"] == 0.0)
    household_data.loc[mask, "Rental Income from Real Estate"] = np.nan
    household_data = apply_iterative_imputer(
        household_data, ["Type", "Value of other Properties", "Rental Income from Real Estate"], min_value=0.0
    )
    return household_data


def fix_property_values(
    household_data: pd.DataFrame, household_without_additional_properties: pd.Series
) -> pd.DataFrame:
    """
    Fixes the property values in the household_data DataFrame based on the household_without_additional_properties Series.

    Parameters:
        household_data (pd.DataFrame): The DataFrame containing household data.
        household_without_additional_properties (pd.Series): The Series indicating whether a household has additional properties.

    Returns:
        pd.DataFrame: The updated household_data DataFrame with fixed property values.
    """
    household_data.loc[household_without_additional_properties, "Value of other Properties"] = 0.0
    mask = ~household_without_additional_properties & (household_data["Value of other Properties"] == 0.0)
    household_data.loc[mask, "Value of other Properties"] = np.nan
    household_data = apply_iterative_imputer(
        household_data,
        columns=[
            "Type",
            "Number of Properties other than Household Main Residence",
            "Value of the Main Residence",
            "Value of other Properties",
        ],
    )
    return household_data


def fill_rent(
    household_data: pd.DataFrame,
    households_owning: pd.Series,
    households_renting: pd.Series,
    scale: int,
    social_housing_rent: float,
) -> pd.DataFrame:
    """
    Fill in missing values for rent paid and value of the main residence in the household data.
    Missing data is filled in using an iterative imputer.

    Parameters:
        household_data (pd.DataFrame): DataFrame containing household data.
        households_owning (pd.Series): Series indicating households that own their main residence.
        households_renting (pd.Series): Series indicating households that rent their main residence.
        scale (int): Scaling factor for rent paid and value of the main residence.
        social_housing_rent (float): Minimum rent for households in social housing.

    Returns:
        pd.DataFrame: Updated household data with filled-in values for rent paid and value of the main residence.
    """
    household_data.loc[:, "Rent Paid"] *= scale
    household_data.loc[:, "Value of the Main Residence"] *= scale
    household_data.loc[households_renting & (household_data["Rent Paid"] == 0.0), "Rent Paid"] = np.nan
    household_data.loc[
        households_owning & (household_data["Value of the Main Residence"] == 0.0), "Value of the Main Residence"
    ] = np.nan
    household_data = apply_iterative_imputer(
        household_data,
        columns=[
            "Type",
            "Rent Paid",
            "Value of the Main Residence",
        ],
    )
    household_data.loc[household_data["Rent Paid"] < social_housing_rent, "Rent Paid"] = social_housing_rent
    household_data.loc[
        households_owning,
        "Rent Paid",
    ] = 0.0
    return household_data
