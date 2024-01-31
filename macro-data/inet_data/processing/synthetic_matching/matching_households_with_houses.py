import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from inet_data.util.clean_data import remove_outliers
from inet_data.util.imputation import apply_iterative_imputer


def set_housing_df(
    synthetic_population: SyntheticPopulation,
    rental_income_taxes: float,
    social_housing_rent: float,
    total_imputed_rent: float,
) -> pd.DataFrame:
    """
    Set the housing market data. This is done by first handling owner-occupied property, where owners are identified through
    the synthetic population data. The corresponding household data is then updated, to indicate owner-occupied property.

    Next, the rental market data is created. This is done by first identifying the number of additional (avaiable for rent) properties and
    the ids of the landlords (households with additional properties).
    Rental income and property values are also computed, and are returned.

    These two datasets (owner-occupied and rental) are then combined into a single DataFrame, which is processed to remove outliers and to
    fill missing values. Rent below the social housing rent is set to the social housing rent.

    Imputed rent for household owners is set to the corresponding rental value of their home.

    Finally, renters are matched to properties, and the results are recorded.

    Args:
        synthetic_population (SyntheticPopulation): An instance of the SyntheticPopulation class.
        rental_income_taxes (float): The rental income tax rate.
        social_housing_rent (float): The social housing rent.
        total_imputed_rent (float): The total imputed rent.

    Returns:
        pd.DataFrame: A DataFrame containing the housing market data.
    """
    owners_df = create_owners_df(synthetic_population)

    # Updating corresponding household data
    synthetic_population.household_data.loc[
        owners_df["Corresponding Owner Household ID"], "Corresponding Inhabited House ID"
    ] = owners_df["House ID"].values

    synthetic_population.household_data.loc[
        owners_df["Corresponding Owner Household ID"], "Corresponding Property Owner"
    ] = owners_df["Corresponding Owner Household ID"].values

    landlord_ids, num_additional_properties = housing_info_from_population(rental_income_taxes, synthetic_population)

    rental_income = synthetic_population.household_data["Rental Income from Real Estate"].values
    property_values = synthetic_population.household_data["Value of other Properties"].values

    rental_df = create_rental_df(
        num_additional_properties=num_additional_properties,
        landlord_ids=landlord_ids,
        property_values=property_values,
        id_start=int(owners_df["House ID"].max()) + 1,
        rental_income=rental_income,
        rental_income_taxes=rental_income_taxes,
    )
    additionnally_owned_house_ids = np.array_split(rental_df.index, np.cumsum(num_additional_properties))[:-1]

    synthetic_population.household_data["Corresponding Additionally Owned Houses ID"] = additionnally_owned_house_ids

    housing_df = pd.concat([owners_df, rental_df], ignore_index=True)

    invalid_values = housing_df["Value"] < 60 * housing_df["Rent"]
    housing_df.loc[invalid_values, "Value"] = np.nan

    housing_df = remove_outliers(housing_df, cols=["Rent", "Value"], quantile=0.2)

    housing_df = apply_iterative_imputer(housing_df, ["Rent", "Value"])

    rent_under_social_housing = housing_df["Rent"] < social_housing_rent
    housing_df.loc[rent_under_social_housing, "Rent"] = social_housing_rent

    restate_values = np.concatenate(
        [
            [property_values[landlord_id] / num_additional_properties[landlord_id]]
            * num_additional_properties[landlord_id]
            for landlord_id in landlord_ids
        ]
    )
    # rented house values stay the same, but I am not sure why this is needed
    # TODO check with Sam
    housing_df.loc[owners_df.shape[0] :, "Value"] = restate_values

    # owner occupied homes have imputed rents that match the total imputed rent
    owned_houses = housing_df["Is Owner-Occupied"]
    rescale_factor = total_imputed_rent / housing_df.loc[owned_houses, "Rent"].sum()
    housing_df.loc[owned_houses, "Rent"] *= rescale_factor

    owner_indices = synthetic_population.household_data["Tenure Status of the Main Residence"] == 1
    synthetic_population.household_data.loc[owner_indices, "Rent Imputed"] = housing_df.loc[owned_houses, "Rent"].values

    match_renters_to_properties(synthetic_population=synthetic_population, housing_market_df=housing_df)

    return housing_df


def create_owners_df(synthetic_population: SyntheticPopulation) -> pd.DataFrame:
    """
    Create a DataFrame of owners' information from the synthetic population.
    Owners are identified through the synthetic population data, through tenure status == 1.

    House data is created, so that each owner has a corresponding house ID, owner-occupied status, corresponding owner household ID,
    and the inhabitant household ID (which is the same as the owner household ID for owners).
    Values of house data are created by merging the owner household ID with the value of the main residence from the household data.

    Rent is initialised to NaN, but will be filled later using imputed rent.

    Args:
        synthetic_population (SyntheticPopulation): The synthetic population data.

    Returns:
        pd.DataFrame: A DataFrame containing owners' information, including house ID,
                      owner-occupied status, corresponding owner household ID,
                      corresponding inhabitant household ID, value of the main residence, and rent.
    """
    # Handle households owning their house
    households_owning = synthetic_population.household_data["Tenure Status of the Main Residence"] == 1
    owners_df = pd.DataFrame(index=range(households_owning.sum()))
    owners_df["House ID"] = owners_df.index

    owners_df["Is Owner-Occupied"] = True
    owners_df["Corresponding Owner Household ID"] = synthetic_population.household_data.loc[households_owning].index
    owners_df["Corresponding Inhabitant Household ID"] = synthetic_population.household_data.loc[
        households_owning
    ].index

    owners_df = pd.merge(
        owners_df,
        synthetic_population.household_data["Value of the Main Residence"],
        right_index=True,
        left_on="Corresponding Owner Household ID",
    )

    owners_df.rename(columns={"Value of the Main Residence": "Value"}, inplace=True)

    owners_df["Rent"] = np.nan
    return owners_df


def create_rental_df(
    num_additional_properties: np.ndarray,
    landlord_ids: np.ndarray | list,
    property_values: np.ndarray,
    id_start: int,
    rental_income: np.ndarray,
    rental_income_taxes: float,
) -> pd.DataFrame:
    """
    Create a rental DataFrame based on the given parameters.
    A DataFrame is created, with a row for each rental property (of which there are num_additional_properties.sum()).
    The index is set to start at id_start, and the house ID is set to the index + id_start.
    The owner-occupied status is set to False, and the corresponding owner household ID is set to the landlord ID.
    The inhabitant ID is not set, as this will be done through the matching.

    The rent of the house is computed from the rental income of the landlord, and the number of additional properties owned. It is assumed
    that the landlord rents all their additional properties at the same price. The same assumption holds for the value of the property.

    Args:
        num_additional_properties (np.ndarray): Array containing the number of additional properties for each landlord.
        landlord_ids (np.ndarray | list): Array or list of landlord IDs.
        property_values (np.ndarray): Array containing the values of the properties.
        id_start (int): Starting ID for the rental properties.
        rental_income (np.ndarray): Array containing the rental income for each landlord.
        rental_income_taxes (float): Rental income tax rate.

    Returns:
        pd.DataFrame: Rental DataFrame with the following columns:
            - "House ID": ID of the rental property.
            - "Is Owner-Occupied": Boolean indicating if the property is owner-occupied.
            - "Corresponding Owner Household ID": ID of the owner household.
            - "Rent": Rental amount for the property.
            - "Value": Value of the property.
    """
    number_available_properties = num_additional_properties.sum()
    rental_df = pd.DataFrame(index=range(number_available_properties))
    rental_df["House ID"] = rental_df.index + id_start
    rental_df["Is Owner-Occupied"] = False
    ownership_array = np.concatenate(
        [[landlord_id] * num_additional_properties[landlord_id] for landlord_id in landlord_ids]
    )

    rental_df["Corresponding Owner Household ID"] = ownership_array

    rental_df["Rent"] = (
        1.0
        / (1 - rental_income_taxes)
        * np.concatenate(
            [
                [rental_income[landlord_id] / num_additional_properties[landlord_id]]
                * num_additional_properties[landlord_id]
                for landlord_id in landlord_ids
            ]
        )
    )

    rental_df["Value"] = np.concatenate(
        [
            [property_values[landlord_id] / num_additional_properties[landlord_id]]
            * num_additional_properties[landlord_id]
            for landlord_id in landlord_ids
        ]
    )

    return rental_df


def housing_info_from_population(rental_income_taxes: float, synthetic_population: SyntheticPopulation):
    """
    Calculate housing information from the synthetic population data.

    Renting households are identified through the tenure status of the main residence (== 0).
    Rental income is readjusted to match the rent paid minus taxes.
    The number of additional properties is computed, along with the ids of landlords (households with additional properties).

    Args:
        rental_income_taxes (float): The tax rate applied to rental income.
        synthetic_population (SyntheticPopulation): The synthetic population data.

    Returns:
        tuple: A tuple containing the following information:
            - landlord_ids (numpy.ndarray): An array of landlord IDs.
            - num_additional_properties (numpy.ndarray): An array of the number of additional properties owned by each landlord.
            - property_values (numpy.ndarray): An array of the values of the additional properties.
            - rental_income (numpy.ndarray): An array of the rental income from the additional properties.
    """
    num_renters = int(np.sum(synthetic_population.household_data["Tenure Status of the Main Residence"] == 0))
    num_other_properties_owned = int(
        np.sum(synthetic_population.household_data["Number of Properties other than Household Main Residence"])
    )
    if num_renters > num_other_properties_owned:
        set_social_housing_renters(num_other_properties_owned, num_renters, synthetic_population)
    # Rescale total rent received to match rent paid minus taxes
    ind_curr_btl = np.flatnonzero(synthetic_population.household_data["Rental Income from Real Estate"] > 0.0)
    rescale_variable = (
        (1 - rental_income_taxes)
        * synthetic_population.household_data["Rent Paid"].sum()
        / synthetic_population.household_data.loc[ind_curr_btl, "Rental Income from Real Estate"].sum()
    )
    synthetic_population.household_data.loc[ind_curr_btl, "Rental Income from Real Estate"] *= rescale_variable
    # Create all remaining properties
    # First get information on landlords and their properties
    num_additional_properties = synthetic_population.household_data[
        "Number of Properties other than Household Main Residence"
    ].values.astype(int)

    landlord_ids = np.flatnonzero(num_additional_properties > 0)
    return landlord_ids, num_additional_properties


def set_social_housing_renters(
    num_other_properties_owned: int, num_renters: int, synthetic_population: SyntheticPopulation
):
    """
    Assigns renters to social housing in case there are more renters than properties.

    This function identifies the current renters in the synthetic population and sorts them by their income.
    It then selects the renters with the lowest income until the number of renters matches the number of properties.
    The tenure status of these renters is updated to -1, indicating that they are now in social housing.
    Their rent paid is also set to the social housing rent.

    Args:
        num_other_properties_owned (int): The number of properties owned.
        num_renters (int): The number of renters.
        synthetic_population (SyntheticPopulation): An instance of the SyntheticPopulation class.

    Returns:
        None
    """
    ind_curr_renting = np.flatnonzero(synthetic_population.household_data["Tenure Status of the Main Residence"] == 0)
    renters_now_in_sh_rel = np.argsort(synthetic_population.household_data["Income"].values[ind_curr_renting])[
        0 : num_renters - num_other_properties_owned
    ]
    renters_now_in_sh = ind_curr_renting[renters_now_in_sh_rel]
    synthetic_population.household_data.loc[renters_now_in_sh, "Tenure Status of the Main Residence"] = -1
    synthetic_population.household_data.loc[renters_now_in_sh, "Rent Paid"] = synthetic_population.social_housing_rent


def match_renters_to_properties(
    synthetic_population: SyntheticPopulation,
    housing_market_df: pd.DataFrame,
    max_matching_size: int = 1000,
) -> None:
    """
    Matches renters to properties based on their rent payments.
    This will be done by identifying renters and the properties. The goal of this function is to match renters to properties
    by minimising the difference between the rent paid by the household and the rent of the property.  Because the data is large,
    the matching is done in chunks, with a maximum chunk size of max_matching_size. These chunks are obtained by sorting the data, as we expect
    e.g. the rent paid by renters to be similar to the rent of the property they are matched to within each chunk.

    Then, the matching is done within each chunk using the linear sum assignment algorithm from scipy. The housing market data and household
    data are updated to reflect this matching.

    The rental income of landlords is then computed to match the rent paid by the renters ex post.

    Args:
        synthetic_population (SyntheticPopulation): The synthetic population data.
        housing_market_df (pd.DataFrame): The housing market data.
        max_matching_size (int, optional): The maximum size of each matching chunk. Defaults to 1000.

    Returns:
        None
    """
    rented = ~housing_market_df["Is Owner-Occupied"]
    rent_rec = housing_market_df.loc[rented, "Rent"].values

    renters_ind = np.flatnonzero(synthetic_population.household_data["Tenure Status of the Main Residence"] == 0)

    renters = synthetic_population.household_data["Tenure Status of the Main Residence"] == 0
    rent_paid = synthetic_population.household_data.loc[renters, "Rent Paid"].values

    # Step 1: Sort the arrays and keep track of the original indices
    sorted_indices_rec = np.argsort(rent_rec)
    sorted_indices_paid = np.argsort(rent_paid)
    rent_rec_sorted = rent_rec[sorted_indices_rec]
    rent_paid_sorted = rent_paid[sorted_indices_paid]

    # Step 2: Split the sorted arrays into chunks
    chunk_size = max(1, int(len(rent_rec_sorted) / max_matching_size))
    rent_rec_split = np.array_split(rent_rec_sorted, chunk_size)
    chunk_size = max(1, int(len(rent_paid_sorted) / max_matching_size))
    rent_paid_split = np.array_split(rent_paid_sorted, chunk_size)

    # Initialize the variables
    split_offset_rec, split_offset_paid = 0, 0
    corr_renters_by_house_id_rel = np.full(rent_rec.shape, np.nan)

    # Step 3: Calculate the cost matrix and find the optimal assignment for each chunk
    for chunk_ind in range(len(rent_rec_split)):
        curr_rent_rec = rent_rec_split[chunk_ind]
        curr_rent_paid = rent_paid_split[chunk_ind]
        cost = sp.spatial.distance_matrix(curr_rent_rec[:, None], curr_rent_paid[:, None])
        curr_properties, curr_renters = lsa(cost)

        # Step 4: Map the indices back to the original indices
        curr_properties = sorted_indices_rec[curr_properties + split_offset_rec]
        curr_renters = sorted_indices_paid[curr_renters + split_offset_paid]

        # Step 5: Update the corr_renters_by_house_id_rel array
        corr_renters_by_house_id_rel[curr_properties] = renters_ind[curr_renters]

        split_offset_rec += len(curr_rent_rec)
        split_offset_paid += len(curr_rent_paid)

    corr_renters_by_house_id_hb = np.full(housing_market_df.shape[0], np.nan)
    corr_renters_by_house_id_hb[rented] = corr_renters_by_house_id_rel
    # add to housing_market_df
    housing_market_df.loc[rented, "Corresponding Inhabitant Household ID"] = corr_renters_by_house_id_hb[rented]

    housing_market_df["Up for Rent"] = housing_market_df["Corresponding Inhabitant Household ID"].isna()

    housing_market_df["Corresponding Owner Household ID"] = housing_market_df[
        "Corresponding Owner Household ID"
    ].astype(int)

    owners_to_renters = housing_market_df.groupby("Corresponding Owner Household ID")[
        "Corresponding Inhabitant Household ID"
    ].apply(list)

    synthetic_population.household_data["Corresponding Renters"] = [
        [] for _ in range(len(synthetic_population.household_data))
    ]
    mapped_renters = synthetic_population.household_data.index.map(owners_to_renters)
    synthetic_population.household_data.loc[~mapped_renters.isna(), "Corresponding Renters"] = mapped_renters[
        ~mapped_renters.isna()
    ]

    mapped_df = housing_market_df.dropna(subset=["Corresponding Inhabitant Household ID"]).set_index(
        "Corresponding Inhabitant Household ID"
    )

    mapped_df.index = mapped_df.index.astype(int)

    synthetic_population.household_data["Corresponding Inhabited House ID"] = mapped_df["House ID"]

    synthetic_population.household_data["Rent Paid"] = mapped_df["Rent"]

    rental_income = rented & ~housing_market_df["Up for Rent"]

    earned_rent = housing_market_df.loc[rental_income].groupby("Corresponding Owner Household ID")["Rent"].sum()

    synthetic_population.household_data.loc[earned_rent.index.values, "Rental Income from Real Estate"] = (
        earned_rent.values
    )
