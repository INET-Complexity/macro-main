"""Module for matching households with housing units in the property market.

This module handles the matching of households with housing units based on ownership
and rental relationships. It uses preprocessed data about:
1. Households:
   - Tenure status (owner/renter)
   - Property ownership
   - Rental income
   - Housing preferences

2. Housing Units:
   - Property values
   - Rental rates
   - Ownership status
   - Social housing designation

The matching process involves:
1. Owner-Occupied Housing:
   - Identifying homeowners
   - Assigning property values
   - Computing imputed rents
   - Recording ownership relationships

2. Rental Market:
   - Processing landlord properties
   - Setting rental rates
   - Handling social housing
   - Matching renters with units

3. Data Validation:
   - Removing outliers
   - Imputing missing values
   - Ensuring market consistency
   - Validating relationships

Note:
    This module focuses on the initial matching of households with housing units.
    The actual housing market dynamics (buying, selling, rental agreements)
    are implemented in the simulation package.
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from macro_data.util.clean_data import remove_outliers
from macro_data.util.imputation import apply_iterative_imputer


def set_housing_df(
    synthetic_population: SyntheticPopulation,
    rental_income_taxes: float,
    social_housing_rent: float,
    total_imputed_rent: float,
) -> pd.DataFrame:
    """Create and initialize the housing market dataset.

    This function processes housing market data through several steps:
    1. Owner-Occupied Properties:
       - Identify homeowners from population data
       - Set property values and relationships
       - Compute imputed rents

    2. Rental Properties:
       - Process landlord holdings
       - Set rental rates and property values
       - Handle social housing allocation

    3. Data Cleaning:
       - Remove outliers in values and rents
       - Impute missing data points
       - Validate price-rent relationships
       - Ensure social housing compliance

    4. Market Matching:
       - Match renters with available properties
       - Record tenant-landlord relationships
       - Update household records

    Args:
        synthetic_population (SyntheticPopulation): Population data with
            household information
        rental_income_taxes (float): Tax rate on rental income
        social_housing_rent (float): Standardized social housing rent
        total_imputed_rent (float): Total imputed rent for owned properties

    Returns:
        pd.DataFrame: Complete housing market data with ownership and
            rental relationships
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

    housing_df = remove_outliers(housing_df, cols=["Rent", "Value"], quantile=0.2, use_logpdf=False)

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
    synthetic_population.household_data["Rent Imputed"] = 0.0
    rescale_factor = total_imputed_rent / housing_df.loc[owned_houses, "Rent"].sum()
    housing_df.loc[owned_houses, "Rent"] *= rescale_factor

    owner_indices = synthetic_population.household_data["Tenure Status of the Main Residence"] == 1
    synthetic_population.household_data.loc[owner_indices, "Rent Imputed"] = housing_df.loc[owned_houses, "Rent"].values

    match_renters_to_properties(
        synthetic_population=synthetic_population, housing_market_df=housing_df, rental_income_taxes=rental_income_taxes
    )

    return housing_df


def create_owners_df(synthetic_population: SyntheticPopulation) -> pd.DataFrame:
    """Create dataset of owner-occupied properties.

    This function processes homeowner data by:
    1. Identifying owner-occupiers from tenure status
    2. Assigning unique property identifiers
    3. Setting property values from household data
    4. Establishing ownership relationships

    The process ensures:
    - Each owner has a unique property ID
    - Property values match household records
    - Ownership relationships are properly recorded
    - Owner-occupancy status is flagged

    Args:
        synthetic_population (SyntheticPopulation): Population data with
            household tenure information

    Returns:
        pd.DataFrame: Owner-occupied property dataset with:
            - House ID: Unique property identifier
            - Is Owner-Occupied: Always True for this dataset
            - Corresponding Owner Household ID: Owner identifier
            - Corresponding Inhabitant Household ID: Same as owner
            - Value: Property value from household data
            - Rent: NaN (filled later with imputed rent)
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
    """Create dataset of rental properties.

    This function processes rental property data by:
    1. Creating entries for each rental unit
    2. Assigning property values from landlord data
    3. Computing rental rates from income data
    4. Establishing ownership relationships

    The process assumes:
    - Equal distribution of landlord property values
    - Uniform rental rates across landlord properties
    - Tax-adjusted rental income allocation

    Args:
        num_additional_properties (np.ndarray): Properties per landlord
        landlord_ids (np.ndarray | list): Unique landlord identifiers
        property_values (np.ndarray): Total property values per landlord
        id_start (int): Starting ID for rental properties
        rental_income (np.ndarray): Rental income per landlord
        rental_income_taxes (float): Tax rate on rental income

    Returns:
        pd.DataFrame: Rental property dataset with:
            - House ID: Unique property identifier
            - Is Owner-Occupied: Always False
            - Corresponding Owner Household ID: Landlord identifier
            - Rent: Computed rental rate
            - Value: Allocated property value
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

    rental_df.index = rental_df["House ID"]

    return rental_df


def housing_info_from_population(rental_income_taxes: float, synthetic_population: SyntheticPopulation):
    """Extract housing market information from population data.

    This function processes population data to:
    1. Identify renters and rental demand
    2. Process landlord property holdings
    3. Adjust rental income for tax effects
    4. Handle social housing allocation

    The process ensures:
    - Rental supply matches demand where possible
    - Social housing fills supply gaps
    - Tax effects are properly accounted for
    - Property allocations are consistent

    Args:
        rental_income_taxes (float): Tax rate on rental income
        synthetic_population (SyntheticPopulation): Population data with
            housing information

    Returns:
        tuple:
            - landlord_ids (np.ndarray): Unique landlord identifiers
            - num_additional_properties (np.ndarray): Properties per landlord
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
    rental_income_taxes: float,
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

    # # Step 1: Sort the arrays and keep track of the original indices
    # sorted_indices_rec = np.argsort(rent_rec)
    # sorted_indices_paid = np.argsort(rent_paid)
    # rent_rec_sorted = rent_rec[sorted_indices_rec]
    # rent_paid_sorted = rent_paid[sorted_indices_paid]

    # Step 2: Split the sorted arrays into chunks
    # chunk_size = max(1, int(len(rent_rec_sorted) / max_matching_size), int(len(rent_paid_sorted) / max_matching_size))
    # rent_rec_split = np.array_split(rent_rec_sorted, chunk_size)
    # rent_paid_split = np.array_split(rent_paid_sorted, chunk_size)

    n_split = int(len(rent_rec) / max_matching_size) if len(rent_rec) > max_matching_size else 1
    rent_rec_split = np.array_split(rent_rec, n_split)
    rent_paid_split = np.array_split(rent_paid, n_split)

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
        # curr_properties = sorted_indices_rec[curr_properties + split_offset_rec]
        # curr_renters = sorted_indices_paid[curr_renters + split_offset_paid]

        # Step 5: Update the corr_renters_by_house_id_rel array
        corr_renters_by_house_id_rel[split_offset_rec + curr_properties] = renters_ind[curr_renters + split_offset_paid]

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

    owners = synthetic_population.household_data["Rental Income from Real Estate"] > 0.0

    owners_to_renters = (
        housing_market_df.loc[rented]
        .groupby("Corresponding Owner Household ID")["Corresponding Inhabitant Household ID"]
        .apply(list)
    )

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

    mapped_df.index.name = synthetic_population.household_data.index.name

    synthetic_population.household_data["Corresponding Inhabited House ID"] = mapped_df["House ID"]

    synthetic_population.household_data["Rent Paid"] = 0

    synthetic_population.household_data["Rent Paid"] = mapped_df.loc[~mapped_df["Is Owner-Occupied"], "Rent"]

    rental_income = rented & ~housing_market_df["Up for Rent"]

    earned_rent = housing_market_df.loc[rental_income].groupby("Corresponding Owner Household ID")["Rent"].sum() * (
        1 - rental_income_taxes
    )

    earned_rent.index.name = synthetic_population.household_data.index.name

    synthetic_population.household_data["Rental Income from Real Estate"] = 0.0

    synthetic_population.household_data.loc[earned_rent.index.values, "Rental Income from Real Estate"] = (
        earned_rent.values
    )

    synthetic_population.household_data["Corresponding Additionally Owned Houses ID"] = (
        synthetic_population.household_data["Corresponding Additionally Owned Houses ID"].apply(list)
    )

    synthetic_population.household_data["Rent Paid"] = synthetic_population.household_data["Rent Paid"].fillna(0)
    synthetic_population.household_data["Rental Income from Real Estate"] = synthetic_population.household_data[
        "Rental Income from Real Estate"
    ].fillna(0)
