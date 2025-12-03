"""Module for harmonizing household and housing unit data.

This module harmonizes housing data from different sources:
1. Household Survey Data:
   - Reported property ownership
   - Rental payments
   - Housing wealth
   - Tenure status

2. Property Register Data:
   - Housing unit values
   - Rental income
   - Ownership records
   - Social housing status

The harmonization process involves:
1. Owner-Occupied Housing:
   - Reconciling ownership records
   - Harmonizing property values
   - Adjusting imputed rents
   - Validating relationships

2. Rental Market:
   - Reconciling landlord holdings
   - Harmonizing rental income
   - Adjusting rental rates
   - Matching tenant records

3. Data Validation:
   - Removing outliers
   - Imputing missing values
   - Ensuring market consistency
   - Validating relationships

Note:
    This module focuses on harmonizing housing data from different sources
    to create a consistent initial state. The actual housing market dynamics
    are implemented in the simulation package.

    Tenure status in HFCS falls into 4 categories:
    1. Own ('1')
    2. Part own ('2')
    3. Rent ('3')
    4. Free use of their home ('4')

    This module treats part-owners and free-users as owner-occupiers. It adds
    a category for social renters ('-1').
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
    """Create and harmonize the housing market dataset.

    This function reconciles housing data through several steps:
    1. Owner-Occupied Properties:
       - Reconciling ownership records with survey data
       - Harmonizing property values across sources
       - Computing consistent imputed rents

    2. Rental Properties:
       - Reconciling landlord holdings with property records
       - Harmonizing rental income with payments
       - Adjusting for tax effects
       - Validating social housing data

    3. Data Cleaning:
       - Removing outliers in values and rents
       - Imputing missing data points
       - Validating price-rent relationships
       - Ensuring data consistency

    4. Market Reconciliation:
       - Harmonizing tenant-property relationships
       - Recording consistent ownership data
       - Updating household records

    Args:
        synthetic_population (SyntheticPopulation): Household survey data
        rental_income_taxes (float): Tax rate on rental income
        social_housing_rent (float): Standardized social housing rent
        total_imputed_rent (float): Total imputed rent for owned properties

    Returns:
        pd.DataFrame: Harmonized housing market data with consistent
            ownership and rental relationships
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

    # List of households that own ('1'), part own ('2') or have free use of their home ('4')
    owner_indices = np.isin(synthetic_population.household_data["Tenure Status of the Main Residence"], [1, 2, 4])
    synthetic_population.household_data.loc[owner_indices, "Rent Imputed"] = housing_df.loc[owned_houses, "Rent"].values

    match_renters_to_properties(
        synthetic_population=synthetic_population, housing_market_df=housing_df, rental_income_taxes=rental_income_taxes
    )

    return housing_df


def create_owners_df(synthetic_population: SyntheticPopulation) -> pd.DataFrame:
    """Create dataframe of owner-occupied properties.

    This function creates a dataframe of owner-occupied properties and their households by:
    1. Fetching a list of all households in HFCS that own, part own or have free use of their home
    2. Creates unique IDs for each of their houses
    3. Creates separate columns for owner ID and occupant ID
    4. Uses corresponding household ID from HFCS to fill owner and occupant ID columns

    The simplifying assumption is that households that part own or have free use of their property
    will one day own their property outright (e.g. through rent-to-buy or inheritance).

    Args:
        synthetic_population (SyntheticPopulation): Household survey data
            with tenure information

    Returns:
        pd.DataFrame: Harmonized owner-occupied property dataset with:
            - House ID: Unique property identifier
            - Is Owner-Occupied: Always True for this dataset
            - Corresponding Owner Household ID: Owner identifier
            - Corresponding Inhabitant Household ID: Same as owner
            - Value: Harmonized property value
            - Rent: NaN (filled later with imputed rent)
    """
    # List of households that own ('1'), part own ('2') or have free use of their home ('4')
    households_owning = np.isin(synthetic_population.household_data["Tenure Status of the Main Residence"], [1, 2, 4])
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
    """Create dataframe of rented properties.

    This function created a dataframe of rented properties, their owners and their occupants by:
    1. Creating a dataframe with one row per rented property
    2. Assigning unique IDs to each property
    3. Assigning landlord IDs to each property
    4. Assigning rental income to each property
    5. Assigning property values to each property

    The process ensures:
    - Property holdings match across sources
    - Rental income is consistently recorded
    - Tax effects are properly handled
    - Values are properly distributed

    Args:
        num_additional_properties (np.ndarray): Properties per landlord
        landlord_ids (np.ndarray | list): Unique landlord identifiers
        property_values (np.ndarray): Total property values per landlord
        id_start (int): Starting ID for rental properties
        rental_income (np.ndarray): Rental income per landlord
        rental_income_taxes (float): Tax rate on rental income

    Returns:
        pd.DataFrame: Harmonized rental property dataset with:
            - House ID: Unique property identifier
            - Is Owner-Occupied: Always False
            - Corresponding Owner Household ID: Landlord identifier
            - Rent: Harmonized rental rate
            - Value: Reconciled property value
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
    """Create landlord IDs and number of properties per landlord.

    This function estimates the number of properties per landlord and their IDs by:
    1. Counting up all the renters using HFCS data on tenure status (response '3' in the data)
    2. Counting up all the non-primary residence properties owned by HFCS respondents
    3. If more renters than properties, the additional renters are assumed to be social renters (using function set_social_housing_renters)
    4. Rescale rent paid to match rent received minus taxes
    5. Fetch owner IDs from HFCS and assign to houses as landlord IDs

    The assumes that all non-primary residence properties are rented out rather than used as holiday homes, short-term lets, etc.

    In future this should be revised to first fetch number of social tenants in each country from another data source, allocate remaining renters to houses, then assume remaining properties are second homes etc.

    Args:
        rental_income_taxes (float): Tax rate on rental income
        synthetic_population (SyntheticPopulation): Household survey data

    Returns:
        tuple:
            - landlord_ids (np.ndarray): Unique landlord identifiers
            - num_additional_properties (np.ndarray): Properties per landlord
    """
    num_renters = int(np.sum(synthetic_population.household_data["Tenure Status of the Main Residence"] == 3))
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
    num_other_properties_owned: int,
    num_renters: int,
    synthetic_population: SyntheticPopulation,
):
    """Social housing allocation.

    If there are more renters than surplus properties owned by owner-occupying households,
    this function defines them as social renters:
    1. Count how many renters there are (response '3' in the HFCS data)
    2. Count whether there are more renters than surplus properties
    3. Assign those excess renters to social housing by giving them tenure status '-1'
    4. Update those households' tenure status records
    5. Update the rent they pay to the social housing rent level

    This is a placeholder simplification pending further work to incorporate social housing data.

    Args:
        num_other_properties_owned (int): Total private rental properties
        num_renters (int): Total number of renters
        synthetic_population (SyntheticPopulation): Household survey data
    """
    ind_curr_renting = np.flatnonzero(synthetic_population.household_data["Tenure Status of the Main Residence"] == 3)
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
    """Match renters to properties.

    This function allocates renters to rented out properties by:
    1. Fetching renters' data on how much rent they pay
    2. Fetching landlords' data on how much rent they receive
    3. Matching renters to landlords by finding the closest rent paid to rent received
    4. Make sure non-renters don't pay rent
    5. Update renters' rent to be in line with housing market data

    This matching is fairly imprecise. Part-owners who pay rent are omitted from this process, instead treated like owner-occupiers.

    Args:
        synthetic_population (SyntheticPopulation): Household survey data
        housing_market_df (pd.DataFrame): Property register data
        rental_income_taxes (float): Tax rate on rental income
        max_matching_size (int, optional): Maximum chunk size for processing.
            Defaults to 1000.
    """
    rented = ~housing_market_df["Is Owner-Occupied"]
    rent_rec = housing_market_df.loc[rented, "Rent"].values

    renters_ind = np.flatnonzero(synthetic_population.household_data["Tenure Status of the Main Residence"] == 3)

    renters = synthetic_population.household_data["Tenure Status of the Main Residence"] == 3
    rent_paid = synthetic_population.household_data.loc[renters, "Rent Paid"].values

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

    # Only reset rent for households that own ('1'), part own ('2') or have free use of their home ('4')
    # Preserve the processed HFCS rent data for renters
    owners = synthetic_population.household_data["Tenure Status of the Main Residence"].isin([1, 2, 4])
    synthetic_population.household_data.loc[owners, "Rent Paid"] = 0

    # Update rent for renting households using housing market data where available
    # This preserves HFCS-processed rent for households not mapped to specific houses
    renter_mapping = mapped_df.loc[~mapped_df["Is Owner-Occupied"]].dropna()
    if len(renter_mapping) > 0:
        renter_house_ids = renter_mapping.index
        synthetic_population.household_data.loc[renter_house_ids, "Rent Paid"] = renter_mapping["Rent"]

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
