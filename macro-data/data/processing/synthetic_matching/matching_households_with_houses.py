import numpy as np
import scipy as sp
import pandas as pd

from scipy.optimize import linear_sum_assignment as lsa

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from data.processing.synthetic_housing_market.synthetic_housing_market import (
    SyntheticHousingMarket,
)

from data.util.clean_data import remove_outliers


def match_households_with_houses(
    synthetic_population: SyntheticPopulation,
    synthetic_housing_market: SyntheticHousingMarket,
    rental_income_taxes: float,
    social_housing_rent: float,
    total_imputed_rent: float,
) -> None:
    # Handling owner-occupied property
    housing_market_data = handle_households_owning(synthetic_population)

    # Preprocess the data
    preprocess(
        synthetic_population=synthetic_population,
        housing_market_data=housing_market_data,
        rental_income_taxes=rental_income_taxes,
        social_housing_rent=social_housing_rent,
        total_imputed_rent=total_imputed_rent,
    )

    # Optimize
    find_optimal_matching(
        synthetic_population=synthetic_population,
        housing_market_data=housing_market_data,
        rental_income_taxes=rental_income_taxes,
    )

    # Record the results
    synthetic_housing_market.housing_market_data = pd.DataFrame(housing_market_data)


def handle_households_owning(synthetic_population: SyntheticPopulation) -> dict:
    housing_market_data = {}

    # Handle households owning their house
    households_owning = synthetic_population.household_data["Tenure Status of the Main Residence"] == 1
    housing_market_data["House ID"] = list(range(np.sum(households_owning)))
    housing_market_data["Is Owner-Occupied"] = [True] * len(housing_market_data["House ID"])
    housing_market_data["Corresponding Owner Household ID"] = list(
        synthetic_population.household_data.loc[households_owning].index.values
    )
    housing_market_data["Corresponding Inhabitant Household ID"] = list(
        synthetic_population.household_data.loc[households_owning].index.values
    )
    housing_market_data["Value"] = list(
        synthetic_population.household_data.loc[households_owning, "Value of the Main Residence"].values
    )
    housing_market_data["Rent"] = [np.nan] * len(housing_market_data["Value"])

    # Updating corresponding household data
    synthetic_population.household_data["Corresponding Inhabited House ID"] = np.full(
        len(synthetic_population.household_data), np.nan
    )
    synthetic_population.household_data.loc[
        households_owning, "Corresponding Inhabited House ID"
    ] = housing_market_data["House ID"]
    synthetic_population.household_data["Corresponding Property Owner"] = np.full(
        len(synthetic_population.household_data), np.nan
    )
    synthetic_population.household_data.loc[households_owning, "Corresponding Property Owner"] = np.where(
        households_owning
    )[0]

    return housing_market_data


def preprocess(
    synthetic_population: SyntheticPopulation,
    housing_market_data: dict,
    rental_income_taxes: float,
    social_housing_rent: float,
    total_imputed_rent: float,
) -> None:
    # Make sure there aren't more renters than houses
    num_renters = int(np.sum(synthetic_population.household_data["Tenure Status of the Main Residence"] == 0))
    num_other_properties_owned = int(
        np.sum(synthetic_population.household_data["Number of Properties other than Household Main Residence"])
    )
    if num_renters > num_other_properties_owned:
        ind_curr_renting = np.where(synthetic_population.household_data["Tenure Status of the Main Residence"] == 0)[0]
        renters_now_in_sh_rel = np.argsort(synthetic_population.household_data["Income"].values[ind_curr_renting])[
            0 : num_renters - num_other_properties_owned
        ]
        renters_now_in_sh = ind_curr_renting[renters_now_in_sh_rel]
        synthetic_population.household_data.loc[renters_now_in_sh, "Tenure Status of the Main Residence"] = -1
        synthetic_population.household_data.loc[
            renters_now_in_sh, "Rent Paid"
        ] = synthetic_population.social_housing_rent

    # Rescale total rent received to match rent paid
    ind_curr_btl = np.where(synthetic_population.household_data["Rental Income from Real Estate"] > 0.0)[0]
    synthetic_population.household_data.loc[ind_curr_btl, "Rental Income from Real Estate"] *= (
        (1 - rental_income_taxes)
        * synthetic_population.household_data["Rent Paid"].values.sum()
        / synthetic_population.household_data.loc[ind_curr_btl, "Rental Income from Real Estate"].values.sum()
    )

    # Create all remaining properties
    num_additional_properties = synthetic_population.household_data[
        "Number of Properties other than Household Main Residence"
    ].values.astype(int)
    rental_income = synthetic_population.household_data["Rental Income from Real Estate"].values
    property_values = synthetic_population.household_data["Value of other Properties"].values
    landlord_ids = np.where(num_additional_properties > 0)[0]
    rented_out_property_ids = list(
        range(
            len(housing_market_data["House ID"]),
            len(housing_market_data["House ID"]) + num_additional_properties.sum(),
        )
    )
    housing_market_data["House ID"] += rented_out_property_ids
    housing_market_data["Is Owner-Occupied"] += [False] * len(rented_out_property_ids)
    housing_market_data["Is Owner-Occupied"] = np.array(housing_market_data["Is Owner-Occupied"])
    housing_market_data["Corresponding Owner Household ID"] += list(
        np.concatenate([[landlord_id] * num_additional_properties[landlord_id] for landlord_id in landlord_ids])
    )
    housing_market_data["Rent"] += list(
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
    housing_market_data["Rent"] = np.array(housing_market_data["Rent"])
    num_non_additional_houses = len(housing_market_data["Value"])
    housing_market_data["Value"] += list(
        np.concatenate(
            [
                [property_values[landlord_id] / num_additional_properties[landlord_id]]
                * num_additional_properties[landlord_id]
                for landlord_id in landlord_ids
            ]
        )
    )
    housing_market_data["Value"] = np.array(housing_market_data["Value"])

    # Assign properties to property owners
    synthetic_population.household_data["Corresponding Additionally Owned Houses ID"] = np.array_split(
        rented_out_property_ids,
        np.cumsum(num_additional_properties),
    )[:-1]

    # Clean rent&value
    housing_market_data["Value"][housing_market_data["Value"] < 60 * housing_market_data["Rent"]] = np.nan
    cleaned_housing_market_data = remove_outliers(
        data=pd.DataFrame({"Rent": housing_market_data["Rent"], "Value": housing_market_data["Value"]}),
        cols=["Rent", "Value"],
        quantile=0.2,
    )
    housing_market_data["Rent"] = cleaned_housing_market_data["Rent"]
    housing_market_data["Value"] = cleaned_housing_market_data["Value"]
    imputed_data = IterativeImputer().fit_transform(
        np.stack([housing_market_data["Rent"], housing_market_data["Value"]], axis=1)
    )
    housing_market_data["Rent"] = imputed_data[:, 0]
    housing_market_data["Value"] = imputed_data[:, 1]
    housing_market_data["Rent"][housing_market_data["Rent"] < social_housing_rent] = social_housing_rent

    # Rescale property values
    housing_market_data["Value"][num_non_additional_houses:] = np.concatenate(
        [
            [property_values[landlord_id] / num_additional_properties[landlord_id]]
            * num_additional_properties[landlord_id]
            for landlord_id in landlord_ids
        ]
    )

    # Fill-in imputed rents
    synthetic_population.household_data["Rent Imputed"] = np.zeros(len(synthetic_population.household_data))
    ind_curr_owning = synthetic_population.household_data["Tenure Status of the Main Residence"] == 1
    housing_market_data["Rent"][0 : ind_curr_owning.sum()] *= (
        total_imputed_rent / housing_market_data["Rent"][0 : ind_curr_owning.sum()].sum()
    )
    synthetic_population.household_data.loc[ind_curr_owning, "Rent Imputed"] = housing_market_data["Rent"][
        0 : ind_curr_owning.sum()
    ]

    housing_market_data["Is Owner-Occupied"]


def find_optimal_matching(
    synthetic_population: SyntheticPopulation,
    housing_market_data: dict,
    rental_income_taxes: float,
    max_matching_size: int = 1000,
) -> None:
    rented_houses = np.logical_not(housing_market_data["Is Owner-Occupied"])
    rent_rec = housing_market_data["Rent"][rented_houses]
    renters = synthetic_population.household_data["Tenure Status of the Main Residence"] == 0
    renters_ind = np.where(renters)[0]
    rent_paid = synthetic_population.household_data["Rent Paid"].values[renters]

    # Find the optimal configuration
    rent_rec_split = np.array_split(rent_rec, int(len(rent_rec) / max_matching_size))
    rent_paid_split = np.array_split(rent_paid, int(len(rent_rec) / max_matching_size))
    split_offset_rec, split_offset_paid = 0, 0
    corr_renters_by_house_id_rel = np.full(rent_rec.shape, np.nan)
    for chunk_ind in range(len(rent_rec_split)):
        curr_rent_rec = rent_rec_split[chunk_ind]
        curr_rent_paid = rent_paid_split[chunk_ind]
        cost = sp.spatial.distance_matrix(curr_rent_rec[:, None], curr_rent_paid[:, None])
        curr_properties, curr_renters = lsa(cost)
        corr_renters_by_house_id_rel[curr_properties + split_offset_rec] = renters_ind[curr_renters + split_offset_paid]
        split_offset_rec += len(curr_rent_rec)
        split_offset_paid += len(curr_rent_paid)
    corr_renters_by_house_id_hb = np.full(rented_houses.shape, np.nan)
    corr_renters_by_house_id_hb[rented_houses] = corr_renters_by_house_id_rel

    # Invert it
    rent_house_map = dict(zip(range(len(corr_renters_by_house_id_hb)), corr_renters_by_house_id_hb))
    house_rent_map = {int(v): k for k, v in rent_house_map.items() if not np.isnan(v)}
    corr_house_by_renter = np.full(len(synthetic_population.household_data), np.nan)
    for k in house_rent_map:
        corr_house_by_renter[k] = house_rent_map[k]
    corr_house_by_renter_rel = corr_house_by_renter[renters].astype(int)

    # Update property data
    housing_market_data["Corresponding Inhabitant Household ID"] += list(corr_renters_by_house_id_hb[rented_houses])
    housing_market_data["Corresponding Inhabitant Household ID"] = np.array(
        housing_market_data["Corresponding Inhabitant Household ID"]
    )
    housing_market_data["Up for Rent"] = np.isnan(housing_market_data["Corresponding Inhabitant Household ID"])

    # Update household data
    property_owner_ids = np.where(synthetic_population.household_data["Rental Income from Real Estate"] != 0.0)[0]
    synthetic_population.household_data["Corresponding Renters"] = [
        [] for _ in range(len(synthetic_population.household_data))
    ]
    for po_id in property_owner_ids:
        synthetic_population.household_data.at[po_id, "Corresponding Renters"] = list(
            corr_renters_by_house_id_hb[
                synthetic_population.household_data.loc[po_id, "Corresponding Additionally Owned Houses ID"]
            ]
        )
    synthetic_population.household_data.loc[renters, "Corresponding Inhabited House ID"] = corr_house_by_renter_rel
    synthetic_population.household_data.loc[renters, "Corresponding Property Owner"] = np.array(
        housing_market_data["Corresponding Owner Household ID"]
    )[corr_house_by_renter_rel]

    # Update rent paid
    synthetic_population.household_data.loc[:, "Rent Paid"] = 0.0
    for renter in np.where(renters)[0]:
        synthetic_population.household_data.at[renter, "Rent Paid"] = housing_market_data["Rent"][
            int(corr_house_by_renter[renter])
        ]

    # Update received rent
    synthetic_population.household_data.loc[:, "Rental Income from Real Estate"] = 0.0
    for po in property_owner_ids:
        additional_properties = synthetic_population.household_data.loc[
            po, "Corresponding Additionally Owned Houses ID"
        ]
        additional_properties = [a for a in additional_properties if not housing_market_data["Up for Rent"][a]]
        synthetic_population.household_data.loc[po, "Rental Income from Real Estate"] = (
            1 - rental_income_taxes
        ) * housing_market_data["Rent"][additional_properties].sum()
