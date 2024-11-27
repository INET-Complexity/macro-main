from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa  # noqa


class HousingMarketClearer(ABC):
    def __init__(self, random_assignment_shock_variance: float):
        self.random_assignment_shock_variance = random_assignment_shock_variance

    @abstractmethod
    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        pass


class NoHousingMarketClearer(HousingMarketClearer):
    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sales_types": [],
                "property_id": [],
                "seller_id": [],
                "buyer_id": [],
                "property_value": [],
                "price_or_rent": [],
            }
        )


class DefaultHousingMarketClearer(HousingMarketClearer):
    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        # Sales market
        matching_sales = self.perform_matching(
            housing_data=housing_data,
            household_main_residence_tenure_status=household_main_residence_tenure_status,
            max_willing_to_pay=max_price_willing_to_pay,
            is_rental_market=False,
        )

        # Rental market
        matching_rental = self.perform_matching(
            housing_data=housing_data,
            household_main_residence_tenure_status=household_main_residence_tenure_status,
            max_willing_to_pay=max_rent_willing_to_pay,
            is_rental_market=True,
            households_already_operated=matching_sales["buyer_id"].values,
        )

        # Combine both
        all_transactions = pd.concat((matching_sales, matching_rental), axis=0).reset_index(drop=True)
        all_transactions["property_id"] = all_transactions["property_id"].astype(int)
        all_transactions["seller_id"] = all_transactions["seller_id"].astype(int)
        all_transactions["buyer_id"] = all_transactions["buyer_id"].astype(int)
        return all_transactions

    @staticmethod
    def perform_matching(
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,  # noqa
        max_willing_to_pay: np.ndarray,
        is_rental_market: bool,
        households_already_operated: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if households_already_operated is None:
            households_already_operated = []

        # Data on potential new sales/rental
        new_sales_types = []
        new_property_id = []
        new_property_value = []
        new_price_or_rent = []
        new_seller_id = []
        new_buyer_id = []

        # The type of sales
        sales_type = "Rental" if is_rental_market else "Sell"

        # Iterate over households with demand for properties
        households_with_demand = np.where(max_willing_to_pay > 0)[0]
        households_with_demand_shuffled = np.random.choice(
            households_with_demand,
            len(households_with_demand),
            replace=False,
        )
        """
        # Prioritisation
        households_in_social_housing = np.where(household_main_residence_tenure_status == -1)[0]
        households_with_demand_shuffled_social_housing = np.intersect1d(
            households_with_demand_shuffled, households_in_social_housing
        )
        households_with_demand_shuffled_not_social_housing = np.setdiff1d(
            households_with_demand_shuffled,
            households_with_demand_shuffled_social_housing,
        )
        households_with_demand_shuffled = np.concatenate(
            (
                households_with_demand_shuffled_social_housing,
                households_with_demand_shuffled_not_social_housing,
            )
        )
        """
        for household_id in households_with_demand_shuffled:
            if household_id in households_already_operated:
                continue

            # Pick a property with a price/rent close to what the household wants
            if is_rental_market:
                property_open_ind = np.where(housing_data["Up for Rent"])[0]
                property_prices = housing_data.loc[property_open_ind, "Rent"]
            else:
                property_open_ind = np.where(housing_data["Temporarily for Sale"])[0]
                property_prices = housing_data.loc[property_open_ind, "Sale Price"]
            price_diff = max_willing_to_pay[household_id] - property_prices
            price_diff[price_diff < 0] = np.inf
            if len(price_diff) == 0:
                continue
            property_id_rel = np.argmin(price_diff.values)
            if np.isinf(price_diff.values[property_id_rel]):
                continue
            property_id = property_open_ind[property_id_rel]

            # Property is not available anymore
            if is_rental_market:
                housing_data.at[property_id, "Up for Rent"] = False
            else:
                housing_data.at[property_id, "Temporarily for Sale"] = False

            # Collect results
            new_sales_types.append(sales_type)
            new_property_id.append(property_id)
            new_property_value.append(float(housing_data.loc[property_id, "Value"]))
            if is_rental_market:
                new_price_or_rent.append(float(housing_data.loc[property_id, "Rent"]))
            else:
                new_price_or_rent.append(float(housing_data.loc[property_id, "Sale Price"]))
            new_seller_id.append(int(housing_data.loc[property_id, "Corresponding Owner Household ID"]))
            new_buyer_id.append(int(household_id))

        # Return new (potential!) transactions
        return pd.DataFrame(
            data={
                "sales_types": new_sales_types,
                "property_id": new_property_id,
                "property_value": new_property_value,
                "price_or_rent": new_price_or_rent,
                "seller_id": new_seller_id,
                "buyer_id": new_buyer_id,
            }
        )


class AutomaticHousingMarketClearer(HousingMarketClearer):
    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        # Sales market
        matching_sales = self.perform_matching(
            housing_data=housing_data,
            household_main_residence_tenure_status=household_main_residence_tenure_status,
            max_willing_to_pay=max_price_willing_to_pay,
            is_rental_market=False,
        )

        # Rental market
        matching_rental = self.perform_matching(
            housing_data=housing_data,
            household_main_residence_tenure_status=household_main_residence_tenure_status,
            max_willing_to_pay=max_rent_willing_to_pay,
            is_rental_market=True,
            households_already_operated=matching_sales["buyer_id"].values,
        )

        # Combine both
        all_transactions = pd.concat((matching_sales, matching_rental), axis=0).reset_index(drop=True)
        all_transactions["property_id"] = all_transactions["property_id"].astype(int)
        all_transactions["seller_id"] = all_transactions["seller_id"].astype(int)
        all_transactions["buyer_id"] = all_transactions["buyer_id"].astype(int)
        return all_transactions

    def perform_matching(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,  # noqa
        max_willing_to_pay: np.ndarray,
        is_rental_market: bool,
        households_already_operated: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if is_rental_market:
            sales_type, price_field, status_field = (
                "Rental",
                "Rent",
                "Up for Rent",
            )
        else:
            sales_type, price_field, status_field = (
                "Sell",
                "Sale Price",
                "Temporarily for Sale",
            )

        # Select which households are looking to rent or to buy
        if households_already_operated is None:
            households_already_operated = np.array([])
        households_with_demand = np.where(max_willing_to_pay > 0)[0]
        households_with_demand = np.setdiff1d(
            households_with_demand,
            households_already_operated,
        )
        """
        households_in_social_housing = np.where(household_main_residence_tenure_status == -1)[0]
        households_with_demand_not_social_housing = np.setdiff1d(
            households_with_demand,
            households_in_social_housing,
        )
        """

        # Collect properties
        property_open_ind = np.where(housing_data[status_field])[0]
        property_prices = housing_data.loc[property_open_ind, price_field].values

        # Create a cost matrix
        cost = sp.spatial.distance_matrix(
            max_willing_to_pay[households_with_demand][:, None],
            (
                (
                    1
                    + np.random.normal(
                        0.0,
                        self.random_assignment_shock_variance,
                        property_prices.shape[0],
                    )
                )
                * property_prices
            )[:, None],
        )
        cost[cost < 0] = np.inf

        # Find an optimal assignment and record the outcome
        rel_households, rel_properties = lsa(cost)
        abs_households, abs_properties = (
            households_with_demand[rel_households],
            property_open_ind[rel_properties],
        )
        housing_data.loc[abs_properties, status_field] = False
        return pd.DataFrame(
            data={
                "sales_types": np.full(abs_households.shape[0], sales_type),
                "property_id": abs_properties,
                "property_value": housing_data.loc[abs_properties, "Value"].values,
                "price_or_rent": housing_data.loc[abs_properties, price_field].values,
                "seller_id": housing_data.loc[abs_properties, "Corresponding Owner Household ID"].values,
                "buyer_id": abs_households,
            }
        )
