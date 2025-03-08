"""Housing market clearing mechanisms and algorithms.

This module implements various algorithms for matching buyers/renters with
properties in the housing market. It provides an abstract base class for
market clearing and several concrete implementations with different matching
strategies.

The module supports:
- Both sales and rental markets
- Price-based matching
- Priority-based allocation
- Multiple clearing strategies

Key components:
1. HousingMarketClearer: Abstract base class defining the interface
2. NoHousingMarketClearer: Null implementation for testing
3. DefaultHousingMarketClearer: Simple price-based matching
4. AutomaticHousingMarketClearer: Optimized matching using linear assignment
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import linear_sum_assignment as lsa  # noqa


class HousingMarketClearer(ABC):
    """Abstract base class for housing market clearing algorithms.

    This class defines the interface for market clearing mechanisms that
    match properties with potential buyers or renters. It supports both
    sales and rental markets with configurable matching behavior.
    """

    def __init__(self, random_assignment_shock_variance: float):
        """Initialize the market clearer.

        Args:
            random_assignment_shock_variance: Variance for random perturbations
                in matching decisions, allowing for some randomness in
                otherwise deterministic matches.
        """
        self.random_assignment_shock_variance = random_assignment_shock_variance

    @abstractmethod
    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        """Clear the housing market by matching properties with buyers/renters.

        Args:
            housing_data: DataFrame containing property information
            household_main_residence_tenure_status: Array indicating current
                housing status for each household
            max_price_willing_to_pay: Maximum purchase prices households
                are willing to pay
            max_rent_willing_to_pay: Maximum rents households are willing
                to pay

        Returns:
            pd.DataFrame: Matched transactions with columns:
                - sales_types: "Sell" or "Rental"
                - property_id: ID of the property
                - seller_id: ID of the seller/landlord
                - buyer_id: ID of the buyer/tenant
                - property_value: Current value of the property
                - price_or_rent: Agreed price or rent
        """
        pass


class NoHousingMarketClearer(HousingMarketClearer):
    """Null implementation that performs no market clearing.

    This class implements a no-op market clearer that always returns an
    empty transaction list. It's useful for testing and as a neutral
    baseline for comparing other clearing mechanisms.
    """

    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        """Return an empty transaction list without performing any matching.

        Args:
            housing_data: Ignored in this implementation
            household_main_residence_tenure_status: Ignored
            max_price_willing_to_pay: Ignored
            max_rent_willing_to_pay: Ignored

        Returns:
            pd.DataFrame: Empty DataFrame with required columns
        """
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
    """Default implementation using price-based matching.

    This class implements a simple market clearing mechanism that matches
    properties with buyers/renters based on prices and willingness to pay.
    It processes sales first, then rentals, ensuring each household
    participates in at most one transaction.
    """

    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        """Clear both sales and rental markets sequentially.

        This method first clears the sales market, then the rental market,
        ensuring households that successfully purchase don't also rent.

        Args:
            housing_data: DataFrame containing property information
            household_main_residence_tenure_status: Current housing status
            max_price_willing_to_pay: Maximum purchase prices
            max_rent_willing_to_pay: Maximum rents

        Returns:
            pd.DataFrame: Combined sales and rental transactions
        """
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
        """Match properties with buyers/renters for one market type.

        This method implements the core matching logic, finding suitable
        properties for each household based on their willingness to pay
        and the properties' prices/rents.

        Args:
            housing_data: Property information
            household_main_residence_tenure_status: Current housing status
            max_willing_to_pay: Maximum prices/rents willing to pay
            is_rental_market: Whether this is rental (True) or sales (False)
            households_already_operated: Optional list of households to exclude

        Returns:
            pd.DataFrame: Matched transactions for this market type
        """
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
    """Optimized implementation using linear assignment algorithm.

    This class implements an efficient market clearing mechanism using
    the Hungarian algorithm for optimal matching. It considers the entire
    market simultaneously rather than processing households sequentially.
    """

    def clear(
        self,
        housing_data: pd.DataFrame,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> pd.DataFrame:
        """Clear both markets using optimal matching algorithm.

        This method applies the Hungarian algorithm to find optimal
        matches in both sales and rental markets, maximizing total
        market satisfaction.

        Args:
            housing_data: Property information
            household_main_residence_tenure_status: Current housing status
            max_price_willing_to_pay: Maximum purchase prices
            max_rent_willing_to_pay: Maximum rents

        Returns:
            pd.DataFrame: Optimally matched transactions
        """
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
        """Perform optimal matching for one market type.

        This method constructs a cost matrix between households and
        properties, then applies the Hungarian algorithm to find the
        optimal assignment that maximizes total market satisfaction.

        Args:
            housing_data: Property information
            household_main_residence_tenure_status: Current housing status
            max_willing_to_pay: Maximum prices/rents willing to pay
            is_rental_market: Whether this is rental (True) or sales (False)
            households_already_operated: Optional list of households to exclude

        Returns:
            pd.DataFrame: Optimally matched transactions for this market type

        Note:
            The cost matrix is constructed using the difference between
            willingness to pay and actual prices, with impossible matches
            assigned infinite cost.
        """
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
