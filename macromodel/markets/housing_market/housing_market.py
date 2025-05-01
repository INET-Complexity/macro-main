"""Core implementation of the housing market mechanism.

This module provides the central implementation of the housing market, managing
the interaction between buyers and sellers across properties and countries.
It handles market clearing, property management, and tracking of market metrics
over time.
"""

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

import macromodel.util.get_histogram
from macro_data import SyntheticHousingMarket
from macromodel.configurations import HousingMarketConfiguration
from macromodel.markets.housing_market.housing_market_ts import (
    create_housing_market_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import (
    functions_from_model,
    get_functions,
    update_functions,
)
from macromodel.util.get_histogram import get_histogram


class HousingMarket:
    """Housing market model implementing property transactions and rental agreements.

    This class implements a comprehensive housing market model that handles both
    property sales and rental agreements. It manages property valuations, market
    clearing mechanisms, and tracks various market metrics over time.

    Key Features:
    1. Property Management:
       - Track property ownership and occupancy
       - Update property valuations
       - Monitor rental status
       - Handle property transactions

    2. Market Clearing:
       - Match buyers with sellers
       - Match tenants with landlords
       - Process transactions
       - Update ownership records

    3. Market Analysis:
       - Track price-to-value ratios
       - Monitor rent-to-value ratios
       - Calculate market statistics
       - Generate market reports

    4. Time Series Tracking:
       - Property values over time
       - Transaction volumes
       - Price and rent distributions
       - Market efficiency metrics

    The model supports:
    - Owner-occupied and rental properties
    - Multiple property ownership
    - Vacant properties
    - Price discovery mechanisms
    - Market clearing algorithms
    """

    def __init__(
        self,
        country_name: str,
        scale: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, pd.DataFrame],
    ):
        """Initialize a housing market instance.

        Creates a new housing market with specified parameters and initial state.
        The market tracks both the physical properties and their economic
        characteristics over time.

        Args:
            country_name: Name of the country/region for this market
            scale: Scale factor for market size normalization
            functions: Dict of market functions (valuation, clearing, etc.)
            ts: Time series object for tracking market evolution
            states: Dict containing property and transaction data
                - properties: DataFrame of all properties and their attributes
                - current_sales: DataFrame of ongoing transactions

        Example:
            market = HousingMarket(
                country_name="USA",
                scale=1000,
                functions=market_functions,
                ts=market_timeseries,
                states={
                    "properties": property_data,
                    "current_sales": pd.DataFrame()
                }
            )
        """
        self.country_name = country_name
        self.scale = scale
        self.functions = functions
        self.ts = ts
        self.states = states
        self.initial_states = deepcopy(states)
        self.current_sales: pd.DataFrame = pd.DataFrame()

    @classmethod
    def from_pickled_market(
        cls,
        synthetic_housing_market: SyntheticHousingMarket,
        housing_market_configuration: HousingMarketConfiguration,
        scale: int,
        country_name: str,
    ) -> "HousingMarket":
        """Create a housing market instance from pickled synthetic market data.

        This class method initializes a housing market using pre-generated
        synthetic data, typically used for simulation or testing purposes.
        It handles data preprocessing and state initialization.

        Args:
            synthetic_housing_market: Pre-generated synthetic market data
            housing_market_configuration: Configuration parameters
            scale: Scale factor for market size normalization
            country_name: Name of the country/region

        Returns:
            HousingMarket: New housing market instance initialized with
                synthetic data

        Note:
            The synthetic data should include property characteristics,
            ownership information, and market conditions.
        """
        # Get corresponding functions
        functions = functions_from_model(
            housing_market_configuration.functions, loc="macromodel.markets.housing_market"
        )

        #     #     store[country_name + "_synthetic_housing_market"] = (
        #     #         self.synthetic_housing_market[country_name].housing_market_data.astype(float)
        #     #     ).rename_axis("Properties")

        data = synthetic_housing_market.housing_market_data.astype(float)
        property_data = data.copy()
        property_data.rename_axis("Properties", inplace=True)
        property_data["Sale Price"] = property_data["Value"]
        property_data["Newly on the Rental Market"] = False
        property_data["Up for Rent"] = None
        property_data["Temporarily for Sale"] = False

        # property_data["Corresponding Inhabitant Household ID"].loc[
        #     :, np.isnan(property_data["Corresponding Inhabitant Household ID"])
        # ] = -1
        property_data["Corresponding Inhabitant Household ID"] = macromodel.util.get_histogram.fillna(-1).astype(int)
        property_data["House ID"] = macromodel.util.get_histogram.fillna(-1).astype(int)
        property_data["Is Owner-Occupied"] = macromodel.util.get_histogram.fillna(-1).astype(int)
        property_data["Corresponding Owner Household ID"] = property_data["Corresponding Owner Household ID"].astype(
            int
        )
        property_data["Corresponding Inhabitant Household ID"] = macromodel.util.get_histogram.fillna(-1).astype(int)

        ts = create_housing_market_timeseries(
            data=property_data,
            initial_observed_fraction_value_price=cls._perform_linear_regression(
                property_data["Value"].values, property_data["Value"].values
            ),
            initial_observed_fraction_rent_value=cls._perform_linear_regression(
                property_data["Value"].values, property_data["Rent"].values
            ),
            scale=scale,
        )

        states = {"properties": property_data, "current_sales": pd.DataFrame()}

        return cls(
            country_name,
            scale,
            functions,
            ts,
            states,
        )

    def reset(self, configuration: HousingMarketConfiguration) -> None:
        """Reset the housing market to its initial state.

        This method restores the market to its original configuration,
        useful for running multiple simulations or scenarios. It resets
        both the time series data and the market states.

        Args:
            configuration: New configuration parameters to apply
                after reset

        Note:
            This preserves the original market structure while allowing
            for new configuration parameters.
        """
        self.ts.reset()
        update_functions(
            model=configuration.functions, loc="macromodel.agents.housing_market", functions=self.functions
        )
        self.states = deepcopy(self.initial_states)

    @classmethod
    def from_data(
        cls,
        country_name: str,
        scale: int,
        data: pd.DataFrame,
        config: dict[str, Any],
    ) -> "HousingMarket":
        """Create a housing market instance from raw market data.

        This class method initializes a housing market using actual market
        data, providing flexibility in data sources and market setup.
        It handles data preprocessing and state initialization.

        Args:
            country_name: Name of the country/region
            scale: Scale factor for market size normalization
            data: DataFrame containing property and market data
            config: Configuration dictionary including function specifications

        Returns:
            HousingMarket: New housing market instance initialized with
                the provided data

        Note:
            The input data should include all necessary property attributes
            and market conditions for proper initialization.
        """
        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="macromodel.markets.housing_market",
            func_dir=Path(__file__).parent / "func",
        )

        # Recording the states of all homes
        states = data.copy()
        states["Corresponding Inhabitant Household ID"][np.isnan(states["Corresponding Inhabitant Household ID"])] = -1
        states["House ID"] = macromodel.util.get_histogram.fillna(-1).astype(int)
        states["Is Owner-Occupied"] = macromodel.util.get_histogram.fillna(-1).astype(int)
        states["Corresponding Owner Household ID"] = macromodel.util.get_histogram.fillna(-1).astype(int)
        states["Corresponding Inhabitant Household ID"] = macromodel.util.get_histogram.fillna(-1).astype(int)

        # Create the corresponding time series object
        ts = create_housing_market_timeseries(
            data=states,
            initial_observed_fraction_value_price=cls._perform_linear_regression(
                states["Value"].values, states["Value"].values
            ),
            initial_observed_fraction_rent_value=cls._perform_linear_regression(
                states["Value"].values, states["Rent"].values
            ),
            scale=scale,
        )

        return cls(
            country_name,
            scale,
            functions,
            ts,
            states,
        )

    def update_property_value(self) -> None:
        """Update the values of all properties in the market.

        This method applies the configured valuation function to update property
        values based on market conditions and property characteristics. It also
        records the new values in the time series for tracking.

        The update process:
        1. Apply valuation function to current property values
        2. Store new values in property states
        3. Update time series records
        4. Generate value distribution histogram

        Note:
            The specific valuation logic is defined in the configured
            value function, allowing for different valuation models.
        """
        self.states["properties"].loc[:, "Value"] = self.functions["value"].compute_value(
            current_property_values=self.states["properties"]["Value"].values
        )
        self.ts.property_values.append(self.states["properties"]["Value"].values)
        self.ts.property_values_histogram.append(get_histogram(self.ts.current("property_values"), self.scale))

    def clear(
        self,
        household_main_residence_tenure_status: np.ndarray,
        max_price_willing_to_pay: np.ndarray,
        max_rent_willing_to_pay: np.ndarray,
    ) -> None:
        """Clear the housing market by matching buyers/renters with sellers/landlords.

        This method executes the market clearing algorithm to match properties
        with potential buyers or renters based on their preferences and
        constraints. It handles both sales and rental transactions.

        Args:
            household_main_residence_tenure_status: Array indicating each
                household's current residence status (owner/renter/none)
            max_price_willing_to_pay: Array of maximum purchase prices each
                household is willing/able to pay
            max_rent_willing_to_pay: Array of maximum rents each household
                is willing/able to pay

        Note:
            The specific matching logic is defined in the configured
            clearing function, allowing for different clearing mechanisms.
        """
        self.states["current_sales"] = self.functions["clearing"].clear(
            housing_data=self.states["properties"],
            household_main_residence_tenure_status=household_main_residence_tenure_status,
            max_price_willing_to_pay=max_price_willing_to_pay,
            max_rent_willing_to_pay=max_rent_willing_to_pay,
        )

    @staticmethod
    def _perform_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform linear regression with error handling.

        This utility method fits a linear relationship between two arrays,
        handling edge cases and numerical issues. Used primarily for
        price-value and rent-value relationship calculations.

        Args:
            x: Independent variable array
            y: Dependent variable array

        Returns:
            np.ndarray: Regression coefficients [slope, intercept]
                Returns [0, 0] if input arrays are empty

        Note:
            Suppresses warnings during fitting to handle potential
            numerical instabilities.
        """
        if len(x) == 0 or len(y) == 0:
            return np.zeros(2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array(np.polyfit(x, y, deg=1))

    def compute_observed_fraction_value_price(self) -> np.ndarray:
        """Calculate the relationship between property values and sale prices.

        This method analyzes completed sales transactions to determine the
        current relationship between property values and actual sale prices.
        It uses linear regression to estimate the value-to-price ratio.

        Returns:
            np.ndarray: Regression coefficients [slope, intercept] representing
                the relationship between property values and sale prices.
                Returns current ratio if no sales occurred.

        Note:
            The ratio helps track market efficiency and price discovery,
            with deviations from 1.0 indicating potential market imbalances.
        """
        current_sells = self.states["current_sales"].loc[self.states["current_sales"]["sales_types"] == "Sell"]
        self.ts.price_value_histogram.append(
            get_histogram(
                current_sells["price_or_rent"].values / current_sells["property_value"].values,
                None,
            )
        )
        if len(current_sells) == 0:
            return self.ts.current("observed_fraction_value_price")
        return self._perform_linear_regression(
            current_sells["property_value"].values,
            current_sells["price_or_rent"].values,
        )

    def compute_observed_fraction_rent_value(self) -> np.ndarray:
        """Calculate the relationship between property values and rental rates.

        This method analyzes completed rental agreements to determine the
        current relationship between property values and rental rates.
        It uses linear regression to estimate the rent-to-value ratio.

        Returns:
            np.ndarray: Regression coefficients [slope, intercept] representing
                the relationship between property values and rental rates.
                Returns current ratio if no rentals occurred.

        Note:
            The ratio helps track rental market efficiency and yield rates,
            providing insights into investment returns and market balance.
        """
        current_rentals = self.states["current_sales"].loc[self.states["current_sales"]["sales_types"] == "Rental"]
        self.ts.rent_value_histogram.append(
            get_histogram(
                current_rentals["price_or_rent"].values / current_rentals["property_value"].values,
                None,
            )
        )
        if len(current_rentals) == 0:
            return self.ts.current("observed_fraction_rent_value")
        return self._perform_linear_regression(
            current_rentals["price_or_rent"].values,
            current_rentals["property_value"].values,
        )

    def process_housing_market_clearing(
        self,
        household_states: dict[str, Any],
        household_received_mortgages: np.ndarray,
        household_financial_wealth: np.ndarray,
    ) -> None:
        """Process and finalize housing market transactions.

        This method executes the actual property transfers and rental
        agreements after market clearing. It updates ownership records,
        occupancy status, and financial positions for all parties.

        The process handles:
        1. Property Sales:
           - Verify buyer's financial capacity
           - Transfer ownership
           - Update occupancy
           - Record transaction details

        2. Rental Agreements:
           - Update tenant records
           - Record rental relationships
           - Handle vacancy changes
           - Update market statistics

        Args:
            household_states: Dict containing household status information
            household_received_mortgages: Array of approved mortgage amounts
            household_financial_wealth: Array of household wealth levels

        Note:
            This method ensures all market clearing outcomes are properly
            reflected in the system state.
        """
        total_number_of_bought_houses = 0
        total_number_of_newly_rented_houses = 0
        for index, sale in self.states["current_sales"].iterrows():
            buyer_id, seller_id, property_id = (
                int(sale["buyer_id"]),
                int(sale["seller_id"]),
                int(sale["property_id"]),
            )
            prev_property_id = household_states["Corresponding Inhabited House ID"][buyer_id]
            if sale["sales_types"] == "Rental":
                self.states["properties"].loc[property_id, "Corresponding Inhabitant Household ID"] = buyer_id
                if prev_property_id != -1:
                    self.states["properties"].loc[
                        prev_property_id,
                        "Corresponding Inhabitant Household ID",
                    ] = -1
                household_states["Corresponding Inhabited House ID"][buyer_id] = property_id
                household_states["Tenure Status of the Main Residence"][buyer_id] = 0
                household_states["corr_renters"][seller_id].append(buyer_id)
                total_number_of_newly_rented_houses += 1
            elif sale["sales_types"] == "Sell":
                if (
                    household_received_mortgages[buyer_id] > 0
                    or household_financial_wealth[buyer_id] >= sale["price_or_rent"]
                ):
                    # Corresponding property owners
                    self.states["properties"].loc[property_id, "Corresponding Owner Household ID"] = buyer_id
                    household_states["Corresponding Property Owner"][buyer_id] = buyer_id
                    # household_states["corr_additionally_owned_properties"][buyer_id].append(property_id)
                    # household_states["corr_additionally_owned_properties"][seller_id].remove(property_id)

                    # Corresponding inhabitant households
                    self.states["properties"].loc[property_id, "Corresponding Inhabitant Household ID"] = buyer_id
                    if prev_property_id != -1:
                        self.states["properties"].loc[
                            prev_property_id,
                            "Corresponding Inhabitant Household ID",
                        ] = -1

                    # Corresponding inhabited house ID
                    household_states["Corresponding Inhabited House ID"][buyer_id] = property_id

                    # Tenure status
                    household_states["Tenure Status of the Main Residence"][buyer_id] = 1

                    # Corresponding renter
                    if buyer_id in household_states["corr_renters"][seller_id]:
                        household_states["corr_renters"][seller_id].remove(buyer_id)

                    # Price
                    self.states["properties"].loc[property_id, "Value"] = sale["price_or_rent"]

                    # Count
                    total_number_of_bought_houses += 1
            else:
                raise ValueError("Unknown housing market sales type", sale["sales_types"])

            # General stuff
            if (
                self.states["properties"].at[property_id, "Corresponding Owner Household ID"]
                == self.states["properties"].at[property_id, "Corresponding Inhabitant Household ID"]
            ):
                self.states["properties"].at[property_id, "Is Owner-Occupied"] = 1
            else:
                self.states["properties"].at[property_id, "Is Owner-Occupied"] = 0

        # Update aggregates
        self.ts.total_number_of_houses_rented.append(
            [
                np.sum(
                    np.logical_and(
                        self.states["properties"]["Corresponding Inhabitant Household ID"] != -1,
                        self.states["properties"]["Corresponding Inhabitant Household ID"]
                        != self.states["properties"]["Corresponding Owner Household ID"],
                    )
                )
            ]
        )
        self.ts.total_number_of_houses_owner_occupied.append(
            [
                np.sum(
                    self.states["properties"]["Corresponding Inhabitant Household ID"]
                    == self.states["properties"]["Corresponding Owner Household ID"]
                )
            ]
        )
        self.ts.total_number_of_houses_unoccupied.append(
            [np.sum(self.states["properties"]["Corresponding Inhabitant Household ID"] == -1)]
        )
        self.ts.total_number_of_bought_houses.append([total_number_of_bought_houses])
        self.ts.total_number_of_newly_rented_houses.append([total_number_of_newly_rented_houses])

    def compute_total_property_value(self) -> float:
        """Calculate the total value of all properties in the market.

        Returns:
            float: Sum of all property values in the market
        """
        return self.states["properties"]["Value"].sum()

    def save_to_h5(self, group: h5py.Group):
        """Save market state to HDF5 format.

        This method persists the current market state to disk using HDF5
        format, allowing for efficient storage and retrieval of market data.

        Args:
            group: HDF5 group to save the market data into

        Note:
            Saves both current market state and time series data.
        """
        self.ts.write_to_h5("housing_market", group)
