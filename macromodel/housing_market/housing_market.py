import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticHousingMarket
from macromodel.configurations import HousingMarketConfiguration
from macromodel.housing_market.housing_market_ts import create_housing_market_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import (
    functions_from_model,
    get_functions,
    update_functions,
)
from macromodel.util.get_histogram import get_histogram


class HousingMarket:
    def __init__(
        self,
        country_name: str,
        scale: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, pd.DataFrame],
    ):
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
        # Get corresponding functions
        functions = functions_from_model(housing_market_configuration.functions, loc="macromodel.housing_market")

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
        property_data["Corresponding Inhabitant Household ID"] = (
            property_data["Corresponding Inhabitant Household ID"].fillna(-1).astype(int)
        )
        property_data["House ID"] = property_data["House ID"].fillna(-1).astype(int)
        property_data["Is Owner-Occupied"] = property_data["Is Owner-Occupied"].fillna(-1).astype(int)
        property_data["Corresponding Owner Household ID"] = property_data["Corresponding Owner Household ID"].astype(
            int
        )
        property_data["Corresponding Inhabitant Household ID"] = (
            property_data["Corresponding Inhabitant Household ID"].fillna(-1).astype(int)
        )

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
        self.ts.reset()
        update_functions(model=configuration.functions, loc="macromodel.housing_market", functions=self.functions)
        self.states = deepcopy(self.initial_states)

    @classmethod
    def from_data(
        cls,
        country_name: str,
        scale: int,
        data: pd.DataFrame,
        config: dict[str, Any],
    ) -> "HousingMarket":
        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="macromodel.housing_market",
            func_dir=Path(__file__).parent / "func",
        )

        # Recording the states of all homes
        states = data.copy()
        states["Corresponding Inhabitant Household ID"][np.isnan(states["Corresponding Inhabitant Household ID"])] = -1
        states["House ID"] = states["House ID"].fillna(-1).astype(int)
        states["Is Owner-Occupied"] = states["Is Owner-Occupied"].fillna(-1).astype(int)
        states["Corresponding Owner Household ID"] = states["Corresponding Owner Household ID"].fillna(-1).astype(int)
        states["Corresponding Inhabitant Household ID"] = (
            states["Corresponding Inhabitant Household ID"].fillna(-1).astype(int)
        )

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
        self.states["current_sales"] = self.functions["clearing"].clear(
            housing_data=self.states["properties"],
            household_main_residence_tenure_status=household_main_residence_tenure_status,
            max_price_willing_to_pay=max_price_willing_to_pay,
            max_rent_willing_to_pay=max_rent_willing_to_pay,
        )

    @staticmethod
    def _perform_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(x) == 0 or len(y) == 0:
            return np.zeros(2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array(np.polyfit(x, y, deg=1))

    def compute_observed_fraction_value_price(self) -> np.ndarray:
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
        return self.states["properties"]["Value"].sum()

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("housing_market", group)
