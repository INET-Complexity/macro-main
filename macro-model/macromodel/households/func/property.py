import warnings
import numpy as np
import pandas as pd

from abc import abstractmethod, ABC

from typing import Tuple


class HouseholdDemandForProperty(ABC):
    def __init__(
        self,
        probability_stay_in_rented_property: float,
        maximum_price_income_coefficient: float,
        maximum_price_income_exponent: float,
        maximum_price_noise_mean: float,
        maximum_price_noise_variance: float,
        psychological_pressure_of_renting: float,
        cost_comparison_temperature: float,
        maximum_rent_income_coefficient: float,
        maximum_rent_income_exponent: float,
        probability_stay_in_owned_property: float,
        price_initial_markup: float,
        price_decrease_probability: float,
        price_decrease_mean: float,
        price_decrease_variance: float,
        rent_initial_markup: float,
        rent_decrease_probability: float,
        rent_decrease_mean: float,
        rent_decrease_variance: float,
        partial_rent_inflation_indexation: float,
        partial_rent_inflation_delay: int,
    ):
        self.probability_stay_in_rented_property = (
            probability_stay_in_rented_property
        )
        self.probability_stay_in_owned_property = (
            probability_stay_in_owned_property
        )
        self.maximum_price_income_coefficient = maximum_price_income_coefficient
        self.maximum_price_income_exponent = maximum_price_income_exponent
        self.maximum_price_noise_mean = maximum_price_noise_mean
        self.maximum_price_noise_variance = maximum_price_noise_variance
        self.maximum_rent_income_coefficient = maximum_rent_income_coefficient
        self.maximum_rent_income_exponent = maximum_rent_income_exponent
        self.psychological_pressure_of_renting = (
            psychological_pressure_of_renting
        )
        self.cost_comparison_temperature = cost_comparison_temperature
        self.price_initial_markup = price_initial_markup
        self.price_decrease_probability = price_decrease_probability
        self.price_decrease_mean = price_decrease_mean
        self.price_decrease_variance = price_decrease_variance
        self.rent_initial_markup = rent_initial_markup
        self.rent_decrease_probability = rent_decrease_probability
        self.rent_decrease_mean = rent_decrease_mean
        self.rent_decrease_variance = rent_decrease_variance
        self.partial_rent_inflation_indexation = (
            partial_rent_inflation_indexation
        )
        self.partial_rent_inflation_delay = partial_rent_inflation_delay

    @abstractmethod
    def compute_demand(
        self,
        housing_data: pd.DataFrame,
        household_residence_tenure_status: np.ndarray,
        household_income: np.ndarray,
        household_financial_wealth: np.ndarray,
        observed_fraction_value_price: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
        expected_hpi_growth: float,
        assumed_mortgage_maturity: int,
        rental_income_taxes: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def compute_initial_sale_price(
        self, property_values: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_updated_sale_price(
        self,
        sale_prices: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_offered_rent_for_existing_properties(
        self,
        current_offered_rent: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        pass


class DefaultHouseholdDemandForProperty(HouseholdDemandForProperty):
    def compute_demand(
        self,
        housing_data: pd.DataFrame,
        household_residence_tenure_status: np.ndarray,
        household_income: np.ndarray,
        household_financial_wealth: np.ndarray,
        observed_fraction_value_price: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
        expected_hpi_growth: float,
        assumed_mortgage_maturity: int,
        rental_income_taxes: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Indices
        ind_in_social_housing = household_residence_tenure_status == -1
        ind_renting = np.array(household_residence_tenure_status == 0)
        ind_renting_not_staying = np.logical_and(
            ind_renting,
            np.random.random(ind_renting.shape[0])
            > self.probability_stay_in_rented_property,
        )
        ind_owning = np.array(household_residence_tenure_status == 1)
        ind_owning_not_staying = np.logical_and(
            ind_owning,
            np.random.random(ind_owning.shape[0])
            > self.probability_stay_in_owned_property,
        )

        # Decision between renting and owning for households renting or in social housing
        ind_dec = (
            ind_in_social_housing
            | ind_renting_not_staying
            | ind_owning_not_staying
        )
        max_amount_pay = (
            self.maximum_price_income_coefficient
            * household_income[ind_dec] ** self.maximum_price_income_exponent
            * np.exp(
                np.random.normal(
                    self.maximum_price_noise_mean,
                    self.maximum_price_noise_variance**0.5,
                    np.sum(ind_dec),
                )
            )
        )
        max_value_affordable = (
            observed_fraction_value_price[0] * max_amount_pay
            + observed_fraction_value_price[1]
        )
        max_corresponding_rent = (
            observed_fraction_rent_value[0] * max_value_affordable
            + observed_fraction_rent_value[1]
        )
        annual_cost_of_renting = (
            4
            * (1 + self.psychological_pressure_of_renting)
            * max_corresponding_rent
        )
        annual_cost_of_purchasing = (
            4
            * np.maximum(
                0, max_amount_pay - household_financial_wealth[ind_dec]
            )
            / assumed_mortgage_maturity
            - ((1 + expected_hpi_growth) ** 4 - 1) * max_value_affordable
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diff_exp = np.exp(
                self.cost_comparison_temperature
                * (annual_cost_of_renting - annual_cost_of_purchasing)
            )
            diff_exp[
                np.logical_and(
                    np.isinf(diff_exp),
                    annual_cost_of_renting - annual_cost_of_purchasing < 0,
                )
            ] *= -1
            prob_buying = 1.0 / diff_exp
        ind_deciding_to_buy_rel = (
            np.random.random(prob_buying.shape) < prob_buying
        )
        ind_deciding_to_rent_rel = ~ind_deciding_to_buy_rel
        ind_deciding_to_buy = np.where(ind_dec)[0][ind_deciding_to_buy_rel]
        ind_deciding_to_rent = np.where(ind_dec)[0][ind_deciding_to_rent_rel]

        # The price households are willing to pay
        max_price_willing_to_pay = np.full(household_income.shape, np.nan)
        max_price_willing_to_pay[ind_deciding_to_buy] = max_amount_pay[
            ind_deciding_to_buy_rel
        ]

        # The rent households are willing to pay
        max_rent_willing_to_pay = np.full(household_income.shape, np.nan)
        max_rent_willing_to_pay[ind_deciding_to_rent] = (
            self.maximum_rent_income_coefficient
            * household_income[ind_deciding_to_rent]
            ** self.maximum_rent_income_exponent
        )

        return (
            max_price_willing_to_pay,
            max_rent_willing_to_pay,
            ind_owning_not_staying,
        )

    def compute_initial_sale_price(
        self,
        property_values: np.ndarray,
    ) -> np.ndarray:
        return (1 + self.price_initial_markup) * property_values

    def compute_updated_sale_price(
        self,
        sale_prices: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        new_sale_prices = sale_prices.copy()
        properties_with_reduced_price = (
            np.random.random(sale_prices.shape)
            < self.price_decrease_probability
        )
        new_sale_prices[properties_with_reduced_price] *= np.maximum(
            max_decrease,
            1
            - np.exp(
                np.random.normal(
                    self.price_decrease_mean,
                    self.price_decrease_variance**0.5,
                    np.sum(properties_with_reduced_price),
                )
            ),
        )
        return new_sale_prices

    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        return (
            1
            + np.maximum(
                0.0,
                self.partial_rent_inflation_indexation
                * historic_inflation[-self.partial_rent_inflation_delay],
            )
        ) * current_rent

    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        return (1 + self.rent_initial_markup) * (
            observed_fraction_rent_value[0] * property_value
            + observed_fraction_rent_value[1]
        )

    def compute_offered_rent_for_existing_properties(
        self,
        current_offered_rent: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        new_offered_rent = current_offered_rent.copy()
        properties_with_reduced_rent = (
            np.random.random(current_offered_rent.shape)
            < self.rent_decrease_probability
        )
        new_offered_rent[properties_with_reduced_rent] *= 1 - np.exp(
            np.random.normal(
                self.rent_decrease_mean,
                self.rent_decrease_variance**0.5,
                np.sum(properties_with_reduced_rent),
            )
        )
        return new_offered_rent
