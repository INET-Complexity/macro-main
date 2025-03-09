"""Household property market behavior implementation.

This module implements household property market decisions through:
- Property demand determination
- Price/rent willingness calculation
- Sale listing management
- Rental offering handling

The implementation handles:
- Property tenure decisions
- Price/rent setting
- Market participation choices
- Housing cost comparisons
- Property value adjustments
"""

import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class HouseholdDemandForProperty(ABC):
    """Abstract base class for household property demand behavior.

    Defines interface for property market participation through:
    - Demand determination
    - Price/rent willingness
    - Market participation decisions
    - Property value adjustments

    Attributes:
        probability_stay_in_rented_property (float): Probability of remaining in rental
        probability_stay_in_owned_property (float): Probability of keeping ownership
        maximum_price_income_coefficient (float): Price/income ratio coefficient
        maximum_price_income_exponent (float): Price/income ratio exponent
        maximum_price_noise_mean (float): Price noise distribution mean
        maximum_price_noise_variance (float): Price noise distribution variance
        maximum_rent_income_coefficient (float): Rent/income ratio coefficient
        maximum_rent_income_exponent (float): Rent/income ratio exponent
        psychological_pressure_of_renting (float): Renting preference factor
        cost_comparison_temperature (float): Decision sensitivity parameter
        price_initial_markup (float): Initial price markup factor
        price_decrease_probability (float): Price reduction probability
        price_decrease_mean (float): Price reduction mean
        price_decrease_variance (float): Price reduction variance
        rent_initial_markup (float): Initial rent markup factor
        rent_decrease_probability (float): Rent reduction probability
        rent_decrease_mean (float): Rent reduction mean
        rent_decrease_variance (float): Rent reduction variance
        partial_rent_inflation_indexation (float): Inflation pass-through
        partial_rent_inflation_delay (int): Inflation adjustment lag
    """

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
        self.probability_stay_in_rented_property = probability_stay_in_rented_property
        self.probability_stay_in_owned_property = probability_stay_in_owned_property
        self.maximum_price_income_coefficient = maximum_price_income_coefficient
        self.maximum_price_income_exponent = maximum_price_income_exponent
        self.maximum_price_noise_mean = maximum_price_noise_mean
        self.maximum_price_noise_variance = maximum_price_noise_variance
        self.maximum_rent_income_coefficient = maximum_rent_income_coefficient
        self.maximum_rent_income_exponent = maximum_rent_income_exponent
        self.psychological_pressure_of_renting = psychological_pressure_of_renting
        self.cost_comparison_temperature = cost_comparison_temperature
        self.price_initial_markup = price_initial_markup
        self.price_decrease_probability = price_decrease_probability
        self.price_decrease_mean = price_decrease_mean
        self.price_decrease_variance = price_decrease_variance
        self.rent_initial_markup = rent_initial_markup
        self.rent_decrease_probability = rent_decrease_probability
        self.rent_decrease_mean = rent_decrease_mean
        self.rent_decrease_variance = rent_decrease_variance
        self.partial_rent_inflation_indexation = partial_rent_inflation_indexation
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
        """Calculate property demand and market participation.

        Args:
            housing_data (pd.DataFrame): Property market data
            household_residence_tenure_status (np.ndarray): Current tenure status
            household_income (np.ndarray): Household income
            household_financial_wealth (np.ndarray): Financial assets
            observed_fraction_value_price (np.ndarray): Price/value ratios
            observed_fraction_rent_value (np.ndarray): Rent/value ratios
            expected_hpi_growth (float): Expected house price growth
            assumed_mortgage_maturity (int): Mortgage term length
            rental_income_taxes (float): Tax rate on rental income

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Maximum price willing to pay,
                maximum rent willing to pay, households hoping to move
        """
        pass

    @abstractmethod
    def compute_initial_sale_price(self, property_values: np.ndarray) -> np.ndarray:
        """Calculate initial listing prices.

        Args:
            property_values (np.ndarray): Property values

        Returns:
            np.ndarray: Initial sale prices
        """
        pass

    @abstractmethod
    def compute_updated_sale_price(
        self,
        sale_prices: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        """Update property sale prices.

        Args:
            sale_prices (np.ndarray): Current sale prices
            max_decrease (float): Maximum price reduction

        Returns:
            np.ndarray: Updated sale prices
        """
        pass

    @abstractmethod
    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        """Calculate rental prices.

        Args:
            current_rent (np.ndarray): Current rents
            historic_inflation (np.ndarray): Past inflation rates

        Returns:
            np.ndarray: Updated rental prices
        """
        pass

    @abstractmethod
    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        """Calculate initial rental offers.

        Args:
            property_value (np.ndarray): Property values
            observed_fraction_rent_value (np.ndarray): Rent/value ratios

        Returns:
            np.ndarray: Initial rental prices
        """
        pass

    @abstractmethod
    def compute_offered_rent_for_existing_properties(
        self,
        current_offered_rent: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        """Update rental offers.

        Args:
            current_offered_rent (np.ndarray): Current rental offers
            max_decrease (float): Maximum rent reduction

        Returns:
            np.ndarray: Updated rental prices
        """
        pass


class DefaultHouseholdDemandForProperty(HouseholdDemandForProperty):
    """Default implementation of household property demand behavior.

    Implements property market decisions through:
    - Tenure choice modeling
    - Price/rent willingness calculation
    - Market participation decisions
    - Value adjustments
    """

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
        """Calculate property demand using default behavior.

        Determines demand through:
        - Tenure status evaluation
        - Income/wealth assessment
        - Cost comparison
        - Market participation decisions

        Args:
            housing_data (pd.DataFrame): Property market data
            household_residence_tenure_status (np.ndarray): Current tenure status
            household_income (np.ndarray): Household income
            household_financial_wealth (np.ndarray): Financial assets
            observed_fraction_value_price (np.ndarray): Price/value ratios
            observed_fraction_rent_value (np.ndarray): Rent/value ratios
            expected_hpi_growth (float): Expected house price growth
            assumed_mortgage_maturity (int): Mortgage term length
            rental_income_taxes (float): Tax rate on rental income

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Maximum price willing to pay,
                maximum rent willing to pay, households hoping to move
        """
        # Indices
        ind_in_social_housing = household_residence_tenure_status == -1
        ind_renting = np.array(household_residence_tenure_status == 0)
        ind_renting_not_staying = np.logical_and(
            ind_renting,
            np.random.random(ind_renting.shape[0]) > self.probability_stay_in_rented_property,
        )
        ind_owning = np.array(household_residence_tenure_status == 1)
        ind_owning_not_staying = np.logical_and(
            ind_owning,
            np.random.random(ind_owning.shape[0]) > self.probability_stay_in_owned_property,
        )

        # Decision between renting and owning for households renting or in social housing
        ind_dec = ind_in_social_housing | ind_renting_not_staying | ind_owning_not_staying
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
        max_value_affordable = observed_fraction_value_price[0] * max_amount_pay + observed_fraction_value_price[1]
        max_corresponding_rent = (
            observed_fraction_rent_value[0] * max_value_affordable + observed_fraction_rent_value[1]
        )
        annual_cost_of_renting = 4 * (1 + self.psychological_pressure_of_renting) * max_corresponding_rent
        annual_cost_of_purchasing = (
            4 * np.maximum(0, max_amount_pay - household_financial_wealth[ind_dec]) / assumed_mortgage_maturity
            - ((1 + expected_hpi_growth) ** 4 - 1) * max_value_affordable
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diff_exp = np.exp(self.cost_comparison_temperature * (annual_cost_of_renting - annual_cost_of_purchasing))
            diff_exp[
                np.logical_and(
                    np.isinf(diff_exp),
                    annual_cost_of_renting - annual_cost_of_purchasing < 0,
                )
            ] *= -1
            prob_buying = 1.0 / diff_exp
        ind_deciding_to_buy_rel = np.random.random(prob_buying.shape) < prob_buying
        ind_deciding_to_rent_rel = ~ind_deciding_to_buy_rel
        ind_deciding_to_buy = np.where(ind_dec)[0][ind_deciding_to_buy_rel]
        ind_deciding_to_rent = np.where(ind_dec)[0][ind_deciding_to_rent_rel]

        # The price households are willing to pay
        max_price_willing_to_pay = np.full(household_income.shape, np.nan)
        max_price_willing_to_pay[ind_deciding_to_buy] = max_amount_pay[ind_deciding_to_buy_rel]

        # The rent households are willing to pay
        max_rent_willing_to_pay = np.full(household_income.shape, np.nan)
        max_rent_willing_to_pay[ind_deciding_to_rent] = (
            self.maximum_rent_income_coefficient
            * household_income[ind_deciding_to_rent] ** self.maximum_rent_income_exponent
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
        """Calculate initial sale prices using default behavior.

        Applies markup to property values to determine listing prices.

        Args:
            property_values (np.ndarray): Property values

        Returns:
            np.ndarray: Initial sale prices
        """
        return (1 + self.price_initial_markup) * property_values

    def compute_updated_sale_price(
        self,
        sale_prices: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        """Update sale prices using default behavior.

        Adjusts prices based on:
        - Random price reductions
        - Maximum decrease limits

        Args:
            sale_prices (np.ndarray): Current sale prices
            max_decrease (float): Maximum price reduction

        Returns:
            np.ndarray: Updated sale prices
        """

        new_sale_prices = sale_prices.copy()
        properties_with_reduced_price = np.random.random(sale_prices.shape) < self.price_decrease_probability
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
        """Calculate rental prices using default behavior.

        Updates rents based on:
        - Historical inflation
        - Partial indexation
        - Adjustment delays

        Args:
            current_rent (np.ndarray): Current rents
            historic_inflation (np.ndarray): Past inflation rates

        Returns:
            np.ndarray: Updated rental prices
        """
        return (
            1
            + np.maximum(
                0.0,
                self.partial_rent_inflation_indexation * historic_inflation[-self.partial_rent_inflation_delay],
            )
        ) * current_rent

    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        """Calculate initial rental offers using default behavior.

        Determines rent based on:
        - Property values
        - Rent/value ratios
        - Initial markup

        Args:
            property_value (np.ndarray): Property values
            observed_fraction_rent_value (np.ndarray): Rent/value ratios

        Returns:
            np.ndarray: Initial rental prices
        """
        return (1 + self.rent_initial_markup) * (
            observed_fraction_rent_value[0] * property_value + observed_fraction_rent_value[1]
        )

    def compute_offered_rent_for_existing_properties(
        self,
        current_offered_rent: np.ndarray,
        max_decrease: float = 0.2,
    ) -> np.ndarray:
        new_offered_rent = current_offered_rent.copy()
        properties_with_reduced_rent = np.random.random(current_offered_rent.shape) < self.rent_decrease_probability
        new_offered_rent[properties_with_reduced_rent] *= 1 - np.exp(
            np.random.normal(
                self.rent_decrease_mean,
                self.rent_decrease_variance**0.5,
                np.sum(properties_with_reduced_rent),
            )
        )
        return new_offered_rent
