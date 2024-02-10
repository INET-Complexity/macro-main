import numpy as np
import pandas as pd
import warnings
from abc import abstractmethod, ABC
from typing import Tuple


class HouseholdDemandForProperty(ABC):
    def __init__(
        self,
        probability_stay_in_rented_property: float,
        probability_stay_in_owned_property: float,
        maximum_price_income_coefficient: float,
        maximum_price_income_exponent: float,
        maximum_price_noise_std: float,
        psychological_pressure_of_renting: float,
        cost_comparison_temperature: float,
        rental_yield_btl_temperature: float,
    ):
        self.probability_stay_in_rented_property = probability_stay_in_rented_property
        self.probability_stay_in_owned_property = probability_stay_in_owned_property
        self.maximum_price_income_coefficient = maximum_price_income_coefficient
        self.maximum_price_income_exponent = maximum_price_income_exponent
        self.maximum_price_noise_std = maximum_price_noise_std
        self.psychological_pressure_of_renting = psychological_pressure_of_renting
        self.cost_comparison_temperature = cost_comparison_temperature
        self.rental_yield_btl_temperature = rental_yield_btl_temperature

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
        ind_renting = household_residence_tenure_status == 0
        ind_renting_not_staying = ind_renting & (
            np.random.random(ind_renting.shape[0]) > self.probability_stay_in_rented_property
        )
        ind_owning = household_residence_tenure_status == 1
        ind_owning_not_staying = ind_owning & (
            np.random.random(ind_owning.shape[0]) > self.probability_stay_in_owned_property
        )

        # Decision between renting and owning for households renting or in social housing
        ind_dec = ind_in_social_housing | ind_renting_not_staying | ind_owning_not_staying
        max_amount_pay = (
            self.maximum_price_income_coefficient
            * household_income[ind_dec] ** self.maximum_price_income_exponent
            * np.exp(np.random.normal(0.0, self.maximum_price_noise_std, np.sum(ind_dec)))
        )
        max_value_affordable = observed_fraction_value_price[0] * max_amount_pay + observed_fraction_value_price[1]
        max_corresponding_rent = (
            observed_fraction_rent_value[0] * max_value_affordable + observed_fraction_rent_value[1]
        )
        annual_cost_of_renting = 12 * (1 + self.psychological_pressure_of_renting) * max_corresponding_rent
        annual_cost_of_purchasing = (
            12 * np.maximum(0, max_value_affordable - household_financial_wealth[ind_dec]) / assumed_mortgage_maturity
            - (1 + 12 * expected_hpi_growth) * max_value_affordable
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

        # Households owning property may decide to buy additional property to let out
        max_amount_pay_btl = (
            self.maximum_price_income_coefficient
            * household_income[ind_owning] ** self.maximum_price_income_exponent
            * np.exp(np.random.normal(0.0, self.maximum_price_noise_std, np.sum(ind_owning)))
        )
        max_value_affordable_btl = (
            observed_fraction_value_price[0] * max_amount_pay_btl + observed_fraction_value_price[1]
        )
        expected_rental_yield = (1 - rental_income_taxes) * (
            observed_fraction_rent_value[0] * max_value_affordable_btl + observed_fraction_rent_value[1]
        ) - (
            np.maximum(0, max_value_affordable_btl - household_financial_wealth[ind_owning]) / assumed_mortgage_maturity
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp_yield = np.exp(-self.rental_yield_btl_temperature * expected_rental_yield)
            exp_yield[np.logical_and(np.isinf(exp_yield), expected_rental_yield < 0)] *= -1
            prob_btl = 1.0 / (1 - exp_yield)
        ind_btl_rel = np.random.random(prob_btl.shape) < prob_btl
        ind_btl = np.where(ind_owning)[0][ind_btl_rel]

        # The price households are willing to pay
        max_price_willing_to_pay = np.full(household_income.shape, np.nan)
        max_price_willing_to_pay[ind_deciding_to_buy] = max_amount_pay[ind_deciding_to_buy_rel]
        max_price_willing_to_pay[ind_btl] = max_amount_pay_btl[ind_btl_rel]

        # The rent households are willing to pay
        max_rent_willing_to_pay = np.full(household_income.shape, np.nan)
        max_rent_willing_to_pay[ind_deciding_to_rent] = (
            (
                self.maximum_price_income_coefficient
                * household_income[ind_deciding_to_rent] ** self.maximum_price_income_exponent
            )
            * observed_fraction_value_price[1]
            + observed_fraction_value_price[0]
        ) * observed_fraction_rent_value[1] + observed_fraction_rent_value[0]

        return max_price_willing_to_pay, max_rent_willing_to_pay, ind_owning_not_staying
