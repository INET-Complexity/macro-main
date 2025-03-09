from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class FirmWageSetter(ABC):
    """Abstract base class for determining firms' wage-setting strategies.

    This class defines strategies for calculating wages based on:
    - Labor market conditions (tightness, supply/demand)
    - Worker productivity and effort
    - Tax considerations
    - Historical wage levels and adjustments

    The wage setting process considers:
    - Market tightness markups based on hiring success
    - Work effort incentives
    - Tax-adjusted gross/net wage conversions

    Attributes:
        labour_market_tightness_markup_scale (float): Scale factor for wage
            adjustments based on labor market tightness
        markup_time_span (int): Number of periods to consider when calculating
            labor market tightness markup
        max_increase_in_work_effort (float): Maximum allowed increase in
            work effort-based wage adjustments
    """

    def __init__(
        self,
        labour_market_tightness_markup_scale: float,
        markup_time_span: int,
        max_increase_in_work_effort: float,
    ):
        """Initialize the wage setter with markup parameters.

        Args:
            labour_market_tightness_markup_scale (float): Scale factor for
                wage adjustments based on labor market tightness
            markup_time_span (int): Number of periods to consider when
                calculating labor market tightness markup
            max_increase_in_work_effort (float): Maximum allowed increase
                in work effort-based wage adjustments
        """
        self.labour_market_tightness_markup_scale = labour_market_tightness_markup_scale
        self.markup_time_span = markup_time_span
        self.max_increase_in_work_effort = max_increase_in_work_effort

    @abstractmethod
    def compute_wage_tightness_markup(
        self,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
    ) -> np.ndarray:
        """Calculate wage markup based on labor market tightness.

        Uses historical data on desired vs. realized labor inputs to
        determine wage adjustments that reflect labor market conditions.

        Args:
            historic_desired_labour_inputs (list[np.ndarray]): Time series
                of desired labor inputs by firm
            historic_realised_labour_inputs (list[np.ndarray]): Time series
                of actually achieved labor inputs by firm

        Returns:
            np.ndarray: Wage markup factors by firm based on labor market tightness
        """
        pass

    @abstractmethod
    def set_employee_income(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        current_individual_stating_new_job: np.ndarray,
        current_employee_income: np.ndarray,
        current_individual_offered_wage: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
        initial_wage_per_capita: np.ndarray,
        current_wage_per_capita: np.ndarray,
        current_labour_productivity_factor: np.ndarray,
        prev_labour_productivity_factor: np.ndarray,
        current_wage_tightness_markup: np.ndarray,
        estimated_ppi_inflation: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> np.ndarray:
        """Set employee incomes considering all relevant factors.

        Determines wages based on:
        - Individual labor inputs and productivity
        - Market conditions and tightness
        - Tax rates and social insurance
        - New vs. existing employee status
        - Production constraints and targets

        Args:
            corresponding_firm (np.ndarray): Firm ID for each employee
            current_individual_labour_inputs (np.ndarray): Labor input by employee
            current_individual_stating_new_job (np.ndarray): New job indicators
            current_employee_income (np.ndarray): Current income levels
            current_individual_offered_wage (np.ndarray): Offered wages for new hires
            current_target_production (np.ndarray): Production targets by firm
            current_limiting_intermediate_inputs (np.ndarray): Input constraints
            current_limiting_capital_inputs (np.ndarray): Capital constraints
            labour_inputs_from_employees (np.ndarray): Employee labor contributions
            industry_labour_productivity_by_firm (np.ndarray): Productivity metrics
            initial_wage_per_capita (np.ndarray): Base wage levels
            current_wage_per_capita (np.ndarray): Current wage levels
            current_labour_productivity_factor (np.ndarray): Current productivity
            prev_labour_productivity_factor (np.ndarray): Previous productivity
            current_wage_tightness_markup (np.ndarray): Market condition adjustments
            estimated_ppi_inflation (float): Expected price inflation
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate

        Returns:
            np.ndarray: Updated employee incomes
        """
        pass

    @abstractmethod
    def get_offered_wage_given_labour_inputs_function(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
        initial_wage_per_capita: np.ndarray,
        current_wage_per_capita: np.ndarray,
        current_labour_productivity_factor: np.ndarray,
        prev_labour_productivity_factor: np.ndarray,
        current_wage_tightness_markup: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
        unemployment_benefits_by_individual: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        """Create a function that calculates offered wages based on labor inputs.

        Returns a callable that firms can use to determine appropriate wage
        offers for different levels of labor input, considering:
        - Current market conditions
        - Productivity factors
        - Tax rates
        - Unemployment benefits (reservation wages)

        Args:
            corresponding_firm (np.ndarray): Firm ID for each employee
            current_individual_labour_inputs (np.ndarray): Labor input by employee
            previous_employee_income (np.ndarray): Previous income levels
            current_target_production (np.ndarray): Production targets by firm
            current_limiting_intermediate_inputs (np.ndarray): Input constraints
            current_limiting_capital_inputs (np.ndarray): Capital constraints
            industry_labour_productivity_by_firm (np.ndarray): Productivity metrics
            initial_wage_per_capita (np.ndarray): Base wage levels
            current_wage_per_capita (np.ndarray): Current wage levels
            current_labour_productivity_factor (np.ndarray): Current productivity
            prev_labour_productivity_factor (np.ndarray): Previous productivity
            current_wage_tightness_markup (np.ndarray): Market condition adjustments
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate
            unemployment_benefits_by_individual (float): Minimum wage floor

        Returns:
            Callable[[int, float | np.ndarray], float | np.ndarray]: Function that
                takes firm ID and labor inputs and returns offered wages
        """
        pass


class WorkEffortFirmWageSetter(FirmWageSetter):
    """Implementation of wage setting based on work effort considerations.

    This class implements a wage-setting strategy that:
    1. Adjusts wages based on labor market tightness
    2. Considers productivity changes
    3. Ensures wages exceed unemployment benefits
    4. Accounts for tax implications

    The approach aims to:
    - Incentivize optimal work effort
    - Respond to labor market conditions
    - Maintain fair compensation relative to productivity
    - Ensure tax compliance
    """

    def compute_wage_tightness_markup(
        self,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
    ) -> np.ndarray:
        """Calculate wage markup based on historical labor market tightness.

        Computes markup as the weighted average of past hiring shortfalls:
        - Zero markup if scale parameter is zero
        - Otherwise, considers up to markup_time_span periods
        - Weights recent periods equally within the span

        Args:
            historic_desired_labour_inputs (list[np.ndarray]): Time series
                of desired labor inputs by firm
            historic_realised_labour_inputs (list[np.ndarray]): Time series
                of actually achieved labor inputs by firm

        Returns:
            np.ndarray: Wage markup factors reflecting hiring difficulty
        """
        if self.labour_market_tightness_markup_scale == 0.0:
            return np.zeros(historic_desired_labour_inputs[0].shape)

        rel_failures = np.zeros_like(historic_desired_labour_inputs[0])
        for t in range(
            1,
            min(len(historic_desired_labour_inputs), self.markup_time_span + 1),
        ):
            rel_failures += np.maximum(
                0.0,
                np.divide(
                    (historic_desired_labour_inputs[-t] - historic_realised_labour_inputs[-t]),
                    historic_desired_labour_inputs[-t],
                    out=np.zeros(historic_desired_labour_inputs[0].shape),
                    where=historic_desired_labour_inputs[-t] != 0.0,
                ),
            )
        return self.labour_market_tightness_markup_scale * 1.0 / self.markup_time_span * rel_failures

    def set_employee_income(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        current_individual_stating_new_job: np.ndarray,
        current_employee_income: np.ndarray,
        current_individual_offered_wage: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
        initial_wage_per_capita: np.ndarray,
        current_wage_per_capita: np.ndarray,
        current_labour_productivity_factor: np.ndarray,
        prev_labour_productivity_factor: np.ndarray,
        current_wage_tightness_markup: np.ndarray,
        estimated_ppi_inflation: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> np.ndarray:
        """Set employee incomes based on work effort and market conditions.

        Calculates wages considering:
        1. Base wages adjusted for productivity changes
        2. Market tightness markup
        3. Tax implications for gross/net conversion
        4. Different treatment for new vs. existing employees

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Updated employee incomes adjusted for all factors
        """
        tax = (1.0 + employer_social_insurance_tax) / (
            1 - employee_social_insurance_tax - income_taxes * (1 - employee_social_insurance_tax)
        )

        """
        new_individual_wages = np.zeros_like(current_employee_income)
        for firm_ind in range(current_target_production.shape[0]):
            ind = np.where(corresponding_firm == firm_ind)[0]
            if current_individual_labour_inputs[ind].sum() > 0.0:
                ind_same_firm = ind[~current_individual_stating_new_job[ind]]
                ind_new_firm = ind[current_individual_stating_new_job[ind]]
                new_individual_wages[ind_same_firm] = (
                    (1 + current_wage_tightness_markup[firm_ind])
                    * current_labour_productivity_factor[firm_ind]
                    / prev_labour_productivity_factor[firm_ind]
                    * current_employee_income[ind_same_firm]
                )
                new_individual_wages[ind_new_firm] = current_individual_offered_wage[ind_new_firm]
        return new_individual_wages
        """
        scaled_real_wages_by_individual = np.zeros_like(current_employee_income)
        emp_ind = corresponding_firm >= 0
        scaled_real_wages = (
            (1 + current_wage_tightness_markup) * current_labour_productivity_factor * initial_wage_per_capita
        )
        scaled_real_wages_by_individual[emp_ind] = scaled_real_wages[corresponding_firm[emp_ind]]
        return scaled_real_wages_by_individual / tax

    def get_offered_wage_given_labour_inputs_function(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
        initial_wage_per_capita: np.ndarray,
        current_wage_per_capita: np.ndarray,
        current_labour_productivity_factor: np.ndarray,
        prev_labour_productivity_factor: np.ndarray,
        current_wage_tightness_markup: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
        unemployment_benefits_by_individual: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        """Create a function for calculating wage offers based on work effort.

        Returns a callable that:
        1. Calculates base wage from historical averages
        2. Adjusts for productivity changes
        3. Applies market tightness markup
        4. Ensures wages exceed unemployment benefits

        Args:
            [same as parent class]

        Returns:
            Callable[[int, float | np.ndarray], float | np.ndarray]: Function that
                calculates appropriate wage offers given firm ID and labor inputs
        """
        total_real_wages = np.bincount(
            corresponding_firm,
            weights=previous_employee_income,
            minlength=current_target_production.shape[0],
        )
        total_labour_inputs = np.bincount(
            corresponding_firm,
            weights=current_individual_labour_inputs,
            minlength=current_target_production.shape[0],
        )
        new_individual_wages = (
            (1 + current_wage_tightness_markup)
            * current_labour_productivity_factor
            / prev_labour_productivity_factor
            * total_real_wages
            / total_labour_inputs
        )

        def f(firm_id: int, labour_inputs: float | np.ndarray) -> float | np.ndarray:
            """Calculate wage offer for given firm and labor inputs.

            Args:
                firm_id (int): ID of the firm making the offer
                labour_inputs (float | np.ndarray): Proposed labor input level(s)

            Returns:
                float | np.ndarray: Wage offer(s) that exceed unemployment benefits
            """
            return np.maximum(
                unemployment_benefits_by_individual,
                labour_inputs * new_individual_wages[firm_id],
            )

        return f
