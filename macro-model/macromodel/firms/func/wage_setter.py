import numpy as np

from abc import abstractmethod, ABC

from typing import Callable


class FirmWageSetter(ABC):
    def __init__(
        self,
        labour_market_tightness_markup_scale: float,
        markup_time_span: int,
        max_increase_in_work_effort: float,
    ):
        self.labour_market_tightness_markup_scale = labour_market_tightness_markup_scale
        self.markup_time_span = markup_time_span
        self.max_increase_in_work_effort = max_increase_in_work_effort

    @abstractmethod
    def compute_wage_tightness_markup(
        self,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
    ) -> np.ndarray:
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
        pass


class WorkEffortFirmWageSetter(FirmWageSetter):
    def compute_wage_tightness_markup(
        self,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
    ) -> np.ndarray:
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

        # Create a function
        def f(firm_id: int, labour_inputs: float | np.ndarray) -> float | np.ndarray:
            return np.maximum(
                unemployment_benefits_by_individual,
                labour_inputs * new_individual_wages[firm_id],
            )

        return f
