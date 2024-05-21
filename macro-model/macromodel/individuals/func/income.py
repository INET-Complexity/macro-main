import numpy as np

from abc import abstractmethod, ABC

from macromodel.individuals.individual_properties import ActivityStatus


class IncomeSetter(ABC):
    @abstractmethod
    def compute_expected_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        expected_firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        expected_bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        expected_inflation: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        pass

    def compute_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        pass


class DefaultIncomeSetter(IncomeSetter):
    def compute_expected_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        expected_firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        expected_bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        expected_inflation: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        income = np.zeros_like(current_individual_activity_status)

        # Employed individuals
        emp_ind = current_individual_activity_status == ActivityStatus.EMPLOYED
        income[emp_ind] = (1 + expected_inflation) * cpi * current_wage[emp_ind]

        # Unemployed individuals
        unemp_ind = current_individual_activity_status == ActivityStatus.UNEMPLOYED
        income[unemp_ind] = 0.0

        # Not-economically active individuals
        nea_ind = current_individual_activity_status == ActivityStatus.NOT_ECONOMICALLY_ACTIVE
        income[nea_ind] = 0.0

        # Firm investors
        firm_inv_ind = current_individual_activity_status == ActivityStatus.FIRM_INVESTOR
        income[firm_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, expected_firm_profits[corr_invested_firms[firm_inv_ind]])
        )

        # Bank investors
        bank_inv_ind = current_individual_activity_status == ActivityStatus.BANK_INVESTOR
        income[bank_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, expected_bank_profits[corr_invested_banks[bank_inv_ind]])
        )
        return (1 + expected_inflation) * cpi * individual_social_benefits + income

    def compute_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        income = np.zeros_like(current_individual_activity_status)

        # Employed individuals
        emp_ind = current_individual_activity_status == ActivityStatus.EMPLOYED
        income[emp_ind] = cpi * current_wage[emp_ind]

        # Unemployed individuals
        unemp_ind = current_individual_activity_status == ActivityStatus.UNEMPLOYED
        income[unemp_ind] = 0.0

        # Not-economically active individuals
        nea_ind = current_individual_activity_status == ActivityStatus.NOT_ECONOMICALLY_ACTIVE
        income[nea_ind] = 0.0

        # Firm investors
        firm_inv_ind = current_individual_activity_status == ActivityStatus.FIRM_INVESTOR
        income[firm_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, firm_profits[corr_invested_firms[firm_inv_ind].astype(int)])
        )

        # Bank investors
        bank_inv_ind = current_individual_activity_status == ActivityStatus.BANK_INVESTOR
        income[bank_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, bank_profits[corr_invested_banks[bank_inv_ind].astype(int)])
        )

        return cpi * individual_social_benefits + income
