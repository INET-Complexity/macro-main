from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numba import boolean, float64, int64, njit


class HouseholdConsumption(ABC):
    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
        elasticity_of_substitution: float = 1.0,  # Ignored by default consumption
    ):
        self.consumption_smoothing_fraction = consumption_smoothing_fraction
        self.consumption_smoothing_window = consumption_smoothing_window
        self.minimum_consumption_fraction = minimum_consumption_fraction
        # Note: elasticity_of_substitution is ignored in default consumption

    @abstractmethod
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        extra_sectoral_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pass


class DefaultHouseholdConsumption(HouseholdConsumption):
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,  # Ignored in default consumption
        initial_prices: np.ndarray = None,  # Ignored in default consumption
        taxes: np.ndarray = None,  # Ignored in default consumption
        initial_taxes: np.ndarray = None,  # Ignored in default consumption
        bundle_matrix: np.ndarray = None,  # Ignored in default consumption
        extra_sectoral_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate target consumption using default behavior.

        Determines consumption targets based on:
        - Income after savings
        - Historical consumption patterns
        - Minimum consumption thresholds
        - Industry allocation weights
        - Tax considerations

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            historic_consumption_sum (np.ndarray): Past consumption totals
            saving_rates (np.ndarray): Household saving rates
            income (np.ndarray): Household income
            household_benefits (np.ndarray): Social benefits received
            consumption_weights (np.ndarray): Industry consumption shares
            consumption_weights_by_income (np.ndarray): Income-based weights
            exogenous_total_consumption (np.ndarray): External consumption target
            current_time (int): Current period
            take_consumption_weights_by_income_quantile (bool): Use income quintiles
            tau_vat (float): Value added tax rate

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        if extra_sectoral_taxes is None:
            extra_sectoral_taxes = np.zeros(consumption_weights.shape[0])
        return self._compute_target_consumption(
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            take_consumption_weights_by_income_quantile=take_consumption_weights_by_income_quantile,
            tau_vat=tau_vat,
            consumption_smoothing_window=self.consumption_smoothing_window,
            consumption_smoothing_fraction=self.consumption_smoothing_fraction,
            minimum_consumption_fraction=self.minimum_consumption_fraction,
            extra_sectoral_taxes=extra_sectoral_taxes,
        )

    @staticmethod
    # @njit(
    #     float64[:, :](
    #         float64[:, :],  # historic_consumption_sum
    #         float64[:],  # saving_rates
    #         float64[:],  # income
    #         float64[:],  # household_benefits
    #         float64[:],  # consumption_weights
    #         float64[:, :],  # consumption_weights_by_income
    #         boolean,  # take_consumption_weights_by_income_quantile
    #         float64,  # tau_vat
    #         int64,  # consumption_smoothing_window
    #         float64,  # consumption_smoothing_fraction
    #         float64,  # minimum_consumption_fraction
    #     ),
    #     cache=True,
    # )
    @njit(cache=True)
    def _compute_target_consumption(
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,  # noqa
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        consumption_smoothing_window: int,
        consumption_smoothing_fraction: float,
        minimum_consumption_fraction: float,
        extra_sectoral_taxes: np.ndarray,
    ) -> np.ndarray:
        smoothing_window = min(consumption_smoothing_window, len(historic_consumption_sum))
        # target_consumption = np.zeros((len(income), len(consumption_weights)))
        # if take_consumption_weights_by_income_quantile:
        #     pass
        #     """
        #     quintiles = partition_into_quintiles(income)
        #     historic_consumption_sum = np.array(historic_consumption)[1:][-smoothing_window:].sum(axis=0)
        #     for q in range(5):
        #         cons = (
        #             consumption_weights_by_income[:, q]
        #             if take_consumption_weights_by_income_quantile
        #             else consumption_weights
        #         )
        #         ind = np.where(quintiles == q)[0]
        #         target_consumption[ind] = (
        #             1.0
        #             / (1 + tau_vat)
        #             * np.outer(
        #                 cons,
        #                 np.maximum(
        #                     minimum_consumption_fraction * (1 - saving_rates[ind]) * household_benefits[ind],
        #                     (1 - saving_rates[ind]) * income[ind],
        #                     consumption_smoothing_fraction
        #                     * (1 + tau_vat)
        #                     * (1 / smoothing_window)
        #                     * historic_consumption_sum[ind],
        #                 ),
        #             ).T
        #         )
        #     return np.maximum(0.0, target_consumption)
        #     """
        # else:
        target_consumption = (
            1.0
            / (1 + tau_vat + extra_sectoral_taxes)
            * np.outer(
                consumption_weights,
                np.maximum(
                    minimum_consumption_fraction * (1 - saving_rates) * household_benefits,
                    (1 - saving_rates) * income,
                    consumption_smoothing_fraction
                    * (1 + tau_vat + extra_sectoral_taxes)
                    * (1 / smoothing_window)
                    * historic_consumption_sum[1:][-smoothing_window:].sum(axis=0),
                ),
            ).T
        )
        return np.maximum(0.0, target_consumption)


class CESHouseholdConsumption(HouseholdConsumption):
    """CES (Constant Elasticity of Substitution) household consumption implementation.

    Implements consumption decisions with substitution within bundles based on:
    - CES utility function with elasticity of substitution
    - Dynamic consumption shares based on relative prices and taxes
    - Bundle-based substitution patterns
    - Initial consumption weights as preference parameters
    """

    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
        elasticity_of_substitution: float = 1.0,
    ):
        super().__init__(
            consumption_smoothing_fraction,
            consumption_smoothing_window,
            minimum_consumption_fraction,
        )
        self.elasticity_of_substitution = elasticity_of_substitution

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,
        initial_prices: np.ndarray = None,
        taxes: np.ndarray = None,
        initial_taxes: np.ndarray = None,
        bundle_matrix: np.ndarray = None,
        extra_sectoral_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate target consumption using CES substitution within bundles.

        Determines consumption based on:
        - CES utility function with substitution elasticity
        - Dynamic consumption shares based on relative prices and taxes
        - Bundle-based substitution patterns
        - Initial consumption preferences

        Args:
            All standard args plus:
            prices (np.ndarray): Current prices by industry
            initial_prices (np.ndarray): Initial prices by industry
            taxes (np.ndarray): Current tax rates by industry
            initial_taxes (np.ndarray): Initial tax rates by industry
            bundle_matrix (np.ndarray): Bundle weight matrix (n_industries, n_bundles)

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        # If no substitution data provided, fall back to default behavior
        if any(x is None for x in [prices, initial_prices, taxes, initial_taxes, bundle_matrix]):
            return self._compute_target_consumption_default(
                historic_consumption_sum,
                saving_rates,
                income,
                household_benefits,
                consumption_weights,
                consumption_weights_by_income,
                take_consumption_weights_by_income_quantile,
                tau_vat,
            )

        # Compute CES consumption shares with substitution
        ces_weights = self._compute_ces_weights(
            consumption_weights, prices, initial_prices, taxes, initial_taxes, bundle_matrix
        )

        return self._compute_target_consumption_ces(
            historic_consumption_sum,
            saving_rates,
            income,
            household_benefits,
            ces_weights,
            consumption_weights_by_income,
            take_consumption_weights_by_income_quantile,
            tau_vat,
            taxes,
        )

    def _compute_target_consumption_default(
        self,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        """Default consumption calculation when substitution data is unavailable."""
        return DefaultHouseholdConsumption._compute_target_consumption(
            historic_consumption_sum,
            saving_rates,
            income,
            household_benefits,
            consumption_weights,
            consumption_weights_by_income,
            take_consumption_weights_by_income_quantile,
            tau_vat,
            self.consumption_smoothing_window,
            self.consumption_smoothing_fraction,
            self.minimum_consumption_fraction,
        )

    def _compute_ces_weights(
        self,
        initial_weights: np.ndarray,
        prices: np.ndarray,
        initial_prices: np.ndarray,
        taxes: np.ndarray,
        initial_taxes: np.ndarray,
        bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute CES consumption weights with substitution within bundles.

        Implements the formula:
        c_i(t) = c_i(0) * ((1+τ_i(0))/(1+τ_i(t)))^σ * p_i(t)^(-σ) * bundle_normalization
        """
        sigma = self.elasticity_of_substitution

        # Compute price and tax ratios
        price_ratio = prices / initial_prices
        tax_ratio = (1 + initial_taxes) / (1 + taxes)

        # Compute individual substitution effects
        substitution_factor = (tax_ratio**sigma) * (price_ratio ** (-sigma))

        # Apply substitution within bundles
        ces_weights = np.zeros_like(initial_weights)
        n_bundles = bundle_matrix.shape[1]

        for bundle_idx in range(n_bundles):
            # Industries in this bundle (bundle_matrix is n_industries x n_bundles)
            bundle_mask = bundle_matrix[:, bundle_idx] > 0

            if not np.any(bundle_mask):
                continue

            # Initial bundle allocation
            bundle_initial_weights = initial_weights[bundle_mask]
            bundle_total = np.sum(bundle_initial_weights)

            if bundle_total == 0:
                continue

            # Apply CES substitution within bundle
            bundle_substitution = substitution_factor[bundle_mask] * bundle_initial_weights
            bundle_substitution_total = np.sum(bundle_substitution)

            # Normalize to maintain bundle total
            if bundle_substitution_total > 0:
                ces_weights[bundle_mask] = bundle_substitution * (bundle_total / bundle_substitution_total)
            else:
                ces_weights[bundle_mask] = bundle_initial_weights

        return ces_weights

    def _compute_target_consumption_ces(
        self,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        ces_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        taxes: np.ndarray,
    ) -> np.ndarray:
        """Compute target consumption using CES-adjusted weights."""
        smoothing_window = min(self.consumption_smoothing_window, len(historic_consumption_sum))
        target_consumption = (
            1.0
            / (1 + tau_vat) #+ taxes)   # why not included?
            * np.outer(
                ces_weights,
                np.maximum(
                    self.minimum_consumption_fraction * (1 - saving_rates) * household_benefits,
                    (1 - saving_rates) * income,
                    self.consumption_smoothing_fraction
                    * (1 + tau_vat) #+ taxes)   # why not included?
                    * (1 / smoothing_window)
                    * historic_consumption_sum[1:][-smoothing_window:].sum(axis=0),
                ),
            ).T
        )
        return np.maximum(0.0, target_consumption)


class ExogenousHouseholdConsumption(HouseholdConsumption):
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,  # Ignored in exogenous consumption
        initial_prices: np.ndarray = None,  # Ignored in exogenous consumption
        taxes: np.ndarray = None,  # Ignored in exogenous consumption
        initial_taxes: np.ndarray = None,  # Ignored in exogenous consumption
        bundle_matrix: np.ndarray = None,  # Ignored in exogenous consumption
        extra_sectoral_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate target consumption using exogenous targets.

        Determines consumption based on:
        - External consumption targets
        - Price level changes
        - Income-based allocation
        - Tax adjustments

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            historic_consumption_sum (np.ndarray): Past consumption totals
            saving_rates (np.ndarray): Household saving rates
            income (np.ndarray): Household income
            household_benefits (np.ndarray): Social benefits received
            consumption_weights (np.ndarray): Industry consumption shares
            consumption_weights_by_income (np.ndarray): Income-based weights
            exogenous_total_consumption (np.ndarray): External consumption target
            current_time (int): Current period
            take_consumption_weights_by_income_quantile (bool): Use income quintiles
            tau_vat (float): Value added tax rate

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        if extra_sectoral_taxes is None:
            extra_sectoral_taxes = np.zeros(consumption_weights.shape[0])
        target_consumption = np.maximum(
            0.0,
            (
                1.0
                / (1 + tau_vat + extra_sectoral_taxes)
                * np.outer(
                    consumption_weights,
                    (1 - saving_rates) * income,
                ).T
            ),
        )
        return (
            (1 + expected_inflation)
            * current_cpi
            / initial_cpi
            * 1.0
            / (1 + tau_vat + extra_sectoral_taxes)
            * exogenous_total_consumption[current_time]
            * target_consumption
            / target_consumption.sum()
        )
