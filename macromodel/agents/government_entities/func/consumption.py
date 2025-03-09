"""Government consumption determination strategies.

This module implements various approaches for determining government
consumption targets, including:
- Autoregressive forecasting
- Constant growth assumptions
- Exogenous consumption paths

The consumption strategies consider:
- Historical consumption patterns
- Price level adjustments
- Growth expectations
- Inflation expectations
- Financial constraints
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import ImplementedAutoregForecaster  # noqa
from macromodel.forecaster.forecaster import ManualAutoregForecaster


class GovernmentConsumptionSetter(ABC):
    """Abstract base class for government consumption strategies.

    This class defines the interface for determining government
    consumption targets based on various factors including:
    - Historical consumption patterns
    - Economic conditions
    - Price level changes
    - Growth expectations
    - Policy objectives

    The consumption setting process considers:
    - Consistency requirements
    - Default growth assumptions
    - Buffer periods for forecasting
    - Price level adjustments
    """

    def __init__(
        self,
        consistency: float,
        default_growth: Optional[float] = None,
    ):
        """Initialize consumption setter.

        Args:
            consistency (float): Must be 0.0 or 1.0, determines whether
                to use consistent forecasting (1.0) or period-by-period
                adjustments (0.0)
            default_growth (float, optional): Default growth rate to use
                when historical data is unavailable
        """
        assert consistency == 1.0 or consistency == 0.0
        self.consistency = consistency
        self.default_growth = default_growth
        self.fixed_total_government_consumption = None
        self.buffer = 20

    @abstractmethod
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate target government consumption.

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray): Historical total
                consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths

        Returns:
            np.ndarray: Target consumption by industry
        """
        pass


class AutoregressiveGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Autoregressive consumption target determination.

    This class implements consumption targeting based on:
    - Autoregressive forecasting of total consumption
    - Price level adjustments
    - Consistency requirements
    - Industry-specific allocation

    The approach provides:
    - Data-driven consumption targets
    - Price-adjusted spending
    - Consistent or period-by-period forecasting
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
        log_it: bool = True,
    ) -> np.ndarray:
        """Calculate consumption targets using autoregression.

        Uses autoregressive forecasting to determine targets based on:
        - Historical consumption patterns
        - Price level changes
        - Consistency requirements
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray): Historical total
                consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths
            log_it (bool, optional): Whether to use log transformation
                in forecasting

        Returns:
            np.ndarray: Target consumption by industry
        """
        if historic_total_consumption[-1] == 0.0:
            return np.zeros(previous_desired_government_consumption.shape)

        # Fitting based on target consumption
        if self.consistency == 1.0:
            if (
                self.fixed_total_government_consumption is None
                or len(self.fixed_total_government_consumption) < current_time
            ):
                if log_it:
                    self.fixed_total_government_consumption = np.exp(
                        ManualAutoregForecaster().forecast(
                            data=np.log(historic_total_consumption),
                            t=max(current_time + self.buffer, current_time),
                            assume_zero_noise=assume_zero_noise,
                        )
                    )
                else:
                    self.fixed_total_government_consumption = ManualAutoregForecaster().forecast(
                        data=historic_total_consumption,
                        t=max(current_time + self.buffer, current_time),
                        assume_zero_noise=assume_zero_noise,
                    )
            consumption = self.fixed_total_government_consumption[current_time - 1]

        # Fitting based on historic consumption
        else:
            consumption = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_consumption),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * consumption
            * previous_desired_government_consumption
            / previous_desired_government_consumption.sum(),
        )


class ConstantGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Constant growth consumption target determination.

    This class implements consumption targeting based on:
    - Fixed growth rate assumptions
    - Price level adjustments
    - Historical growth estimation
    - Industry-specific allocation

    The approach provides:
    - Simple growth-based targets
    - Price-adjusted spending
    - Stable consumption paths
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using constant growth.

        Determines targets based on:
        - Fixed or estimated growth rate
        - Price level changes
        - Previous consumption levels
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray, optional): Historical
                total consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths

        Returns:
            np.ndarray: Target consumption by industry
        """
        if historic_total_consumption is None:
            return np.maximum(
                0.0,
                (1 + expected_inflation) * (1 + self.default_growth) * previous_desired_government_consumption,
            )
        if self.default_growth is None:
            self.default_growth = np.mean(
                np.log(
                    historic_total_consumption[1 : -current_time - 1]
                    / historic_total_consumption[0 : -current_time - 2]
                )
            )

        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * (1 + self.default_growth)
            * previous_desired_government_consumption,
        )


class AutoregressiveGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Autoregressive growth consumption target determination.

    This class implements consumption targeting based on:
    - Autoregressive forecasting of growth rates
    - Price level adjustments
    - Consistency requirements
    - Industry-specific allocation

    The approach provides:
    - Data-driven growth forecasts
    - Price-adjusted spending
    - Consistent or period-by-period forecasting
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
        log_it: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using growth autoregression.

        Uses autoregressive forecasting of growth rates to determine
        targets based on:
        - Historical growth patterns
        - Price level changes
        - Consistency requirements
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray): Historical total
                consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths
            log_it (bool, optional): Whether to use log transformation
                in forecasting

        Returns:
            np.ndarray: Target consumption by industry
        """
        if historic_total_consumption[-1] == 0.0:
            return np.zeros(previous_desired_government_consumption.shape)

        # Fitting based on target consumption
        if self.consistency == 1.0:
            if self.fixed_total_government_consumption is None:
                historic_total_consumption_growth = (
                    historic_total_consumption[1:] / historic_total_consumption[:-1] - 1.0
                )
                self.fixed_total_government_consumption = (
                    np.exp(
                        ManualAutoregForecaster().forecast(
                            data=historic_total_consumption_growth,
                            t=20,
                            assume_zero_noise=assume_zero_noise,
                        )
                    )
                    - 1
                )
                self.fixed_total_government_consumption = (
                    np.cumprod(1 + self.fixed_total_government_consumption) * historic_total_consumption[-1]
                )

            consumption = self.fixed_total_government_consumption[current_time - 1]

        # Fitting based on historic consumption
        else:
            consumption = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_consumption),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * consumption
            * previous_desired_government_consumption
            / previous_desired_government_consumption.sum(),
        )


class ExogenousGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Exogenous consumption target determination.

    This class implements consumption targeting based on:
    - Pre-specified consumption paths
    - Price level adjustments
    - Default growth fallback
    - Industry-specific allocation

    The approach provides:
    - Externally determined targets
    - Price-adjusted spending
    - Fallback to growth-based targets
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using exogenous path.

        Determines targets based on:
        - Pre-specified consumption values
        - Price level changes
        - Default growth fallback
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray, optional): Historical
                total consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths

        Returns:
            np.ndarray: Target consumption by industry

        Raises:
            ValueError: If exogenous data not available for current time
        """
        if exogenous_total_consumption is None:
            return np.maximum(
                0.0,
                (1 + expected_inflation)
                * current_good_prices
                / initial_good_prices
                * (1 + self.default_growth)
                * previous_desired_government_consumption,
            )
        if current_time >= len(exogenous_total_consumption):
            raise ValueError("No exogenous data available beyond this point.")
        return (
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * exogenous_total_consumption[current_time]
            * previous_desired_government_consumption
            / previous_desired_government_consumption.sum()
        )
