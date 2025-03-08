"""Property value computation and update mechanisms.

This module implements the logic for updating property values in the housing
market. It provides an abstract base class for value setters and a default
implementation that adds random fluctuations to current values.

The module supports:
- Abstract interface for value computation
- Random value fluctuations
- Extensible value update mechanisms
- Market-wide value adjustments

Key components:
1. PropertyValueSetter: Abstract base class defining the interface
2. DefaultPropertyValueSetter: Simple implementation with random fluctuations
"""

from abc import ABC, abstractmethod

import numpy as np


class PropertyValueSetter(ABC):
    """Abstract base class for property value computation.

    This class defines the interface for computing property values in the
    housing market. Implementations can incorporate various factors such as:
    - Market conditions
    - Property characteristics
    - Location effects
    - Economic indicators
    """

    @abstractmethod
    def compute_value(self, current_property_values: np.ndarray) -> np.ndarray:
        """Compute updated property values.

        Args:
            current_property_values: Array of current property values

        Returns:
            np.ndarray: Array of updated property values

        Note:
            Implementations should preserve array shape and handle
            numerical edge cases appropriately.
        """
        pass


class DefaultPropertyValueSetter(PropertyValueSetter):
    """Default implementation adding random fluctuations to property values.

    This class implements a simple value update mechanism that applies
    random normally-distributed fluctuations to current property values.
    It's useful for simulating market volatility and price discovery.

    The fluctuations are controlled by a standard deviation parameter,
    allowing for different levels of market volatility.
    """

    def __init__(self, random_fluctuation_std: float):
        """Initialize the default value setter.

        Args:
            random_fluctuation_std: Standard deviation for random fluctuations
                Higher values create more volatile prices
                Typical values range from 0.01 to 0.1

        Example:
            setter = DefaultPropertyValueSetter(random_fluctuation_std=0.05)
            # This creates a setter with 5% standard deviation in price changes
        """
        self.random_fluctuation_std = random_fluctuation_std

    def compute_value(self, current_property_values: np.ndarray) -> np.ndarray:
        """Compute new property values with random fluctuations.

        This method applies normally distributed random fluctuations to
        current property values. The fluctuations are multiplicative,
        ensuring values remain positive.

        Args:
            current_property_values: Array of current property values

        Returns:
            np.ndarray: Array of updated property values

        Example:
            For random_fluctuation_std = 0.05:
            - A property worth 100,000 might fluctuate to:
              * 95,000 (5% decrease)
              * 105,000 (5% increase)
            - The changes follow a normal distribution
        """
        return (
            1 + np.random.normal(0.0, self.random_fluctuation_std, current_property_values.shape)
        ) * current_property_values
