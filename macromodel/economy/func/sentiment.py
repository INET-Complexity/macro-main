"""Economic sentiment computation module.

This module provides mechanisms for setting and computing economic sentiment
across industries. Sentiment values can influence agent behavior and
decision-making in the economic model. Currently implements:

1. Abstract Sentiment Interface:
   - Defines standard interface for sentiment computation
   - Supports industry-specific sentiment values

2. Constant Sentiment:
   - Uniform sentiment across all industries
   - Useful for baseline scenarios and testing
"""

from abc import ABC, abstractmethod

import numpy as np


class SentimentSetter(ABC):
    """Abstract base class for economic sentiment computation.

    Provides interface for computing sentiment values across industries.
    Implementations can vary from constant values to complex dynamic calculations.
    """

    @abstractmethod
    def compute_sentiment(self, n_industries: int) -> np.ndarray:
        """Compute sentiment values for each industry.

        Args:
            n_industries (int): Number of industries to compute sentiment for

        Returns:
            np.ndarray: Array of sentiment values, one per industry
        """
        pass


class ConstantSentimentSetter(SentimentSetter):
    """Constant sentiment implementation.

    Sets uniform sentiment value across all industries.
    Useful for baseline scenarios and model testing.

    Attributes:
        sentiment_value (float): Fixed sentiment value to apply
    """

    def __init__(self, value: float):
        """Initialize constant sentiment setter.

        Args:
            value (float): Fixed sentiment value to use
        """
        self.sentiment_value = value

    def compute_sentiment(self, n_industries: np.ndarray) -> np.ndarray:
        """Compute uniform sentiment values.

        Args:
            n_industries (np.ndarray): Number of industries

        Returns:
            np.ndarray: Array of constant sentiment values
        """
        return np.full(n_industries, self.sentiment_value)
