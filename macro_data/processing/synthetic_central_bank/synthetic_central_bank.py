"""Module for preprocessing synthetic central bank data.

This module provides an abstract base class for preprocessing and storing synthetic central bank data
that will be used to initialize behavioral models in the simulation package. Key preprocessing includes:

1. Data Collection:
   - Historical policy rates
   - Inflation data
   - Growth indicators

2. Data Organization:
   - Policy rate preprocessing
   - Parameter estimation
   - Data validation

3. Country-Specific Data:
   - Currency area identification
   - Policy rate histories
   - Economic indicators

Note:
    This module is NOT used for simulating central bank behavior. It only handles
    the preprocessing and organization of data that will later be used to initialize
    behavioral models in the simulation package.
"""

from abc import ABC, abstractmethod

import pandas as pd


class SyntheticCentralBank(ABC):
    """Abstract base class for preprocessing and storing central bank data.

    This class provides a framework for collecting and organizing central bank data
    that will be used to initialize behavioral models. It is NOT used for simulating
    central bank behavior - it only handles data preprocessing.

    The central bank data is stored in a pandas DataFrame containing:
        - Policy Rate: Historical/initial policy rates
        - Additional metrics may be added by concrete implementations

    Note:
        This is a data container class. The actual central bank behavior (policy decisions,
        rate setting, etc.) is implemented in the simulation package, which uses this
        preprocessed data for initialization.

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Reference year for data preprocessing
        central_bank_data (pd.DataFrame): Preprocessed central bank data
    """

    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
        central_bank_data: pd.DataFrame,
    ):
        """Initialize the central bank data container.

        Args:
            country_name (str): Country identifier for data collection
            year (int): Reference year for data preprocessing
            central_bank_data (pd.DataFrame): Initial central bank data to preprocess
        """
        self.country_name = country_name
        self.year = year

        # Preprocessed bank data
        self.central_bank_data = central_bank_data
