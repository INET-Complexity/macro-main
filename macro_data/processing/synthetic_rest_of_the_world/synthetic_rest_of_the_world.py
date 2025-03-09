"""Module for preprocessing Rest of the World (ROW) data in macroeconomic simulations.

This module provides the abstract base class for managing data related to the Rest of
the World (ROW) agent, which represents all countries not explicitly simulated in
the model. The ROW agent is crucial for:

1. Trade Relationships:
   - Aggregating exports to non-simulated countries
   - Tracking imports from non-simulated countries
   - Managing international price levels
   - Handling exchange rate effects

2. Data Aggregation:
   - Combining economic data from non-simulated countries
   - Scaling trade flows appropriately
   - Preserving global trade balance
   - Maintaining consistent units

3. Growth Modeling:
   - Processing historical growth patterns
   - Estimating export/import trends
   - Handling structural changes
   - Projecting future relationships

4. Market Structure:
   - Determining number of trading agents
   - Allocating market shares
   - Setting initial conditions
   - Preserving key relationships

Note:
    This module focuses on preprocessing and organizing ROW data for initialization.
    The actual trade dynamics are implemented in the simulation package.

Example:
    ```python
    from macro_data.processing.synthetic_rest_of_the_world import SyntheticRestOfTheWorld

    class CustomROW(SyntheticRestOfTheWorld):
        def __init__(self, year, row_data, ...):
            super().__init__(...)
            # Custom initialization logic
    ```
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class SyntheticRestOfTheWorld(ABC):
    """Abstract base class for Rest of the World (ROW) data preprocessing.

    This class provides the framework for managing data related to countries not
    explicitly simulated in the model. It handles:
    1. Trade data aggregation and scaling
    2. Price level and exchange rate processing
    3. Export/import relationship modeling
    4. Market structure initialization

    The ROW data is organized in a DataFrame with columns:
        - Exports: Value of exports to simulated countries
        - Imports in USD: Value of imports from simulated countries in USD
        - Imports in LCU: Value of imports in local currency units
        - Price in USD: Price levels in USD
        - Price in LCU: Price levels in local currency units

    Attributes:
        country_name (str): Always "ROW" for this class
        year (int): Reference year for the data
        row_data (pd.DataFrame): Aggregated economic data for non-simulated countries
        exports_model (Optional[LinearRegression]): Model for export growth trends
        imports_model (Optional[LinearRegression]): Model for import growth trends
        n_exporters_by_industry (np.ndarray): Number of exporting agents by industry
        n_importers (int): Number of importing agents
    """

    @abstractmethod
    def __init__(
        self,
        year: int,
        row_data: pd.DataFrame,
        n_exporters_by_industry: np.ndarray,
        n_importers: int,
        exports_model: Optional[LinearRegression],
        imports_model: Optional[LinearRegression],
    ):
        """Initialize the Rest of the World agent.

        Args:
            year (int): Reference year for the data
            row_data (pd.DataFrame): Aggregated economic data for ROW
            n_exporters_by_industry (np.ndarray): Number of exporting agents by industry
            n_importers (int): Number of importing agents
            exports_model (Optional[LinearRegression]): Model for export growth
            imports_model (Optional[LinearRegression]): Model for import growth
        """
        self.country_name = "ROW"
        self.year = year

        # Rest of the World data
        self.row_data = row_data

        # Models
        self.exports_model = exports_model
        self.imports_model = imports_model

        self.n_exporters_by_industry = n_exporters_by_industry
        self.n_importers = n_importers
