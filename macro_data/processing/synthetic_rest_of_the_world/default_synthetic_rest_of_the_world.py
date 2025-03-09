"""Default implementation of Rest of the World (ROW) data preprocessing.

This module provides the standard implementation for managing Rest of the World data,
focusing on:

1. Data Integration:
   - Reading from standard data sources
   - Converting between currencies
   - Aggregating trade flows
   - Computing market shares

2. Market Structure:
   - Determining exporter counts
   - Scaling importer numbers
   - Preserving trade relationships
   - Initializing agent distributions

3. Growth Modeling:
   - Fitting export growth models
   - Estimating import trends
   - Processing historical patterns
   - Handling missing data

4. Configuration Options:
   - Flexible exporter allocation
   - Configurable growth modeling
   - Currency conversion handling
   - Market structure settings

Note:
    This implementation provides reasonable defaults for ROW preprocessing,
    suitable for most standard simulation scenarios. Custom implementations
    can extend this class for specific requirements.

Example:
    ```python
    from macro_data.readers import DataReaders
    from macro_data.configuration import ROWDataConfiguration

    readers = DataReaders(...)
    config = ROWDataConfiguration(...)

    row = DefaultSyntheticRestOfTheWorld.from_readers(
        year=2023,
        readers=readers,
        industry_data=industry_data,
        n_sellers_by_industry=sellers,
        n_buyers=buyers,
        row_configuration=config
    )
    ```
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.dataconfiguration import ROWDataConfiguration
from macro_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world import (
    SyntheticRestOfTheWorld,
)
from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticRestOfTheWorld(SyntheticRestOfTheWorld):
    """Default implementation of Rest of the World data preprocessing.

    This class provides a standard implementation for ROW data management with:
    1. Automated data reading and currency conversion
    2. Flexible market structure initialization
    3. Configurable growth model fitting
    4. Trade flow aggregation and scaling

    The implementation supports:
    - Optional growth model fitting for exports/imports
    - Configurable exporter allocation by industry
    - Automatic scaling of importer numbers
    - Currency conversion handling
    """

    def __init__(
        self,
        year: int,
        row_data: pd.DataFrame,
        n_exporters_by_industry: np.ndarray,
        n_importers,
        exports_model: Optional[LinearRegression],
        imports_model: Optional[LinearRegression],
    ):
        """Initialize the default ROW implementation.

        Args:
            year (int): Reference year for the data
            row_data (pd.DataFrame): Aggregated economic data for ROW
            n_exporters_by_industry (np.ndarray): Number of exporting agents by industry
            n_importers (int): Number of importing agents
            exports_model (Optional[LinearRegression]): Model for export growth
            imports_model (Optional[LinearRegression]): Model for import growth
        """
        super().__init__(
            year=year,
            row_data=row_data,
            n_exporters_by_industry=n_exporters_by_industry,
            exports_model=exports_model,
            imports_model=imports_model,
            n_importers=n_importers,
        )

    @classmethod
    def from_readers(
        cls,
        year: int,
        readers: DataReaders,
        industry_data: dict[str, dict[str, pd.DataFrame]],
        n_sellers_by_industry: np.ndarray,
        n_buyers: int,
        row_configuration: ROWDataConfiguration,
        row_exports_growth: Optional[pd.Series] = None,
        row_imports_growth: Optional[pd.Series] = None,
    ):
        """Create a ROW instance from data readers and configuration.

        This method:
        1. Aggregates trade data for non-simulated countries
        2. Converts currencies using exchange rates
        3. Fits growth models if configured
        4. Initializes market structure

        Args:
            year (int): Reference year for the data
            readers (DataReaders): Data source readers
            industry_data (dict[str, dict[str, pd.DataFrame]]): Industry data by country
            n_sellers_by_industry (np.ndarray): Number of sellers by industry in
                simulated countries
            n_buyers (int): Number of buyers in simulated countries
            row_configuration (ROWDataConfiguration): Configuration settings for ROW
            row_exports_growth (Optional[pd.Series], optional): Historical export
                growth data. Required if fit_exports is True.
            row_imports_growth (Optional[pd.Series], optional): Historical import
                growth data. Required if fit_imports is True.

        Returns:
            DefaultSyntheticRestOfTheWorld: Initialized ROW instance

        Raises:
            ValueError: If growth data is required but not provided
        """
        row_industry_data = industry_data["ROW"]

        total_imports = sum(
            [industry_data[c]["industry_vectors"]["Imports in USD"].sum() for c in industry_data if c != "ROW"]
        )
        exports_by_industry = np.sum(
            [industry_data[c]["industry_vectors"]["Exports in USD"].values for c in industry_data if c != "ROW"], axis=1
        )

        row_exports = row_industry_data["industry_vectors"]["Exports in USD"]
        row_imports = row_industry_data["industry_vectors"]["Imports in USD"]
        exchange_rate = readers.exchange_rates.from_usd_to_lcu("ROW", year)

        row_data = pd.DataFrame(
            {
                "Exports": row_exports,
                "Imports in USD": row_imports,
                "Imports in LCU": exchange_rate * row_imports,
            }
        )

        row_data["Price in USD"] = 1
        row_data["Price in LCU"] = exchange_rate * row_data["Price in USD"]

        if row_configuration.fit_exports:
            if row_exports_growth is None:
                raise ValueError("Exports growth data is required.")
            exports_model = LinearRegression()
            exports_model.fit([[0], [1]], [row_exports_growth.mean(), row_exports_growth.mean()])
        else:
            exports_model = None

        if row_configuration.fit_imports:
            if row_imports_growth is None:
                raise ValueError("Imports growth data is required.")
            imports_model = LinearRegression()
            imports_model.fit([[0], [1]], [row_imports_growth.mean(), row_imports_growth.mean()])
        else:
            imports_model = None

        if row_configuration.assume_one_exporter_by_industry:
            n_exporters_by_industry = np.ones(row_data.shape[0])
        else:
            n_exporters_by_industry = np.maximum(
                1, row_data["Exports"] / exports_by_industry * n_sellers_by_industry
            ).astype(int)

        n_importers = int(max(1, row_data["Imports in USD"].sum() / total_imports * n_buyers))

        return cls(
            year=year,
            row_data=row_data,
            n_exporters_by_industry=n_exporters_by_industry,
            exports_model=exports_model,
            imports_model=imports_model,
            n_importers=n_importers,
        )
