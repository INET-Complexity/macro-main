"""Data loading utilities for model results and configurations.

This module provides utilities for loading and processing model data from
HDF5 files, including time series data, agent states, and configurations.
It handles various data dimensionalities and supports industry-level
aggregation.

Key Features:
1. Data Loading:
   - HDF5 file handling
   - Configuration parsing
   - Multi-dimensional data support

2. Data Processing:
   - Time series formatting
   - Industry-level aggregation
   - Dimension reduction

3. Output Formatting:
   - DataFrame conversion
   - Index management
   - Column organization
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import yaml


class Loader:
    """Data loader for model results and configurations.

    This class handles loading and processing of model data from HDF5
    files, including configuration information and time series data
    for various agents and fields.

    Attributes:
        file: Open HDF5 file handle
        config: Parsed YAML configuration
    """

    def __init__(self, path: Path | str):
        """Initialize loader with data file path.

        Args:
            path: Path to HDF5 file containing model data
        """
        self.file = h5py.File(path, "r")
        self.config = yaml.safe_load(self.file.attrs["configuration"])

    def get_country_agent_field_dataframe(
        self,
        country_name: str,
        agent_name: str,
        field: Optional[str] = None,
        aggregate_by_industry: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load field data for specific country and agent.

        This method loads and processes data for a specific field,
        optionally aggregating by industry and handling various
        data dimensionalities.

        Args:
            country_name: Name of country to load data for
            agent_name: Type of agent (e.g., "households", "firms")
            field: Specific field to load (optional)
            aggregate_by_industry: Aggregation method ("sum" or "mean")

        Returns:
            pd.DataFrame: Processed data with appropriate indexing

        Raises:
            ValueError: If dataset shape is unsupported or aggregation
                method is unknown
        """
        # Load data
        dataset = self.file[country_name][agent_name]
        if field is not None:
            dataset = dataset[field]
        if len(dataset.shape) == 1:
            data = load_1d_field_as_dataframe(dataset)
        elif len(dataset.shape) == 2:
            data = load_2d_field_as_dataframe(dataset)
        elif len(dataset.shape) == 3:
            data = load_3d_field_as_dataframe(dataset)
        else:
            raise ValueError("Unsupported dataset shape", dataset.shape)

        # Create index
        if len(data) != self.config["t_max"] and agent_name != "exogenous":
            logging.warning("Time series length does not match")
            logging.warning("Country %s, agent %s, field %s", country_name, agent_name, field)
            logging.warning("Time series length: %d", len(data))
            logging.warning("t_max: %d", self.config["t_max"])
            logging.warning(f"Data: {data}")
        dates = []

        # TODO shoudln't be hardcoded
        init_year = 2014

        for year in range(
            init_year,
            init_year + 1 + int(np.floor(self.config["t_max"] / 12)),
        ):
            for month in range(1, 13):
                dates.append(pd.Timestamp(year=year, month=month, day=1))
        data.index = pd.Index(dates[0 : len(data)], name="Date")

        # Aggregate by industry
        if aggregate_by_industry is not None:
            industries = np.array(self.file[country_name]["industry_firms"]).flatten()
            data.columns = pd.Index(industries, name="Industry")
            if aggregate_by_industry == "sum":
                return data.T.groupby(level=0).sum().T
            elif aggregate_by_industry == "mean":
                return data.T.groupby(level=0).mean().T
            else:
                raise ValueError("Unknown function for industry aggregation.", aggregate_by_industry)

        return data


def load_1d_field_as_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    """Convert 1D HDF5 dataset to DataFrame.

    Args:
        dataset: 1D HDF5 dataset to convert

    Returns:
        pd.DataFrame: Single-column DataFrame with numeric index
    """
    return pd.DataFrame(
        pd.Series(
            dataset,
            index=pd.Index(np.arange(len(dataset))),
        )
    )


def load_2d_field_as_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    """Convert 2D HDF5 dataset to DataFrame.

    Args:
        dataset: 2D HDF5 dataset to convert

    Returns:
        pd.DataFrame: DataFrame with agent IDs as columns and
            numeric date index
    """
    ts_data = np.array(dataset)
    return pd.DataFrame(
        ts_data,
        columns=pd.Index(range(ts_data.shape[1]), name="Agent ID"),
        index=pd.Index(np.arange(ts_data.shape[0]), name="Date"),
    )


def load_3d_field_as_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    """Convert 3D HDF5 dataset to DataFrame.

    Args:
        dataset: 3D HDF5 dataset to convert

    Returns:
        pd.DataFrame: DataFrame with multi-index columns for agent
            and industry IDs, and numeric date index
    """
    ts_data = np.array(dataset)
    columns = pd.MultiIndex.from_tuples(dataset.attrs["columns"], names=("Agent ID", "Industry"))
    return pd.DataFrame(
        ts_data.reshape(-1, ts_data.shape[-1]),
        columns=columns,
        index=pd.Index(np.arange(ts_data.shape[0]), name="Date"),
    )
