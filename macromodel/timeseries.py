"""Time series data management for the macroeconomic model.

This module provides a flexible time series data structure that stores and manages
temporal data for various economic variables. It supports:
- Multiple time series variables stored as lists
- Access to initial, current, and previous values
- HDF5 file storage and retrieval
- Multi-dimensional data handling (1D, 2D, and 3D arrays)
"""

import logging

import h5py
import numpy as np
import pandas as pd


class TimeSeries:
    """A flexible container for managing multiple time series variables.

    This class stores multiple time series as lists in a dictionary, with each key
    representing a different economic variable. It provides methods to access values
    at different time points and save/load data to/from HDF5 files.

    The class is particularly designed for economic simulations where multiple
    variables need to be tracked over time, with special handling for initial
    conditions, current values, and historical data.

    Attributes:
        dicts (dict): Dictionary storing time series data, where each key is a variable
            name and each value is a list of that variable's values over time.
    """

    def __init__(self, **kwargs):
        """Initialize a new TimeSeries instance.

        Args:
            **kwargs: Keyword arguments where each key becomes a time series variable
                     and its value becomes the initial value for that series.
        """
        self.dicts = {}
        for key, value in kwargs.items():
            if value is not None:
                self.dicts[key] = [value]

    def __getattr__(self, item):
        """Get the time series for a variable.

        Args:
            item (str): Name of the time series variable

        Returns:
            list: The complete time series for the variable

        Raises:
            KeyError: If the variable doesn't exist in the time series
        """
        return self.dicts[item]

    def __setitem__(self, key, value):
        """Set or create a new time series with an initial value.

        Args:
            key (str): Name of the time series variable
            value: Initial value for the time series
        """
        self.dicts[key] = [value]

    def initial(self, item):
        """Get the initial value of a time series.

        Args:
            item (str): Name of the time series variable

        Returns:
            The first value in the time series
        """
        return self.dicts[item][0]

    def current(self, item):
        """Get the most recent value of a time series.

        Args:
            item (str): Name of the time series variable

        Returns:
            The last value in the time series
        """
        return self.dicts[item][-1]

    def prev(self, item):
        """Get the previous value of a time series.

        If the time series only has one value, returns the current value.

        Args:
            item (str): Name of the time series variable

        Returns:
            The second-to-last value in the time series, or the last value if
            there is only one value
        """
        if len(self.dicts[item]) == 1:
            return self.dicts[item][-1]
        return self.dicts[item][-2]

    def historic(self, item):
        """Get the complete history of a time series.

        Args:
            item (str): Name of the time series variable

        Returns:
            list: Complete time series data for the variable
        """
        return self.dicts[item]

    def get_keys(self):
        """Get all variable names in the time series.

        Returns:
            list[str]: List of all time series variable names
        """
        return list(self.dicts.keys())

    def write_to_h5(self, agent_name: str, country_group: h5py.Group) -> None:
        """Write all time series data to an HDF5 file.

        Args:
            agent_name (str): Name of the agent (used as group name in HDF5)
            country_group (h5py.Group): HDF5 group to write to

        Note:
            Handles errors for inhomogeneous shapes by logging the problematic fields
        """
        agent_group = country_group.create_group(agent_name)
        for field in self.get_keys():
            try:
                ts_data = np.array(self.historic(field))
                self.write_field_to_h5(ts_data, field, agent_group)
            except ValueError:
                logging.error("inhomogeneous shape", agent_name, field)
                for i in range(len(self.historic(field))):
                    logging.error(self.historic(field)[i].shape)

    def write_field_to_h5(self, ts_data: np.ndarray, field: str, agent_group: h5py.Group) -> None:
        """Write a single time series field to an HDF5 file.

        Handles different dimensionality of data (1D, 2D, 3D) appropriately.

        Args:
            ts_data (np.ndarray): Time series data to write
            field (str): Name of the field
            agent_group (h5py.Group): HDF5 group to write to
        """
        if len(ts_data.shape) == 1 or len(ts_data.shape) == 2:
            self.write_2d_field_to_h5(ts_data, field, agent_group)
        elif len(ts_data.shape) == 3:
            self.write_3d_field_to_h5(ts_data, field, agent_group)

    @staticmethod
    def write_2d_field_to_h5(ts_data: np.ndarray, field: str, agent_group: h5py.Group) -> None:
        """Write a 1D or 2D time series field to an HDF5 file.

        Args:
            ts_data (np.ndarray): 1D or 2D time series data
            field (str): Name of the field
            agent_group (h5py.Group): HDF5 group to write to
        """
        if len(ts_data.shape) == 1:
            ts_data = ts_data.reshape((1, ts_data.shape[0]))
        agent_group.create_dataset(
            field,
            data=ts_data,
            dtype=float,
        )

    def write_3d_field_to_h5(self, ts_data: np.ndarray, field: str, agent_group: h5py.Group) -> None:
        """Write a 3D time series field to an HDF5 file.

        Converts 3D data to a multi-index DataFrame format before saving.

        Args:
            ts_data (np.ndarray): 3D time series data
            field (str): Name of the field
            agent_group (h5py.Group): HDF5 group to write to
        """
        multiindex_df = self.create_multiindex_dataframe(ts_data)
        multiindex_array = multiindex_df.to_numpy()
        agent_group.create_dataset(
            field,
            data=multiindex_array,
            dtype=float,
        )
        columns_array = multiindex_df.columns.to_numpy()
        columns_array = np.array([*columns_array]).astype("int")

        # Can't save many columns in attrs field, so saving them as a dataset
        agent_group.create_dataset(f"{field}_columns", data=columns_array, dtype=int)

    @staticmethod
    def create_multiindex_dataframe(ts_data: np.ndarray) -> pd.DataFrame:
        """Create a multi-index DataFrame from 3D time series data.

        Args:
            ts_data (np.ndarray): 3D array with shape (time, agents, industries)

        Returns:
            pd.DataFrame: Multi-index DataFrame with time, agent, and industry indices
        """
        multiindex_data = []
        for agent_id in range(ts_data.shape[1]):
            multiindex_data.append(
                pd.DataFrame(
                    data=ts_data[:, agent_id, :],
                    index=pd.Index(np.arange(ts_data.shape[0]), name="Months"),
                    columns=pd.Index(np.arange(ts_data.shape[2]), name="Industry"),
                )
            )
        return pd.concat(
            multiindex_data,
            axis=1,
            keys=range(ts_data.shape[1]),
            names=["Agent ID", "Industry"],
        )

    def reset(self):
        """Reset all time series to their initial values."""
        for field in self.get_keys():
            self.dicts[field] = [self.dicts[field][0]]

    def __eq__(self, other: "TimeSeries"):
        """Compare two TimeSeries instances for equality.

        Args:
            other (TimeSeries): Another TimeSeries instance to compare with

        Returns:
            bool: True if all time series in both instances are equal
        """
        for field in self.get_keys():
            this_field = np.array(self.historic(field))
            other_field = np.array(other.historic(field))
            if not np.array_equal(this_field, other_field):
                return False
        return True

    def get_aggregate(self, name: str):
        """Get the sum of a time series across all dimensions except time.

        Args:
            name (str): Name of the time series variable

        Returns:
            np.ndarray: Array of sums for each time point
        """
        return np.array(self.historic(name)).sum(axis=1)
