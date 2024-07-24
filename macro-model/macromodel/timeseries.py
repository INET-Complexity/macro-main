import h5py
import logging
import numpy as np
import pandas as pd


class TimeSeries:
    def __init__(self, **kwargs):
        self.dicts = {}
        for key, value in kwargs.items():
            self.dicts[key] = [value]

    def __getattr__(self, item):
        return self.dicts[item]

    def __setitem__(self, key, value):
        self.dicts[key] = [value]

    def initial(self, item):
        return self.dicts[item][0]

    def current(self, item):
        return self.dicts[item][-1]

    def prev(self, item):
        if len(self.dicts[item]) == 1:
            return self.dicts[item][-1]
        return self.dicts[item][-2]

    def historic(self, item):
        return self.dicts[item]

    def get_keys(self):
        return list(self.dicts.keys())

    def write_to_h5(self, agent_name: str, country_group: h5py.Group) -> None:
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
        if len(ts_data.shape) == 1 or len(ts_data.shape) == 2:
            self.write_2d_field_to_h5(ts_data, field, agent_group)
        elif len(ts_data.shape) == 3:
            self.write_3d_field_to_h5(ts_data, field, agent_group)

    @staticmethod
    def write_2d_field_to_h5(ts_data: np.ndarray, field: str, agent_group: h5py.Group) -> None:
        if len(ts_data.shape) == 1:
            ts_data = ts_data.reshape((1, ts_data.shape[0]))
        agent_group.create_dataset(
            field,
            data=ts_data,
            dtype=float,
        )

    def write_3d_field_to_h5(self, ts_data: np.ndarray, field: str, agent_group: h5py.Group) -> None:
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
        for field in self.get_keys():
            self.dicts[field] = [self.dicts[field][0]]

    def __eq__(self, other: "TimeSeries"):
        for field in self.get_keys():
            this_field = np.array(self.historic(field))
            other_field = np.array(other.historic(field))
            if not np.array_equal(this_field, other_field):
                return False
        return True

    def get_aggregate(self, name: str):
        return np.array(self.historic(name)).sum(axis=1)
