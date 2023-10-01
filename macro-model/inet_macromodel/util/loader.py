import logging

import yaml
import h5py
import numpy as np
import pandas as pd

from pathlib import Path

from typing import Optional


class Loader:
    def __init__(self, path: Path | str):
        self.file = h5py.File(path, "r")
        self.config = yaml.safe_load(self.file["config"].attrs["config_data"])
        self.industries = self.config["model"]["industries"]["value"]

    def get_country_agent_field_dataframe(
        self,
        country_name: str,
        agent_name: str,
        field: Optional[str] = None,
        aggregate_by_industry: Optional[str] = None,
    ) -> pd.DataFrame:
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
        if len(data) != self.config["model"]["t_max"]["value"] and agent_name != "exogenous":
            logging.warning("Time series length does not match")
            logging.warning("Country %s, agent %s, field %s", country_name, agent_name, field)
            logging.warning("Time series length: %d", len(data))
            logging.warning("t_max: %d", self.config["model"]["t_max"]["value"])
            logging.warning(f"Data: {data}")
        dates = []
        for year in range(
            self.config["model"]["year"]["value"],
            self.config["model"]["year"]["value"] + 1 + int(np.floor(self.config["model"]["t_max"]["value"] / 12)),
        ):
            for month in range(1, 13):
                dates.append(pd.Timestamp(year=year, month=month, day=1))
        data.index = pd.Index(dates[0 : len(data)], name="Date")

        # Aggregate by industry
        if aggregate_by_industry is not None:
            industries = np.array(self.file[country_name]["industry_firms"]).flatten()
            data.columns = pd.Index(industries, name="Industry")
            if aggregate_by_industry == "sum":
                return data.groupby(axis=1, level=0).sum()
            elif aggregate_by_industry == "mean":
                return data.groupby(axis=1, level=0).mean()
            else:
                raise ValueError("Unknown function for industry aggregation.", aggregate_by_industry)

        return data


def load_1d_field_as_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    return pd.DataFrame(
        pd.Series(
            dataset,
            index=pd.Index(np.arange(len(dataset))),
        )
    )


def load_2d_field_as_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    ts_data = np.array(dataset)
    return pd.DataFrame(
        ts_data,
        columns=pd.Index(range(ts_data.shape[1]), name="Agent ID"),
        index=pd.Index(np.arange(ts_data.shape[0]), name="Date"),
    )


def load_3d_field_as_dataframe(dataset: h5py.Dataset) -> pd.DataFrame:
    ts_data = np.array(dataset)
    columns = pd.MultiIndex.from_tuples(dataset.attrs["columns"], names=("Agent ID", "Industry"))
    return pd.DataFrame(
        ts_data.reshape(-1, ts_data.shape[-1]),
        columns=columns,
        index=pd.Index(np.arange(ts_data.shape[0]), name="Date"),
    )
