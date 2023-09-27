import pandas as pd

from pathlib import Path


class GoodsCriticalityReader:
    def __init__(self, df):
        self.criticality_matrix = df

    @classmethod
    def from_csv(cls, path: Path | str) -> "GoodsCriticalityReader":
        df = pd.read_csv(path, header=0, index_col=0)
        df = cls.aggregate(df)
        return cls(df)

    @staticmethod
    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        map_to_nace1 = {}
        for col in df.index:
            map_to_nace1[col] = col[0:1]
        map_to_nace1["R"] = "R_S"
        map_to_nace1["S"] = "R_S"
        df.index = df.index.map(map_to_nace1)
        df.columns = df.columns.map(map_to_nace1)
        df = df.stack().reset_index().rename(columns={0: "Value"})
        df = df.groupby(["level_0", "level_1"])["Value"].max().unstack()
        df.index.name = "Demand"
        df.columns.name = "Supply"
        return df
