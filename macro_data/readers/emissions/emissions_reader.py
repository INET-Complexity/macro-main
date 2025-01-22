from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.io_tables.icio_reader import ICIOReader

COAL_TCO2_PER_TON = 1.57
OIL_TCO2_PER_BARREL = 0.43
GAS_TCO2_PER_MBTU = 0.053


@dataclass
class EmissionsReader:
    prices_df: pd.DataFrame

    @classmethod
    def read_price_data(cls, data_path: Path | str):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        coal = pd.read_csv(data_path / "PCOALAUUSDM.csv")
        coal["observation_date"] = pd.to_datetime(coal["observation_date"])
        coal.rename(columns={"PCOALAUUSDM": "coal_price", "observation_date": "DATE"}, inplace=True)
        coal.set_index("DATE", inplace=True)
        coal = coal.resample("YS").first()

        oil = pd.read_csv(data_path / "POILBREUSDM.csv")
        oil["observation_date"] = pd.to_datetime(oil["observation_date"])
        oil.rename(columns={"POILBREUSDM": "oil_price", "observation_date": "DATE"}, inplace=True)
        oil.set_index("DATE", inplace=True)
        oil = oil.resample("YS").first()

        gas = pd.read_csv(data_path / "PNGASEUUSDM.csv")
        gas["observation_date"] = pd.to_datetime(gas["observation_date"])
        gas.rename(columns={"PNGASEUUSDM": "gas_price", "observation_date": "DATE"}, inplace=True)
        gas.set_index("DATE", inplace=True)
        gas = gas.resample("YS").first()

        prices_df = pd.merge(coal, oil, left_index=True, right_index=True)
        prices_df = pd.merge(prices_df, gas, left_index=True, right_index=True)

        return cls(prices_df=prices_df)

    def get_emissions_factors(self, year: int) -> dict[str, float]:
        coal_tco2_per_usd = COAL_TCO2_PER_TON / self.prices_df.loc[f"{year}", "coal_price"].iloc[0]
        oil_tco2_per_usd = OIL_TCO2_PER_BARREL / self.prices_df.loc[f"{year}", "oil_price"].iloc[0]
        gas_tco2_per_usd = GAS_TCO2_PER_MBTU / self.prices_df.loc[f"{year}", "gas_price"].iloc[0]

        return {
            "coal": coal_tco2_per_usd,
            "oil": oil_tco2_per_usd,
            "gas": gas_tco2_per_usd,
        }


@dataclass
class EmissionsData:
    oil_factor_lcu: float
    gas_factor_lcu: float
    coal_factor_lcu: float
    refining_factor_lcu: float

    @classmethod
    def from_readers(
        cls,
        usd_emission_factors: dict[str, float],
        exchange_rate: float,
    ):
        oil_factor_lcu = usd_emission_factors["oil"] / exchange_rate
        gas_factor_lcu = usd_emission_factors["gas"] / exchange_rate
        coal_factor_lcu = usd_emission_factors["coal"] / exchange_rate
        refining_factor_lcu = usd_emission_factors["coke_refining"] / exchange_rate

        return cls(
            oil_factor_lcu=oil_factor_lcu,
            gas_factor_lcu=gas_factor_lcu,
            coal_factor_lcu=coal_factor_lcu,
            refining_factor_lcu=refining_factor_lcu,
        )

    @property
    def emissions_array(self):
        return np.array([self.oil_factor_lcu, self.gas_factor_lcu, self.coal_factor_lcu, self.refining_factor_lcu])
