from dataclasses import dataclass

import pandas as pd


@dataclass
class SyntheticCreditMarket:
    country_name: str
    year: int
    credit_market_data: pd.DataFrame
