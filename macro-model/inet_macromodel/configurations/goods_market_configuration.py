from pydantic import BaseModel
from typing import Literal


class Clearing(BaseModel):
    name: Literal["NoGoodsMarketClearer", "DefaultGoodsMarketClearer", "ProRataGoodsMarketClearer"] = (
        "ProRataGoodsMarketClearer"
    )
    path_name: str = "clearing"
    parameters: dict = {
        "trade_temperature": 0.0,
        "price_temperature": 1.0,
        "prio_domestic_sellers": False,
        "prio_high_prio_buyers": True,
        "prio_high_prio_sellers": False,
        "prio_real_countries": False,
        "probability_keeping_previous_seller": 0.2,
    }


class GoodsMarketFunctions(BaseModel):
    clearing: Clearing = Clearing()


class GoodsMarketConfiguration(BaseModel):
    """
    Configuration for the goods market.

    Attributes:
    - functions: The functions used in the goods market.
    - parameters: The parameters used in the goods market.
    """

    functions: GoodsMarketFunctions = GoodsMarketFunctions()
