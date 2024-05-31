from pydantic import BaseModel
from typing import Literal


class Clearing(BaseModel):
    name: Literal["NoGoodsMarketClearer", "DefaultGoodsMarketClearer", "WaterBucketGoodsMarketClearer"] = (
        "WaterBucketGoodsMarketClearer"
    )
    path_name: str = "clearing"
    parameters: dict = {
        "consider_trade_proportions": True,
        "consider_buyer_priorities": True,
        "additionally_available_factor": 0.2,
        "price_markup": 0.2,
        "price_temperature": 2.0,
        "prio_domestic_sellers": False,
        "prio_high_prio_buyers": False,
        "prio_high_prio_sellers": False,
        "real_country_prioritisation": 10.0,
        "probability_keeping_previous_seller": 0.0,
        "trade_temperature": 0.0,
        "seller_selection_distribution_type": "additive",
        "seller_minimum_fill": 0.9,
        "buyer_minimum_fill_macro": 0.0,
        "buyer_minimum_fill_micro": 0.95,
        "deterministic": True,
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
