from typing import Literal

from pydantic import BaseModel


class Growth(BaseModel):
    """
    The function used for setting how growth is centrally forecasted.
    """

    name: Literal["GrowthForecastingConstant"] = "GrowthForecastingConstant"
    parameters: dict = {"value": 0.0}
    path_name: str = "growth"


class HPI(BaseModel):
    """
    The function used for setting how the house price index is centrally forecasted.
    """

    name: Literal["HPIForecastingConstant"] = "HPIForecastingConstant"
    parameters: dict = {"value": 0.0}
    path_name: str = "house_price_index"


class Inflation(BaseModel):
    """
    The function used for setting how inflation is centrally forecasted.
    """

    name: Literal["InflationForecastingConstant"] = "InflationForecastingConstant"
    parameters: dict = {"value": 0.0}
    path_name: str = "inflation"


class Sentiment(BaseModel):
    """
    The function used for setting how sector sentiment is centrally forecasted.
    """

    name: Literal["ConstantSentimentSetter"] = "ConstantSentimentSetter"
    parameters: dict = {"value": 0.0}
    path_name: str = "sentiment"


class EconomyFunctions(BaseModel):
    """
    The functions used for the economy.
    """

    growth: Growth = Growth()
    house_price_index: HPI = HPI()
    inflation: Inflation = Inflation()
    sentiment: Sentiment = Sentiment()


class EconomyConfiguration(BaseModel):
    """
    The configuration settings for the economy.
    """

    functions: EconomyFunctions = EconomyFunctions()
