from pydantic import BaseModel
from typing import Literal


class Growth(BaseModel):
    """
    The function used for setting how growth is centrally forecasted.
    """

    name: Literal[
        "GrowthForecastingConstant", "GrowthManualForecastingAutoReg", "GrowthImplementedForecastingAutoReg"
    ] = "GrowthForecastingConstant"
    parameters: dict = {"value": 0.0}
    path_name: str = "growth"


class HPI(BaseModel):
    """
    The function used for setting how the house price index is centrally forecasted.
    """

    name: Literal["HPIForecastingConstant", "HPIManualForecastingAutoReg", "HPIImplementedForecastingAutoReg"] = (
        "HPIForecastingConstant"
    )
    parameters: dict = {"value": 0.0}
    path_name: str = "house_price_index"


class Inflation(BaseModel):
    """
    The function used for setting how inflation is centrally forecasted.
    """

    name: Literal[
        "InflationForecastingConstant", "InflationImplementedForecastingAutoReg", "InflationManualForecastingAutoReg"
    ] = "InflationForecastingConstant"
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
