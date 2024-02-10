from pydantic import BaseModel
from typing import Literal, Any


class ClearingFunction(BaseModel):
    path_name: str = "clearing"
    name: Literal["NoHousingMarketClearer", "DefaultHousingMarketClearer"] = "NoHousingMarketClearer"
    parameters: dict[str, Any] = {}


class PropertyValueFunction(BaseModel):
    path_name: str = "value"
    name: Literal["DefaultPropertyValueSetter"] = "DefaultPropertyValueSetter"
    parameters: dict[str, Any] = {"random_fluctuation_std": 0.0}


class HousingMarketFunctions(BaseModel):
    clearing: ClearingFunction = ClearingFunction()
    value: PropertyValueFunction = PropertyValueFunction()


class HousingMarketConfiguration(BaseModel):
    functions: HousingMarketFunctions = HousingMarketFunctions()
