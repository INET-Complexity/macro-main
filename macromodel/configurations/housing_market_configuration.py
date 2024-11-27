from typing import Any, Literal

from pydantic import BaseModel


class ClearingFunction(BaseModel):
    path_name: str = "clearing"
    name: Literal["NoHousingMarketClearer", "DefaultHousingMarketClearer", "AutomaticHousingMarketClearer"] = (
        "AutomaticHousingMarketClearer"
    )
    parameters: dict[str, Any] = {"random_assignment_shock_variance": 0.0}


class PropertyValueFunction(BaseModel):
    path_name: str = "value"
    name: Literal["DefaultPropertyValueSetter"] = "DefaultPropertyValueSetter"
    parameters: dict[str, Any] = {"random_fluctuation_std": 0.0}


class HousingMarketFunctions(BaseModel):
    clearing: ClearingFunction = ClearingFunction()
    value: PropertyValueFunction = PropertyValueFunction()


class HousingMarketConfiguration(BaseModel):
    functions: HousingMarketFunctions = HousingMarketFunctions()
