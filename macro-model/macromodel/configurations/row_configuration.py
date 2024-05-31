from pydantic import BaseModel
from typing import Literal


class Exports(BaseModel):
    name: Literal["GrowthRoWExportsSetter", "DefaultRoWExportsSetter"] = "GrowthRoWExportsSetter"
    path_name: str = "exports"
    parameters: dict = {"consistency": 0.0}


class Imports(BaseModel):
    name: Literal["InflationRoWImportsSetter", "DefaultRoWImportsSetter"] = "InflationRoWImportsSetter"
    path_name: str = "imports"
    parameters: dict = {"consistency": 1.0}


class ExcessDemand(BaseModel):
    name: Literal["ZeroExcessDemandSetter", "InfinityExcessDemandSetter"] = "InfinityExcessDemandSetter"
    path_name: str = "excess_demand"
    parameters: dict = {}


class Prices(BaseModel):
    name: Literal["InflationRoWPriceSetter"] = "InflationRoWPriceSetter"
    path_name: str = "prices"
    parameters: dict = {}


class RestOfTheWorldFunctions(BaseModel):
    exports: Exports = Exports()
    imports: Imports = Imports()
    excess_demand: ExcessDemand = ExcessDemand()
    prices: Prices = Prices()


class RestOfTheWorldParameters(BaseModel):
    adjustment_speed: float = 1.0


class RestOfTheWorldConfiguration(BaseModel):
    functions: RestOfTheWorldFunctions = RestOfTheWorldFunctions()
    parameters: RestOfTheWorldParameters = RestOfTheWorldParameters()

    forecasting_window: int = 60
    assume_zero_growth: bool = False
    assume_zero_noise: bool = False
