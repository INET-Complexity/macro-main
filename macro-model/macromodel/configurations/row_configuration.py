from pydantic import BaseModel
from typing import Literal


class Exports(BaseModel):
    name: Literal["ConstantRoWExportsSetter", "DefaultRoWExportsSetter"] = "ConstantRoWExportsSetter"
    path_name: str = "exports"
    parameters: dict = {}


class Imports(BaseModel):
    name: Literal["ConstantRoWImportsSetter", "DefaultRoWImportsSetter"] = "ConstantRoWImportsSetter"
    path_name: str = "imports"
    parameters: dict = {}


class Inflation(BaseModel):
    name: Literal["DefaultRoWInflationSetter"] = "DefaultRoWInflationSetter"
    path_name: str = "inflation"
    parameters: dict = {}


class Prices(BaseModel):
    name: Literal["ConstantRoWPriceSetter", "InflationRoWPriceSetter"] = "ConstantRoWPriceSetter"
    path_name: str = "prices"
    parameters: dict = {}


class RestOfTheWorldFunctions(BaseModel):
    exports: Exports = Exports()
    imports: Imports = Imports()
    inflation: Inflation = Inflation()
    prices: Prices = Prices()


class RestOfTheWorldConfiguration(BaseModel):
    functions: RestOfTheWorldFunctions = RestOfTheWorldFunctions()
