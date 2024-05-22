from pydantic import BaseModel
from typing import Literal


class Consumption(BaseModel):
    name: Literal[
        "AutoregressiveGovernmentConsumptionSetter",
        "ConstantGrowthGovernmentConsumptionSetter",
        "ExogenousGovernmentConsumptionSetter",
    ] = "AutoregressiveGovernmentConsumptionSetter"
    path_name: str = "consumption"
    parameters: dict = {"consistency": 1.0}


class GovernmentFunctions(BaseModel):
    consumption: Consumption = Consumption()


class GovernmentEntitiesConfiguration(BaseModel):
    functions: GovernmentFunctions = GovernmentFunctions()
