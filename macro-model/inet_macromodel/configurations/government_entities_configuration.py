from pydantic import BaseModel
from typing import Literal


class Consumption(BaseModel):
    name: Literal[
        "ConstantGovernmentConsumptionSetter", "DefaultGovernmentConsumptionSetter"
    ] = "DefaultGovernmentConsumptionSetter"
    path_name: str = "consumption"
    parameters: dict = {}


class GovernmentFunctions(BaseModel):
    consumption: Consumption = Consumption()


class GovernmentEntitiesConfiguration(BaseModel):
    functions: GovernmentFunctions = GovernmentFunctions()
