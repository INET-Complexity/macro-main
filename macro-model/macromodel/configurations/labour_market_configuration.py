from pydantic import BaseModel
from typing import Literal, Any


class Clearing(BaseModel):
    """
    The function for clearing the labour market.
    """

    name: Literal["NoLabourMarketClearer", "DefaultLabourMarketClearer", "PolednaLabourMarketClearer"] = (
        "PolednaLabourMarketClearer"
    )
    parameters: dict[str, Any] = {
        "compare_with_normalised_inputs": True,
        "round_target_employment": True,
        "allow_switching_industries": True,
        "consider_reservation_wages": True,
        "firing_cost_fraction": 0.0,
        "firing_speed": 1.0,
        "hiring_cost_fraction": 0.0,
        "hiring_speed": 1.0,
        "individuals_quitting": False,
        "individuals_quitting_temperature": 1.0,
        "optimised_hiring": True,
        "random_firing_probability": 0.0,
        "sorted_firing": True,
    }
    path_name: str = "clearing"


class LabourMarketFunctions(BaseModel):
    clearing: Clearing = Clearing()


class LabourMarketConfiguration(BaseModel):
    functions: LabourMarketFunctions = LabourMarketFunctions()
