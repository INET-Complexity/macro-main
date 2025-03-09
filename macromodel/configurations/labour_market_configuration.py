from typing import Any, Literal

from pydantic import BaseModel


class Clearing(BaseModel):
    """Labor market clearing mechanism configuration.

    Defines the approach for matching workers with jobs through:
    - Employment adjustment mechanisms
    - Industry mobility settings
    - Hiring/firing behavior
    - Cost considerations

    The configuration supports:
    - Multiple clearing strategies (None, Default, Poledna)
    - Employment target handling
    - Industry switching rules
    - Wage considerations
    - Worker mobility

    Attributes:
        name (Literal): Selected clearing mechanism
        parameters (dict): Configuration parameters including:
            - compare_with_normalised_inputs (bool): Use normalized comparisons
            - round_target_employment (bool): Round employment targets
            - allow_switching_industries (bool): Allow industry changes
            - consider_reservation_wages (bool): Consider wage requirements
            - firing_cost_fraction (float): Cost of termination
            - firing_speed (float): Speed of workforce reduction
            - hiring_cost_fraction (float): Cost of hiring
            - hiring_speed (float): Speed of workforce expansion
            - individuals_quitting (bool): Allow voluntary departures
            - individuals_quitting_temperature (float): Quit decision sensitivity
            - optimised_hiring (bool): Use optimized hiring strategy
            - random_firing_probability (float): Random termination chance
            - sorted_firing (bool): Use ordered termination
        path_name (str): Module path for clearing functions
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
    """Collection of labor market function configurations.

    Aggregates the various functional components that define
    labor market behavior through:
    - Market clearing mechanisms
    - Employment adjustment functions
    - Worker-job matching algorithms

    Attributes:
        clearing (Clearing): Market clearing mechanism configuration
    """

    clearing: Clearing = Clearing()


class LabourMarketConfiguration(BaseModel):
    """Complete labor market behavior configuration.

    Defines the overall configuration for labor market operations through:
    - Function implementations
    - Behavioral parameters
    - Market clearing settings

    Attributes:
        functions (LabourMarketFunctions): Function configurations
    """

    functions: LabourMarketFunctions = LabourMarketFunctions()
