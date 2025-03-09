from typing import Literal

from pydantic import BaseModel


class CentralBankPolicy(BaseModel):
    """Central bank policy rate determination configuration.

    Defines the mechanism for setting monetary policy through:
    - Interest rate determination
    - Policy rule implementation
    - Rate adjustment mechanisms

    The configuration supports:
    - Constant policy rates: Fixed interest rates
    - Poledna policy rates: Dynamic rate adjustment based on economic conditions

    Attributes:
        path_name (str): Module path for policy rate functions
        name (Literal): Selected policy mechanism ("ConstantPolicyRate" or "PolednaPolicyRate")
        parameters (dict): Configuration parameters for policy implementation
    """

    path_name: str = "policy_rate"
    name: Literal["ConstantPolicyRate", "PolednaPolicyRate"] = "ConstantPolicyRate"
    parameters: dict = {}


class CentralBankFunctions(BaseModel):
    """Collection of central bank function configurations.

    Aggregates the various functional components that define
    central bank operations through:
    - Monetary policy implementation
    - Interest rate management
    - Policy rule execution

    Attributes:
        policy_rate (CentralBankPolicy): Policy rate determination configuration
    """

    policy_rate: CentralBankPolicy = CentralBankPolicy()


class CentralBankConfiguration(BaseModel):
    """Complete central bank behavior configuration.

    Defines the overall configuration for central bank operations through:
    - Policy frameworks
    - Operational procedures
    - Monetary tools
    - Implementation mechanisms

    The configuration determines how the central bank:
    - Sets interest rates
    - Implements monetary policy
    - Responds to economic conditions
    - Manages policy transmission

    Attributes:
        functions (CentralBankFunctions): Collection of function configurations
            that define central bank behavior
    """

    functions: CentralBankFunctions = CentralBankFunctions()
