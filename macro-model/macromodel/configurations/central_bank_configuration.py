from pydantic import BaseModel
from typing import Literal


class CentralBankPolicy(BaseModel):
    """
    The policy of the central bank.
    """

    path_name: str = "policy_rate"
    name: Literal["ConstantPolicyRate", "PolednaPolicyRate"] = "ConstantPolicyRate"
    parameters: dict = {}


class CentralBankFunctions(BaseModel):
    """
    The functions used by the central bank.
    """

    policy_rate: CentralBankPolicy = CentralBankPolicy()


class CentralBankConfiguration(BaseModel):
    """
    Represents the configuration of a central bank.

    Attributes:
        functions (CentralBankFunctions): The functions used by the central bank.
    """

    functions: CentralBankFunctions = CentralBankFunctions()
