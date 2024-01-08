from typing import Literal

from pydantic import BaseModel


class CentralBankPolicy(BaseModel):
    """
    The policy of the central bank.
    """

    path_name: str = "policy_rate"
    name: Literal["ConstantPolicyRate"] = "ConstantPolicyRate"


class CentralBankFunctions(BaseModel):
    """
    The functions used by the central bank.
    """

    policy: CentralBankPolicy = CentralBankPolicy()


class CentralBankConfiguration(BaseModel):
    """
    Represents the configuration of a central bank.

    Attributes:
        functions (CentralBankFunctions): The functions used by the central bank.
    """

    functions: CentralBankFunctions = CentralBankFunctions()
