from typing import Literal, Any

from pydantic import BaseModel


class ClearingFunction(BaseModel):
    """
    The function used for clearing the credit market.
    """

    path_name: str = "clearing"
    name: Literal["NoCreditMarketClearer", "DefaultCreditMarketClearer"] = "NoCreditMarketClearer"
    parameters: dict[str, Any] = {
        "firms_max_number_of_banks_visiting": 5,
        "households_max_number_of_banks_visiting": 5,
    }


class CreditMarketFunctions(BaseModel):
    clearing: ClearingFunction = ClearingFunction()


class CreditMarketConfiguration(BaseModel):
    functions: CreditMarketFunctions = CreditMarketFunctions()
