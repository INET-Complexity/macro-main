from pydantic import BaseModel
from typing import Literal


class ExchangeRatesConfiguration(BaseModel):
    """
    The configuration settings for the exchange rates.
    """

    exchange_rate_type: Literal["constant", "exogenous"] = "constant"
