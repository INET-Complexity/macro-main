from typing import Literal

from pydantic import BaseModel


class ExchangeRatesConfiguration(BaseModel):
    """
    The configuration settings for the exchange rates.
    """

    exchange_rate_type: Literal["constant", "exogenous"] = "constant"
