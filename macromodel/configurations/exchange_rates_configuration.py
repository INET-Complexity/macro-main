from typing import Literal

from pydantic import BaseModel


class ExchangeRatesConfiguration(BaseModel):
    """Configuration for exchange rate determination.

    Defines the mechanism for determining exchange rates through:
    - Fixed rate settings
    - External rate sources
    - Rate update mechanisms

    The configuration supports:
    - Constant rates: Fixed exchange rates throughout simulation
    - Exogenous rates: Externally provided exchange rate paths

    Attributes:
        exchange_rate_type (Literal): Type of exchange rate mechanism to use.
            Options are "constant" (fixed rates) or "exogenous" (external rates).
            Defaults to "constant".
    """

    exchange_rate_type: Literal["constant", "exogenous"] = "constant"
