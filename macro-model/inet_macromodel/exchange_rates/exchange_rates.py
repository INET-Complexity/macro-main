import pandas as pd

from inet_macromodel.exchange_rates.exchange_rates_ts import create_exchange_rates_timeseries


class ExchangeRates:
    def __init__(
        self,
        exchange_rate_type: str,
        initial_year: int,
        country_names: list[str],
        historic_exchange_rate_data: pd.DataFrame,
    ):
        self.exchange_rate_type = exchange_rate_type
        self.initial_year = initial_year
        self.country_names = country_names
        self.historic_exchange_rate_data = historic_exchange_rate_data

        # Create the corresponding time series object
        self.ts = create_exchange_rates_timeseries(
            initial_exchange_rates=self.get_current_exchange_rates_from_usd_to_lcu(self.initial_year)
        )

    def get_current_exchange_rates_from_usd_to_lcu(self, current_year: int) -> list[float]:
        if self.exchange_rate_type == "constant":
            exchange_rate_dict = self.historic_exchange_rate_data[str(self.initial_year)].to_dict()
        elif self.exchange_rate_type == "exogenous":
            exchange_rate_dict = self.historic_exchange_rate_data[str(current_year)].to_dict()
        else:
            raise ValueError("Unknown Exchange Rates Type")

        return [exchange_rate_dict[c] for c in self.country_names + ["ROW"]]

    def set_current_exchange_rates(self, current_year: int) -> None:
        self.ts.exchange_rates.append(self.get_current_exchange_rates_from_usd_to_lcu(current_year))
