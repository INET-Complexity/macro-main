from model.timeseries import TimeSeries


def create_exchange_rates_timeseries(initial_exchange_rates: list[float]) -> TimeSeries:
    return TimeSeries(exchange_rates=initial_exchange_rates)
