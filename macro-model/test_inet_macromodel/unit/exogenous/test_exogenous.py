from configurations import ExchangeRatesConfiguration
from exchange_rates import ExchangeRates
from exogenous import Exogenous


class TestExogenous:
    def test_create(self, test_exogenous):
        assert test_exogenous.iot_industry_data_during.index[0].year == 2014
