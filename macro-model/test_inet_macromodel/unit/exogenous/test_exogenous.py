from configurations import ExchangeRatesConfiguration
from exchange_rates import ExchangeRates
from exogenous import Exogenous


class TestExogenous:
    def test_create(self, datawrapper):
        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates_df = datawrapper.exchange_rates
        initial_year = 2014
        country_names = ["FRA"]

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=exchange_rates_df,
            exchange_rate_config=exchange_rates_config,
            initial_year=initial_year,
            country_names=country_names,
        )

        country = datawrapper.synthetic_countries["FRA"]

        t_max = 20

        exogenous = Exogenous.from_pickled_agent(
            synthetic_country=country,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA"],
            initial_year=2014,
            t_max=t_max,
        )

        assert exogenous.iot_industry_data_during.index[0].year == 2014
