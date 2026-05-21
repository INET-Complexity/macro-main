import numpy as np
import pandas as pd

from macromodel.configurations import CountryConfiguration, ExchangeRatesConfiguration
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates


class TestCountry:
    def test__init(self, datawrapper):
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()

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

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        assert country is not None

    def test__country(self, test_country):
        assert test_country is not None

    def test_prepare_credit_market_uses_home_sales_for_household_mortgages(self):
        """Regression: household mortgages are requested for purchases, not rentals.

        The previous country-level preparation passed rental rows to household
        target-credit computation. Mortgage demand then missed actual home-sale
        buyers and households could not finance purchases correctly.
        """

        class DummyTimeSeries:
            def current(self, key):
                return [0.0]

        class DummyFirms:
            def compute_target_credit(self, estimated_growth, estimated_inflation):
                pass

        class DummyHouseholds:
            def compute_target_credit(self, current_sales):
                self.current_sales = current_sales

        class DummyBanks:
            def set_interest_rates(self, central_bank_policy_rate):
                pass

        country = object.__new__(Country)
        country.firms = DummyFirms()
        country.households = DummyHouseholds()
        country.banks = DummyBanks()
        country.economy = type("DummyEconomy", (), {"ts": DummyTimeSeries()})()
        country.central_bank = type("DummyCentralBank", (), {"ts": DummyTimeSeries()})()
        country.housing_market = type(
            "DummyHousingMarket",
            (),
            {
                "states": {
                    "current_sales": pd.DataFrame(
                        {
                            "sales_types": ["Rental", "Sell"],
                            "buyer_id": [1, 2],
                            "price_or_rent": [10.0, 100.0],
                        }
                    )
                }
            },
        )()

        country.prepare_credit_market_clearing()

        assert country.households.current_sales["sales_types"].tolist() == ["Sell"]
        assert country.households.current_sales["price_or_rent"].tolist() == [100.0]

    def test_housing_market_ratios_are_recorded_after_clearing(self):
        """Regression: observed housing ratios use post-clearing market state."""

        class DummyObservedSeries(list):
            def __init__(self, events, name):
                super().__init__()
                self.events = events
                self.name = name

            def append(self, value):
                self.events.append(self.name)
                super().append(value)

        class DummyHouseholdTimeSeries:
            def current(self, key):
                if key == "received_mortgages":
                    return np.array([100.0])
                if key == "wealth_financial_assets":
                    return np.array([0.0])
                raise KeyError(key)

        class DummyHouseholds:
            def __init__(self, events):
                self.events = events
                self.states = {}
                self.ts = DummyHouseholdTimeSeries()

            def process_housing_market_clearing(self, **kwargs):
                self.events.append("households")

        class DummyGovernmentTimeSeries:
            def current(self, key):
                return [np.array([])]

        class DummyHousingMarket:
            def __init__(self):
                self.events = []
                self.states = {
                    "properties": pd.DataFrame(),
                    "current_sales": pd.DataFrame(
                        {
                            "sales_types": ["Sell"],
                            "buyer_id": [0],
                            "price_or_rent": [100.0],
                        }
                    ),
                }
                self.ts = type(
                    "DummyHousingTimeSeries",
                    (),
                    {
                        "observed_fraction_value_price": DummyObservedSeries(self.events, "value"),
                        "observed_fraction_rent_value": DummyObservedSeries(self.events, "rent"),
                    },
                )()

            def process_housing_market_clearing(self, **kwargs):
                self.events.append("housing")

            def compute_observed_fraction_value_price(self):
                assert self.events == ["housing"]
                return np.array([1.0, 0.0])

            def compute_observed_fraction_rent_value(self):
                assert self.events == ["housing", "value"]
                return np.array([0.01, 0.0])

        country = object.__new__(Country)
        country.housing_market = DummyHousingMarket()
        country.households = DummyHouseholds(country.housing_market.events)
        country.central_government = type(
            "DummyGovernment",
            (),
            {
                "functions": {"social_housing": object()},
                "ts": DummyGovernmentTimeSeries(),
            },
        )()

        country.process_housing_market_clearing()

        assert country.housing_market.events == ["housing", "value", "rent", "households"]
