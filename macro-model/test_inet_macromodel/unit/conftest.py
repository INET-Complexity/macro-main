import numpy as np
import pandas as pd
import pytest
import yaml
from inet_data import DataWrapper
from pathlib import Path

from inet_macromodel.banks import Banks
from inet_macromodel.central_bank import CentralBank
from inet_macromodel.central_government import CentralGovernment
from inet_macromodel.configurations import (
    IndividualsConfiguration,
    FirmsConfiguration,
    CentralGovernmentConfiguration,
    BanksConfiguration,
    HouseholdsConfiguration,
    ExchangeRatesConfiguration,
    GovernmentEntitiesConfiguration,
    EconomyConfiguration,
    CentralBankConfiguration,
    GoodsMarketConfiguration,
)
from inet_macromodel.country import Country
from inet_macromodel.credit_market.credit_market import CreditMarket
from inet_macromodel.economy import Economy
from inet_macromodel.exchange_rates import ExchangeRates
from inet_macromodel.exogenous import Exogenous
from inet_macromodel.firms import Firms
from inet_macromodel.goods_market.goods_market import GoodsMarket
from inet_macromodel.government_entities import GovernmentEntities
from inet_macromodel.households import Households
from inet_macromodel.housing_market.housing_market import HousingMarket
from inet_macromodel.individuals import Individuals
from inet_macromodel.individuals.individual_properties import ActivityStatus
from inet_macromodel.labour_market.labour_market import LabourMarket
from inet_macromodel.rest_of_the_world import RestOfTheWorld


@pytest.fixture(scope="module", name="test_config")
def test_config():
    name = "default_unit_test"
    config = yaml.safe_load(open(Path(__file__).parent / (name + ".yaml"), "r"))
    return config


@pytest.fixture(scope="module", name="test_industries")
def test_industries():
    return [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R_S",
    ]


@pytest.fixture(scope="module", name="test_industry_vectors")
def test_industry_vectors():
    return pd.DataFrame(
        {
            "Output": np.full(18, 1.0),
            "Value Added": np.full(18, 0.5),
            "Household Consumption": np.full(18, 0.1),
            "Household Consumption Weights": np.full(18, 1.0 / 18),
            "Government Consumption": np.full(18, 0.1),
            "Government Consumption Weights": np.full(18, 1.0 / 18),
            "Labour Compensation": np.full(18, 0.1),
            "Capital Compensation": np.full(18, 0.1),
            "Capital Stock": np.full(18, 0.1),
            "Taxes Less Subsidies Rates": np.full(18, 0.0),
            "Average Initial Price": np.full(18, 1.0),
        }
    )


@pytest.fixture(scope="module", name="test_individuals")
def test_individuals(datawrapper):
    synthetic_population = datawrapper.synthetic_countries["FRA"].population

    test_individuals = Individuals.from_pickled_agent(
        synthetic_population=synthetic_population,
        configuration=IndividualsConfiguration(),
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        n_industries=18,
        scale=10_000,
    )
    return test_individuals


@pytest.fixture(scope="module", name="test_households")
def test_households(datawrapper):
    data_config = datawrapper.configuration
    industries = data_config.industries
    country = datawrapper.synthetic_countries["FRA"]
    population = country.population
    initial_consumption_by_industry = country.industry_data["industry_vectors"]["Household Consumption in LCU"]
    scale = data_config.country_configs["FRA"].scale

    households = Households.from_pickled_agent(
        synthetic_population=population,
        configuration=HouseholdsConfiguration(),
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        industries=industries,
        initial_consumption_by_industry=initial_consumption_by_industry,
        value_added_tax=country.tax_data.value_added_tax,
        scale=scale,
    )

    return households


@pytest.fixture(scope="module", name="test_firms")
def test_firms(datawrapper):
    country = datawrapper.synthetic_countries["FRA"]

    firm_config = FirmsConfiguration()

    firms = Firms.from_pickled_agent(
        synthetic_firms=country.firms,
        configuration=firm_config,
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        goods_criticality_matrix=country.goods_criticality_matrix,
        average_initial_price=country.industry_data["industry_vectors"]["Average Initial Price"].values,
    )

    return firms


@pytest.fixture(scope="module", name="test_central_government")
def test_central_government(datawrapper, test_individuals):
    country = datawrapper.synthetic_countries["FRA"]
    synthetic_central_government = country.central_government

    central_government_config = CentralGovernmentConfiguration()

    taxes_less_subsidies = country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

    n_industries = len(datawrapper.configuration.industries)

    n_unemployed = np.sum(test_individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED)

    central_government = CentralGovernment.from_pickled_agent(
        synthetic_central_government=synthetic_central_government,
        configuration=central_government_config,
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        taxes_net_subsidies=taxes_less_subsidies,
        tax_data=country.tax_data,
        n_industries=n_industries,
        number_of_unemployed_individuals=n_unemployed,
    )

    return central_government


@pytest.fixture(scope="module", name="test_government_entities")
def test_government_entities(datawrapper):
    country = datawrapper.synthetic_countries["FRA"]

    n_industries = len(datawrapper.configuration.industries)

    government_entities_config = GovernmentEntitiesConfiguration()

    government_entities = GovernmentEntities.from_pickled_agent(
        synthetic_government_entities=country.government_entities,
        configuration=government_entities_config,
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        n_industries=n_industries,
    )
    return government_entities


@pytest.fixture(scope="module", name="test_central_bank")
def test_central_bank(datawrapper):
    synthetic_central_bank = datawrapper.synthetic_countries["FRA"].central_bank

    central_bank = CentralBank.from_pickled_agent(
        synthetic_central_bank=synthetic_central_bank,
        configuration=CentralBankConfiguration(),
        country_name="FRA",
        all_country_names=["FRA"],
        n_industries=18,
    )
    return central_bank


@pytest.fixture(scope="module", name="test_economy")
def test_economy(
    test_firms, test_households, test_individuals, test_government_entities, test_central_government, test_exogenous
):
    return Economy.from_agents(
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        economy_configuration=EconomyConfiguration(),
        individuals=test_individuals,
        households=test_households,
        firms=test_firms,
        government_entities=test_government_entities,
        central_government=test_central_government,
        exogenous=test_exogenous,
        initial_sentiment=0.0,
    )


@pytest.fixture(scope="module", name="test_row")
def test_row(test_industries, test_config):
    return RestOfTheWorld.from_data(
        country_name="ROW",
        all_country_names=["FRA", "ROW"],
        n_industries=len(test_industries),
        data=pd.DataFrame(
            {
                "Exports": np.full(18, 0.2),
                "Imports in USD": np.full(18, 0.1),
                "Imports in LCU": np.full(18, 0.1),
                "Price in USD": np.full(18, 1.0),
                "Price in LCU": np.full(18, 1.0),
            }
        ),
        config=test_config["ROW"]["ROW"],
        row_exports_model=None,
        row_imports_model=None,
        average_country_ppi_inflation=0.0,
    )


@pytest.fixture(scope="module", name="test_labour_market")
def test_labour_market(test_config):
    return LabourMarket.from_data(
        country_name="FRA",
        n_industries=len(test_config["model"]["industries"]["value"]),
        initial_individual_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
        initial_individual_employment_industry=np.array([0, 1]),
        config=test_config["FRA"]["labour_market"],
    )


@pytest.fixture(scope="module", name="test_credit_market")
def test_credit_market(test_industries, test_config):
    return CreditMarket.from_data(
        country_name="ROW",
        data=pd.DataFrame(
            {
                "loan_type": [],
                "loan_value": [],
                "loan_bank_id": [],
                "loan_recipient_id": [],
            }
        ),
        config=test_config["FRA"]["credit_market"],
    )


@pytest.fixture(scope="module", name="test_housing_market")
def test_housing_market(test_industries, test_config):
    return HousingMarket.from_data(
        country_name="ROW",
        scale=1,
        data=pd.DataFrame(
            {
                "House ID": [0],
                "Value": [100.0],
                "Rent": [1.0],
                "Corresponding Inhabitant Household ID": [0],
                "Corresponding Owner Household ID": [0],
                "Is Owner-Occupied": [1],
            }
        ),
        config=test_config["FRA"]["housing_market"],
    )


@pytest.fixture(scope="module", name="test_banks")
def test_banks(datawrapper):
    synthetic_banks = datawrapper.synthetic_countries["FRA"].banks

    test_banks = Banks.from_pickled_agent(
        synthetic_banks=synthetic_banks,
        configuration=BanksConfiguration(),
        policy_rate_markup=0.1,
        long_term_ir=0.1,
        n_industries=18,
        country_name="FRA",
        scale=10000,
        all_country_names=["FRA", "ROW"],
    )

    test_banks.set_interest_rates(central_bank_policy_rate=0.02)
    return test_banks


@pytest.fixture(scope="module", name="test_default_goods_market")
def test_default_goods_market(
    test_firms,
    test_households,
    test_row,
    test_config,
):
    goods_market = GoodsMarket.from_data(
        n_industries=len(test_config["model"]["industries"]["value"]),
        trade_proportions=pd.DataFrame(),
        configuration=GoodsMarketConfiguration(),
        goods_market_participants={
            "FRA": [test_firms, test_households],
            "ROW": [test_row],
        },
    )
    return goods_market


@pytest.fixture(scope="module", name="test_goods_market")
def test_goods_market(
    test_firms,
    test_households,
    test_row,
    test_config,
):
    goods_market = GoodsMarket.from_data(
        n_industries=len(test_config["model"]["industries"]["value"]),
        configuration=GoodsMarketConfiguration(),
        trade_proportions=pd.DataFrame(),
        goods_market_participants={
            "FRA": [test_firms, test_households],
            "ROW": [test_row],
        },
    )

    return goods_market


@pytest.fixture(scope="module", name="test_exogenous")
def test_exogenous(datawrapper):
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
        all_country_names=["FRA", "ROW"],
        initial_year=2014,
        t_max=t_max,
    )

    return exogenous


@pytest.fixture(scope="module", name="test_country")
def test_country(
    test_firms,
    test_individuals,
    test_households,
    test_central_government,
    test_government_entities,
    test_banks,
    test_central_bank,
    test_economy,
    test_labour_market,
    test_credit_market,
    test_housing_market,
    test_exogenous,
):
    return Country(
        country_name="FRA",
        scale=1,
        individuals=test_individuals,
        households=test_households,
        firms=test_firms,
        central_government=test_central_government,
        government_entities=test_government_entities,
        banks=test_banks,
        central_bank=test_central_bank,
        economy=test_economy,
        labour_market=test_labour_market,
        credit_market=test_credit_market,
        housing_market=test_housing_market,
        exogenous=test_exogenous,
    )


@pytest.fixture(scope="module", name="datawrapper")
def read_data():
    pickle_path = Path(__file__).parent.parent / "pickled_data" / "agents.pkl"
    return DataWrapper.init_from_pickle(pickle_path)
