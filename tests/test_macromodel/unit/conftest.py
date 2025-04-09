import os.path
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from macro_data import DataWrapper
from macro_data.configuration.countries import Country as CountryCode
from macro_data.configuration.region import Region
from macro_data.configuration_utils import default_data_configuration
from macromodel.agents.banks import Banks
from macromodel.agents.central_bank import CentralBank
from macromodel.agents.central_government import CentralGovernment
from macromodel.agents.firms import Firms
from macromodel.agents.government_entities import GovernmentEntities
from macromodel.agents.households import Households
from macromodel.agents.individuals import Individuals
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import (
    BanksConfiguration,
    CentralBankConfiguration,
    CentralGovernmentConfiguration,
    EconomyConfiguration,
    ExchangeRatesConfiguration,
    FirmsConfiguration,
    GoodsMarketConfiguration,
    GovernmentEntitiesConfiguration,
    HouseholdsConfiguration,
    IndividualsConfiguration,
    RestOfTheWorldConfiguration,
)
from macromodel.country import Country
from macromodel.economy import Economy
from macromodel.exchange_rates import ExchangeRates
from macromodel.exogenous import Exogenous
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.goods_market.goods_market import GoodsMarket
from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.labour_market.labour_market import LabourMarket
from macromodel.rest_of_the_world import RestOfTheWorld


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
    industries = datawrapper.industries
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
        synthetic_country=country,
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
        industries=datawrapper.industries,
    )

    return firms


@pytest.fixture(scope="module", name="test_central_government")
def test_central_government(datawrapper, test_individuals):
    country = datawrapper.synthetic_countries["FRA"]
    synthetic_central_government = country.central_government

    central_government_config = CentralGovernmentConfiguration()

    taxes_less_subsidies = country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

    n_industries = datawrapper.n_industries

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

    n_industries = datawrapper.n_industries

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
    test_firms,
    test_households,
    test_individuals,
    test_government_entities,
    test_central_government,
    test_exogenous,
    datawrapper,
):
    synthetic_country = datawrapper.synthetic_countries["FRA"]

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
        industry_vectors=synthetic_country.industry_data["industry_vectors"],
    )


@pytest.fixture(scope="module", name="test_row")
def test_row(datawrapper):
    countries_with_row = datawrapper.all_country_names

    row_configuration = RestOfTheWorldConfiguration()
    rest_of_the_world = RestOfTheWorld.from_pickled_row(
        country_name="ROW",
        all_country_names=countries_with_row,
        n_industries=datawrapper.n_industries,
        synthetic_row=datawrapper.synthetic_rest_of_the_world,
        configuration=row_configuration,
        calibration_data_before=datawrapper.calibration_before,
        calibration_data_during=datawrapper.calibration_during,
    )

    return rest_of_the_world


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
        country_name="FRA",
        st_loans=np.zeros((3, 1, 18)),
        lt_loans=np.zeros((3, 1, 18)),
        cons_loans=np.zeros((3, 1, 18)),
        mort_loans=np.zeros((3, 1, 18)),
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
        n_industries=18,
        country_name="FRA",
        scale=10000,
        all_country_names=["FRA", "ROW"],
    )

    test_banks.set_interest_rates(central_bank_policy_rate=0.02)
    return test_banks


# @pytest.fixture(scope="module", name="test_default_goods_market")
# def test_default_goods_market(
#     test_firms,
#     test_households,
#     test_row,
#     test_config,
# ):
#     goods_market = GoodsMarket.from_data(
#         n_industries=len(test_config["model"]["industries"]["value"]),
#         trade_proportions=pd.DataFrame(),
#         configuration=GoodsMarketConfiguration(),
#         goods_market_participants={
#             "FRA": [test_firms, test_households],
#             "ROW": [test_row],
#         },
#     )
#     return goods_market


@pytest.fixture(scope="module", name="test_goods_market")
def test_goods_market(
    test_firms,
    test_households,
    test_row,
    test_config,
    datawrapper,
):
    goods_market = GoodsMarket.from_data(
        n_industries=len(test_config["model"]["industries"]["value"]),
        configuration=GoodsMarketConfiguration(),
        goods_market_participants={
            "FRA": [test_firms, test_households],
            "ROW": [test_row],
        },
        origin_trade_proportions=datawrapper.origin_trade_proportions.values,
        destin_trade_proportions=datawrapper.destination_trade_proportions.values,
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
        running_multiple_countries=False,
    )


@pytest.fixture(scope="module", name="datawrapper")
def instantiate_datawrapper() -> DataWrapper:
    data_config = default_data_configuration(countries=["FRA"])
    raw_data_path = Path(__file__).parent.parent.parent / "test_macro_data" / "unit" / "sample_raw_data"
    return DataWrapper.from_config(data_config, raw_data_path, single_hfcs_survey=True)


@pytest.fixture(scope="module", name="allind_datawrapper")
def instantiate_allind_datawrapper() -> DataWrapper:
    data_config = default_data_configuration(countries=["FRA"], aggregate_industries=False)
    raw_data_path = Path(__file__).parent.parent.parent / "test_macro_data" / "unit" / "sample_raw_data"
    return DataWrapper.from_config(data_config, raw_data_path, single_hfcs_survey=True)


@pytest.fixture(scope="module", name="can_disagg_datawrapper")
def instantiate_can_disagg_datawrapper() -> DataWrapper:
    data_config = default_data_configuration(
        countries=["CAN"],
        aggregate_industries=False,
        proxy_country_dict={"CAN": "FRA"},
        use_disagg_can_2014_reader=True,
    )
    raw_data_path = Path(__file__).parent.parent.parent / "test_macro_data" / "unit" / "sample_raw_data"
    return DataWrapper.from_config(data_config, raw_data_path, single_hfcs_survey=True)


@pytest.fixture(scope="module", name="can_provincial_datawrapper")
def instantiate_can_provincial_datawrapper() -> DataWrapper:
    pkl_path = (
        Path(__file__).parent.parent.parent / "test_macro_data" / "unit" / "sample_raw_data" / "canada_provincial.pkl"
    )

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data_wrapper = pkl.load(f)
        return data_wrapper
    else:

        data_config = default_data_configuration(
            countries=["CAN"],
            aggregate_industries=False,
            proxy_country_dict={"CAN": "FRA"},
            use_disagg_can_2014_reader=True,
        )

        data_config.can_disaggregation = False
        data_config.aggregate_industries = False
        data_config.prune_date = None
        data_config.seed = 0

        base_config = data_config.country_configs[CountryCode("CAN")]
        base_config.single_firm_per_industry = True
        base_config.single_bank = True
        base_config.single_government_entity = True

        base_config.firms_configuration.constructor = "Default"

        base_config.scale = 1000

        # Define Canadian provinces
        provinces = [
            Region.from_code("CAN_AB", "Alberta"),
            Region.from_code("CAN_BC", "British Columbia"),
            Region.from_code("CAN_MB", "Manitoba"),
            Region.from_code("CAN_NB", "New Brunswick"),
            Region.from_code("CAN_NL", "Newfoundland and Labrador"),
            Region.from_code("CAN_NS", "Nova Scotia"),
            Region.from_code("CAN_ON", "Ontario"),
            Region.from_code("CAN_PE", "Prince Edward Island"),
            Region.from_code("CAN_QC", "Quebec"),
            Region.from_code("CAN_SK", "Saskatchewan"),
        ]

        # Add configurations for all provinces
        for province in provinces:
            data_config.country_configs[province] = base_config
            data_config.country_configs[province].eu_proxy_country = CountryCode("FRA")

        data_config.aggregation_structure = {CountryCode("CAN"): provinces}

        raw_data_path = Path(__file__).parent.parent.parent / "test_macro_data" / "unit" / "sample_raw_data"
        data_wrapper = DataWrapper.from_config(data_config, raw_data_path, single_hfcs_survey=True)

        with open(pkl_path, "wb") as f:
            pkl.dump(data_wrapper, f)

    return data_wrapper
