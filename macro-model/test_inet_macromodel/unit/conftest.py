import yaml
import pytest
import numpy as np
import pandas as pd

from pathlib import Path

from inet_data import DataWrapper

from configurations import (
    IndividualsConfiguration,
    FirmsConfiguration,
    CentralGovernmentConfiguration,
    BankConfiguration,
)
from inet_macromodel.individuals import Individuals
from inet_macromodel.households import Households
from inet_macromodel.firms import Firms
from inet_macromodel.central_government import CentralGovernment
from inet_macromodel.government_entities import GovernmentEntities
from inet_macromodel.banks import Banks
from inet_macromodel.central_bank import CentralBank
from inet_macromodel.economy import Economy
from inet_macromodel.labour_market.labour_market import LabourMarket
from inet_macromodel.credit_market.credit_market import CreditMarket
from inet_macromodel.housing_market.housing_market import HousingMarket
from inet_macromodel.goods_market.goods_market import GoodsMarket
from inet_macromodel.rest_of_the_world import RestOfTheWorld
from inet_macromodel.country import Country
from inet_macromodel.exogenous import Exogenous

from inet_macromodel.individuals.individual_properties import ActivityStatus


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
        all_country_names=["FRA"],
        n_industries=18,
        scale=10_000,
    )
    return test_individuals


@pytest.fixture(scope="module", name="test_households")
def test_households(test_industries, test_config):
    return Households.from_data(
        country_name="FRA",
        all_country_names=["FRA"],
        year=2014,
        t_max=12,
        n_industries=len(test_industries),
        scale=1,
        data=pd.DataFrame(
            {
                "Type": np.full(18, 51),
                "Rental Income from Real Estate": np.full(18, 200),
                "Wealth": np.full(18, 200),
                "Wealth in Real Assets": np.full(18, 100),
                "Value of the Main Residence": np.full(18, 10),
                "Value of other Properties": np.full(18, 10),
                "Wealth Other Real Assets": np.full(18, 10),
                "Wealth in Deposits": np.full(18, 10),
                "Wealth in Other Financial Assets": np.full(18, 10),
                "Wealth in Financial Assets": np.full(18, 10),
                "Outstanding Balance of other Non-Mortgage Loans": np.full(18, 10),
                "Outstanding Balance of HMR Mortgages": np.full(18, 10),
                "Outstanding Balance of Mortgages on other Properties": np.full(18, 10),
                "Debt": np.full(18, 10),
                "Debt Installments": np.full(18, 0.0),
                "Net Wealth": np.full(18, 10),
                "Regular Social Transfers": np.full(18, 10),
                "Employee Income": np.full(18, 100),
                "Income": np.full(18, 110),
                "Income from Financial Assets": np.full(18, 0.0),
                "Rent Paid": np.zeros(18),
                "Rent Imputed": np.full(18, 10.0),
                "Saving Rate": np.full(18, 0.2),
                "Corresponding Individuals ID": np.arange(18),
                "Corresponding Bank ID": np.full(18, 0),
                "Corresponding Inhabited House ID": np.arange(18),
                "Corresponding Property Owner": np.arange(18),
                "Tenure Status of the Main Residence": np.full(18, 1),
            }
        ),
        corr_individuals=pd.DataFrame(
            {
                "Corresponding Individuals ID": np.arange(18),
            }
        ),
        individual_ages=np.full(18, 35),
        corr_additionally_owned_properties=pd.DataFrame(
            {
                "Corresponding Additionally Owned Properties ID": np.zeros(18),
            }
        ),
        corr_renters=pd.DataFrame(
            {
                "Corresponding Renters ID": np.zeros(18),
            }
        ),
        consumption_weights=pd.DataFrame(
            {
                "Household Consumption Weights": np.full(18, 1.0 / 18),
            }
        ),
        consumption_weights_by_income=pd.DataFrame(
            {
                "Q1": np.full(18, 1.0 / 18),
                "Q2": np.full(18, 1.0 / 18),
                "Q3": np.full(18, 1.0 / 18),
                "Q4": np.full(18, 1.0 / 18),
                "Q5": np.full(18, 1.0 / 18),
            }
        ),
        initial_industry_consumption=np.array([10.0, 20.0]),
        saving_rates_model=None,
        social_transfers_model=None,
        wealth_distribution_model=None,
        value_added_tax=0.1,
        coefficient_fa_income=pd.DataFrame([0.01]),
        config=test_config["FRA"]["households"],
        init_config=test_config["init"]["FRA"]["households"],
    )


@pytest.fixture(scope="module", name="test_firms")
def test_firms(datawrapper):
    country = datawrapper.synthetic_countries["FRA"]

    firm_config = FirmsConfiguration()

    firms = Firms.from_pickled_agent(
        synthetic_firms=country.firms,
        configuration=firm_config,
        country_name="FRA",
        all_country_names=["FRA"],
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
        all_country_names=["FRA"],
        taxes_net_subsidies=taxes_less_subsidies,
        tax_data=country.tax_data,
        n_industries=n_industries,
        number_of_unemployed_individuals=n_unemployed,
    )

    return central_government


@pytest.fixture(scope="module", name="test_government_entities")
def test_government_entities(test_industries, test_config):
    return GovernmentEntities.from_data(
        country_name="FRA",
        all_country_names=["FRA"],
        year=2014,
        t_max=12,
        n_industries=len(test_industries),
        data=pd.DataFrame(
            {
                "Consumption in USD": np.full(18, 100.0),
                "Consumption in LCU": np.full(18, 100.0),
            }
        ),
        number_of_entities=3,
        government_consumption_model=None,
        config=test_config["FRA"]["government_entities"],
    )


@pytest.fixture(scope="module", name="test_central_bank")
def test_central_bank(test_industries, test_config):
    central_bank = CentralBank.from_data(
        country_name="FRA",
        all_country_names=["FRA"],
        year=2014,
        t_max=12,
        n_industries=len(test_industries),
        data=pd.DataFrame(
            {
                "Debt to the ROW": [0.0],
                "Equity": [100.0],
                "Policy Rate": [0.02],
            }
        ),
        config=test_config["FRA"]["central_bank"],
    )
    return central_bank


@pytest.fixture(scope="module", name="test_economy")
def test_economy(test_industries, test_industry_vectors, test_config):
    return Economy.from_data(
        country_name="FRA",
        all_country_names=["FRA"],
        year=2014,
        t_max=12,
        n_industries=len(test_industries),
        initial_firm_prices=np.full(18, 1.0),
        initial_firm_total_sales=300.0,
        initial_firm_total_used_ii=100.0,
        initial_total_taxes_on_products=3.0,
        initial_change_in_firm_stock_inventories=40.0,
        initial_total_operating_surplus_plus_wages=70.0,
        initial_individual_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
        initial_cpi_inflation=0.0,
        initial_ppi_inflation=0.0,
        initial_nominal_house_price_index_growth=0.0,
        initial_real_rent_paid=np.array([10.0, 20.0]),
        initial_imp_rent_paid=np.array([10.0, 20.0]),
        initial_hh_rental_income=np.array([10.0, 20.0]),
        initial_hh_consumption=100.0,
        initial_gov_consumption=50.0,
        initial_cg_rent_received=20.0,
        initial_cg_taxes_rental_income=5.0,
        initial_sectoral_growth=np.zeros(18),
        initial_exports=np.array([10.0, 20.0]),
        initial_exports_by_country={"FRA": np.array([1.0, 2.0])},
        initial_imports=np.array([10.0, 20.0]),
        initial_imports_by_country={"FRA": np.array([1.0, 2.0])},
        export_taxes=0.1,
        config=test_config["FRA"]["economy"],
    )


@pytest.fixture(scope="module", name="test_row")
def test_row(test_industries, test_config):
    return RestOfTheWorld.from_data(
        country_name="ROW",
        all_country_names=["FRA"],
        year=2014,
        t_max=12,
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
        year=test_config["model"]["year"]["value"],
        t_max=test_config["model"]["t_max"]["value"],
        n_industries=len(test_config["model"]["industries"]["value"]),
        initial_individual_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
        initial_individual_employment_industry=np.array([0, 1]),
        config=test_config["FRA"]["labour_market"],
    )


@pytest.fixture(scope="module", name="test_credit_market")
def test_credit_market(test_industries, test_config):
    return CreditMarket.from_data(
        country_name="ROW",
        year=2014,
        t_max=12,
        n_industries=len(test_industries),
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
        year=2014,
        t_max=12,
        n_industries=len(test_industries),
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


@pytest.fixture(scope="module", name="test_exogenous")
def test_exogenous():
    iot_industry_data = {}
    for i in range(18):
        iot_industry_data[("Output in LCU", i)] = (np.ones(12),)
        iot_industry_data[("Profits", i)] = (np.ones(12),)
        iot_industry_data[("Household Consumption in LCU", i)] = (np.ones(12),)
        iot_industry_data[("Government Consumption in LCU", i)] = (np.ones(12),)
        iot_industry_data[("Imports in LCU", i)] = (np.ones(12),)
        iot_industry_data[("Exports in LCU", i)] = (np.ones(12),)
    return Exogenous(
        country_name="FRA",
        initial_year=2014,
        t_max=20,
        log_inflation=pd.DataFrame(
            data={
                "Real CPI Inflation": [0.01, 0.02],
                "Real PPI Inflation": [0.01, 0.02],
            },
            index=["2014-1", "2014-2"],
        ),
        sectoral_growth=pd.DataFrame(
            data={g: [0.01, 0.0] for g in range(18)},
            index=["2014-1", "2014-2"],
        ),
        unemployment_rate=pd.DataFrame(
            data={"Unemployment Rate": [0.1, 0.12]},
            index=["2014-1", "2014-2"],
        ),
        vacancy_rate=pd.DataFrame(
            data={"Vacancy Rate": [0.1, 0.12]},
            index=["2014-1", "2014-2"],
        ),
        house_price_index=pd.DataFrame(
            data={
                "Real House Price Index Growth": [0.01, 0.02],
                "Nominal House Price Index Growth": [0.01, 0.02],
            },
            index=["2014-1", "2014-2"],
        ),
        total_firm_deposits_and_debt=pd.DataFrame(
            data={
                "Total Deposits": [100.0, 110.0],
                "Total Debt": [50.0, 60.0],
            },
            index=["2014-1", "2014-2"],
        ),
        iot_industry_data=pd.DataFrame(
            data=iot_industry_data,
            index=[
                "2014-1",
                "2014-2",
                "2014-3",
                "2014-4",
                "2014-5",
                "2014-6",
                "2014-7",
                "2014-8",
                "2014-9",
                "2014-10",
                "2014-11",
                "2014-12",
            ],
            columns=pd.MultiIndex.from_product(
                [
                    [
                        "Output in LCU",
                        "Profits",
                        "Household Consumption in LCU",
                        "Government Consumption in LCU",
                        "Imports in LCU",
                        "Exports in LCU",
                    ],
                    range(18),
                ]
            ),
        ),
        all_country_names=["FRA"],
        exchange_rates_data=pd.DataFrame(
            data={"Exchange Rate": [1.0, 1.0, 1.0]},
            index=["2013-12", "2014-1", "2014-2"],
        ).T,
    )


@pytest.fixture(scope="module", name="test_banks")
def test_banks(datawrapper):
    synthetic_banks = datawrapper.synthetic_countries["FRA"].banks

    test_banks = Banks.from_pickled_agent(
        synthetic_banks=synthetic_banks,
        configuration=BankConfiguration(),
        policy_rate_markup=0.1,
        long_term_ir=0.1,
        n_industries=18,
        country_name="FRA",
        scale=10000,
        all_country_names=["FRA"],
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
        year=test_config["model"]["year"]["value"],
        t_max=test_config["model"]["t_max"]["value"],
        n_industries=len(test_config["model"]["industries"]["value"]),
        trade_proportions=pd.DataFrame(),
        config=test_config["goods_market"]["goods_market"],
    )
    goods_market.functions["clearing"].initiate_agents(
        n_industries=len(test_config["model"]["industries"]["value"]),
        goods_market_participants={
            "FRA": [test_firms, test_households],
            "ROW": [test_row],
        },
    )
    goods_market.functions["clearing"].initiate_the_supply_chain(
        initial_supply_chain=None,
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
        year=test_config["model"]["year"]["value"],
        t_max=test_config["model"]["t_max"]["value"],
        n_industries=len(test_config["model"]["industries"]["value"]),
        config=test_config["goods_market"]["goods_market"],
        trade_proportions=pd.DataFrame(),
    )
    goods_market.functions["clearing"].initiate_agents(
        n_industries=len(test_config["model"]["industries"]["value"]),
        goods_market_participants={
            "FRA": [test_firms, test_households],
            "ROW": [test_row],
        },
    )
    goods_market.functions["clearing"].initiate_the_supply_chain(
        initial_supply_chain=None,
    )
    return goods_market


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
        year=2014,
        t_max=12,
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
