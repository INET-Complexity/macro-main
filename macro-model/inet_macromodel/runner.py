import os
import yaml
import h5py
import numpy as np
import pandas as pd
import skops.io as sio

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from .firms import Firms
from .banks import Banks
from .economy import Economy
from .country import Country
from .households import Households
from .individuals import Individuals
from .central_bank import CentralBank
from .exogenous.exogenous import Exogenous
from .central_government import CentralGovernment
from .government_entities import GovernmentEntities
from .exchange_rates.exchange_rates import ExchangeRates
from .individuals.individual_properties import ActivityStatus
from .rest_of_the_world.rest_of_the_world import RestOfTheWorld

from .goods_market.goods_market import GoodsMarket
from .credit_market.credit_market import CreditMarket
from .labour_market.labour_market import LabourMarket
from .housing_market.housing_market import HousingMarket

from .timestep import Timestep

from typing import Optional, Any
import logging


class Runner:
    def __init__(
        self,
        config_path: Path,
        processed_data_path: Path,
        output_path: Path,
    ):
        # Parameters
        self.config = self.process_config(config_path)
        self.config_string = self.yaml_to_string(config_path)
        self.processed_data_path = processed_data_path
        self.output_path = output_path

        # Model settings
        self.t_max = self.config["model"]["t_max"]["value"]
        self.n_industries = len(self.config["model"]["industries"]["value"])
        self.scale = self.config["model"]["scale"]["value"]

        # Time steps
        self.timestep = Timestep(
            year=self.config["model"]["year"]["value"],
            month=1,
        )

        # Agents
        self.countries = None
        self.row = None
        self.goods_market = None

        # Exchange rates
        self.exchange_rates = None

        # Random seed
        self.random_seed = None

    def run(self, random_seed: int = 0) -> None:
        self.set_random_seed(random_seed)
        self.set_exchange_rates()
        self.initialise_agents()
        logging.info("Starting simulation")
        logging.info("Countries: %s", self.config["model"]["country_names"]["value"])
        logging.info("Timesteps: %d", self.t_max)
        logging.info("Scale: %d", self.scale)
        logging.info("Random seed: %d", self.random_seed)
        with tqdm(range(1, self.t_max), desc="Running...") as tqdm_obj:
            for _ in tqdm_obj:
                self.iterate()
        logging.info("Simulation finished")
        logging.info("Saving data")
        self.save()
        logging.info("Data saved")

    def iterate(self) -> None:
        logging.info("Date: %d/%d", self.timestep.month, self.timestep.year)
        # Setting exchange rates
        self.exchange_rates.set_current_exchange_rates(current_year=self.timestep.year)

        # Before goods market clearing
        for ind, country in enumerate(self.countries.values()):
            logging.info("Country: %s", country.country_name)
            country.initialisation_phase(exchange_rate_usd_to_lcu=self.exchange_rates.ts.current("exchange_rates")[ind])
            country.estimation_phase()
            country.target_setting_phase()
            country.clear_labour_market()
            country.update_planning_metrics()

            # Clearing the housing and the credit market
            logging.info("Clearing the housing and the credit market")
            country.prepare_housing_market_clearing()
            country.clear_housing_market()
            country.prepare_credit_market_clearing()
            country.clear_credit_market()
            country.process_housing_market_clearing()
            country.process_credit_market_clearing()

            # Prepare goods market clearing
            logging.info("Prepare goods market clearing")
            country.prepare_goods_market_clearing()

        # Prepare goods market clearing
        logging.info("Prepare goods market clearing (ROW)")
        self.row.update_planning_metrics(
            average_country_ppi_inflation=np.mean(
                [self.countries[c].economy.ts.current("ppi_inflation")[0] for c in self.countries.keys()]
            ),
        )

        logging.info("Clearing the goods market")
        # Clearing the goods market
        self.goods_market.prepare()
        self.goods_market.clear()
        self.goods_market.record()

        logging.info("Updating metrics")
        # After goods market clearing
        self.row.record_bought_goods()
        for country in self.countries.values():
            country.update_realised_metrics()
            country.update_population_structure()

        # Next month
        self.timestep.step()

    @staticmethod
    def process_config(config_path: Path) -> dict[str, Any]:
        config = yaml.safe_load(open(config_path, "r"))

        # Handle init countries
        keys = list(config["init"].keys())
        for key in keys:
            if "&" in key:
                for c in key.split("&"):
                    if c in config["model"]["country_names"]["value"]:
                        config["init"][c] = deepcopy(config["init"][key])
        for key in keys:
            if "&" in key:
                del config["init"][key]

        # Handle run countries
        keys = list(config.keys())
        for key in keys:
            if "&" in key:
                for c in key.split("&"):
                    if c in config["model"]["country_names"]["value"]:
                        config[c] = deepcopy(config[key])
        for key in keys:
            if "&" in key:
                del config[key]

        return config

    @staticmethod
    def yaml_to_string(file_path: str | Path) -> str:
        with open(file_path, "r") as yaml_file:
            yaml_content = yaml_file.read()
        return yaml_content

    def set_random_seed(self, random_seed: int = 0) -> None:
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def set_exchange_rates(self) -> None:
        self.exchange_rates = ExchangeRates(
            exchange_rate_type=self.config["model"]["exchange_rates_type"]["value"],
            initial_year=int(self.config["model"]["year"]["value"]),
            country_names=self.config["model"]["country_names"]["value"],
            historic_exchange_rate_data=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    "exchange_rates",
                )
            ),
        )

    def initialise_agents(self) -> None:
        self.countries = self._init_countries()
        self.row = self._init_rest_of_the_world()
        self.goods_market = self._init_goods_market()

    @staticmethod
    def create_output_directory(output_path: str) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def save_config(self, h5_file: h5py.File) -> None:
        config_group = h5_file.create_group("config")
        config_group.attrs["config_data"] = self.config_string

    def save(self, i_run: Optional[int] = None) -> None:
        output_path = self.construct_output_path(i_run)
        self.create_output_directory(output_path)
        with h5py.File(str(output_path + "data.h5"), mode="w") as h5_file:
            self.save_random_seed(h5_file)
            self.save_config(h5_file)
            self.save_country_data(h5_file)
            self.row.ts.write_to_h5("rest_of_the_world", h5_file.create_group("ROW"))
            self.goods_market.ts.write_to_h5("GM", h5_file.create_group("GM"))
            self.exchange_rates.ts.write_to_h5("EXCH_RATES", h5_file.create_group("EXCH_RATES"))
            h5_file.close()

    def save_random_seed(self, h5_file: h5py.File) -> None:
        h5_file.attrs["random_seed"] = self.random_seed

    def construct_output_path(self, i_run: Optional[int]) -> str:
        # Open new file
        if i_run is None:
            output_path = str(self.output_path) + "/"
        else:
            output_path = str(self.output_path) + "_mc/run_" + str(i_run) + "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def save_country_data(self, h5_file: h5py.File) -> None:
        for country_name in self.config["model"]["country_names"]["value"]:
            country_group = h5_file.create_group(country_name)

            # Agent time series
            self.countries[country_name].individuals.ts.write_to_h5("individuals", country_group)
            self.countries[country_name].households.ts.write_to_h5("households", country_group)
            self.countries[country_name].firms.ts.write_to_h5("firms", country_group)
            self.countries[country_name].central_government.ts.write_to_h5("central_government", country_group)
            self.countries[country_name].government_entities.ts.write_to_h5("government_entities", country_group)
            self.countries[country_name].banks.ts.write_to_h5("banks", country_group)
            self.countries[country_name].central_bank.ts.write_to_h5("central_bank", country_group)
            self.countries[country_name].economy.ts.write_to_h5("economy", country_group)

            # Market time series
            self.countries[country_name].labour_market.ts.write_to_h5("labour_market", country_group)
            self.countries[country_name].credit_market.ts.write_to_h5("credit_market", country_group)
            self.countries[country_name].housing_market.ts.write_to_h5("housing_market", country_group)

            # Exogenous time series
            self.countries[country_name].exogenous.ts.write_to_h5("exogenous", country_group)

            # Firm industry
            industry_firms_df = pd.DataFrame(
                data=np.array(self.countries[country_name].firms.states["Industry"]),
                index=pd.Index(
                    range(self.countries[country_name].firms.ts.current("n_firms")),
                    name="Firm ID",
                ),
                columns=pd.Index(["Industry"], name="Field"),
            )
            country_group.create_dataset("industry_firms", data=industry_firms_df.to_numpy())
            country_group["industry_firms"].attrs["columns"] = industry_firms_df.columns.to_list()

            # Household consumption weights
            household_consumption_weights = pd.DataFrame(
                data=self.countries[country_name].households.parameters["consumption_weights_by_income_data"].T,
                index=pd.Index(["Q1", "Q2", "Q3", "Q4", "Q5"], name="Income Quantile"),
                columns=pd.Index(range(self.n_industries), name="Industry"),
            )
            country_group.create_dataset(
                "household_consumption_weights_by_income", data=household_consumption_weights.to_numpy()
            )
            country_group["household_consumption_weights_by_income"].attrs[
                "columns"
            ] = household_consumption_weights.columns.to_list()

            # Previous exogenous data
            country_group.create_dataset(
                "exogenous_historic_data", data=self.countries[country_name].exogenous.compiled_historic_data.to_numpy()
            )
            country_group["exogenous_historic_data"].attrs["index"] = list(
                dict.fromkeys(
                    self.countries[country_name]
                    .exogenous.compiled_historic_data.index.get_level_values(0)
                    .astype(str)
                    .to_list()
                )
            )
            country_group["exogenous_historic_data"].attrs["columns"] = (
                self.countries[country_name].exogenous.compiled_historic_data.columns.astype(str).to_list()
            )

    def _init_individuals(self, country_name: str) -> Individuals:
        return Individuals.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            scale=self.scale,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_individuals")),
            config=self.config[country_name]["individuals"],
        )

    def _init_households(
        self,
        country_name: str,
        initial_industry_consumption: np.ndarray,
        individual_ages: np.ndarray,
    ) -> Households:
        return Households.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            scale=self.scale,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_households")),
            initial_industry_consumption=initial_industry_consumption,
            individual_ages=individual_ages,
            corr_individuals=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_households_corr_individuals",
                )
            ),
            corr_renters=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_households_corr_renters",
                )
            ),
            corr_additionally_owned_properties=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_households_corr_additionally_owned_houses",
                )
            ),
            consumption_weights=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_household_consumption_weights",
                )
            ),
            consumption_weights_by_income=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_household_consumption_weights_by_income",
                )
            ),
            coefficient_fa_income=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_household_coefficient_fa_income",
                )
            ),
            value_added_tax=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_Taxes"))[
                "Value-added Tax"
            ].values[0],
            saving_rates_model=sio.load(self.processed_data_path.parent / "saving_rates_model.skops", trusted=True),
            social_transfers_model=sio.load(
                self.processed_data_path.parent / "social_transfers_model.skops", trusted=True
            ),
            wealth_distribution_model=sio.load(
                self.processed_data_path.parent / "wealth_distribution_model.skops", trusted=True
            ),
            config=self.config[country_name]["households"],
            init_config=self.config["init"][country_name]["households"],
        )

    def _init_firms(self, country_name: str) -> Firms:
        return Firms.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_firms")),
            corr_employees=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_firms_corr_employees",
                )
            ),
            intermediate_inputs_stock=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_firms_intermediate_inputs_stock",
                )
            ),
            used_intermediate_inputs=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_firms_used_intermediate_inputs",
                )
            ),
            capital_inputs_stock=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_firms_capital_inputs_stock",
                )
            ),
            used_capital_inputs=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_firms_used_capital_inputs",
                )
            ),
            intermediate_inputs_productivity_matrix=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_intermediate_inputs_productivity_matrix",
                )
            ),
            capital_inputs_productivity_matrix=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_capital_inputs_productivity_matrix",
                )
            ),
            capital_inputs_depreciation_matrix=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_capital_inputs_depreciation_matrix",
                )
            ),
            industry_vectors=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_industry_vectors",
                )
            ),
            goods_criticality_matrix=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    "goods_criticality_matrix",
                )
            ),
            calculate_hill_exponent=True,
            config=self.config[country_name]["firms"],
            init_config=self.config["init"][country_name]["firms"],
        )

    def _init_central_government(
        self,
        country_name: str,
        initial_individual_activity: np.ndarray,
    ) -> CentralGovernment:
        return CentralGovernment.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_central_gov")),
            tax_data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_Taxes")),
            taxes_net_subsidies=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_industry_vectors"))[
                "Taxes Less Subsidies Rates"
            ].values,
            number_of_unemployed_individuals=int(np.sum(initial_individual_activity == ActivityStatus.UNEMPLOYED)),
            unemployment_benefits_model=sio.load(
                self.processed_data_path.parent / "unemployment_benefits.skops", trusted=True
            ),
            other_benefits_model=sio.load(
                self.processed_data_path.parent / "other_social_benefits.skops", trusted=True
            ),
            config=self.config[country_name]["central_government"],
            init_config=self.config["init"][country_name]["central_government"],
        )

    def _init_gov_entities(self, country_name: str) -> GovernmentEntities:
        return GovernmentEntities.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_gov_entities")),
            number_of_entities=int(
                pd.DataFrame(
                    pd.read_hdf(
                        self.processed_data_path,
                        country_name + "_number_of_gov_entities",
                    )
                ).values[
                    0
                ][0]
            ),
            government_consumption_model=sio.load(
                self.processed_data_path.parent / "government_consumption_model.skops", trusted=True
            ),
            config=self.config[country_name]["government_entities"],
        )

    def _init_banks(self, country_name: str) -> Banks:
        return Banks.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            scale=self.scale,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_banks")),
            corr_firms=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_banks_corr_firms",
                )
            ),
            corr_households=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_synthetic_banks_corr_households",
                )
            ),
            policy_rate_markup=pd.DataFrame(
                pd.read_hdf(self.processed_data_path, country_name + "_policy_rate_markup")
            ).values[0][0],
            long_term_ir=pd.DataFrame(
                pd.read_hdf(self.processed_data_path, country_name + "_long_term_interest_rates")
            ).values[0][0],
            config=self.config[country_name]["banks"],
            init_config=self.config["init"][country_name]["banks"],
        )

    def _init_central_bank(self, country_name: str) -> CentralBank:
        return CentralBank.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_central_bank")),
            config=self.config[country_name]["central_bank"],
        )

    def _init_economy(
        self,
        country_name: str,
        initial_firm_prices: np.ndarray,
        initial_individual_activity: np.ndarray,
        initial_cpi_inflation: float,
        initial_ppi_inflation: float,
        initial_nominal_house_price_index_growth: float,
        initial_real_rent_paid: np.ndarray,
        initial_imp_rent_paid: np.ndarray,
        initial_rental_income: np.ndarray,
        initial_sectoral_growth: np.ndarray,
        initial_imports: np.ndarray,
        initial_imports_by_country: dict[str, np.ndarray],
        initial_exports: np.ndarray,
        initial_exports_by_country: dict[str, np.ndarray],
        export_taxes: float,
    ) -> Economy:
        return Economy.from_data(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            initial_firm_prices=initial_firm_prices,
            initial_individual_activity=initial_individual_activity,
            initial_cpi_inflation=initial_cpi_inflation,
            initial_ppi_inflation=initial_ppi_inflation,
            initial_nominal_house_price_index_growth=initial_nominal_house_price_index_growth,
            initial_real_rent_paid=initial_real_rent_paid,
            initial_imp_rent_paid=initial_imp_rent_paid,
            initial_rental_income=initial_rental_income,
            initial_sectoral_growth=initial_sectoral_growth,
            initial_imports=initial_imports,
            initial_imports_by_country=initial_imports_by_country,
            initial_exports=initial_exports,
            initial_exports_by_country=initial_exports_by_country,
            export_taxes=export_taxes,
            config=self.config[country_name]["economy"],
        )

    def _init_labour_market(
        self,
        country_name: str,
        initial_individual_activity: np.ndarray,
        initial_individual_employment_industry: np.ndarray,
    ) -> LabourMarket:
        return LabourMarket.from_data(
            country_name=country_name,
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            initial_individual_activity=initial_individual_activity,
            initial_individual_employment_industry=initial_individual_employment_industry,
            config=self.config[country_name]["labour_market"],
        )

    def _init_credit_market(self, country_name: str) -> CreditMarket:
        return CreditMarket.from_data(
            country_name=country_name,
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_credit_market")),
            config=self.config[country_name]["credit_market"],
        )

    def _init_housing_market(self, country_name: str) -> HousingMarket:
        return HousingMarket.from_data(
            country_name=country_name,
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            scale=self.scale,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, country_name + "_synthetic_housing_market")),
            config=self.config[country_name]["housing_market"],
        )

    def _init_exogenous(self, country_name: str) -> Exogenous:
        return Exogenous(
            country_name=country_name,
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            initial_year=self.config["model"]["year"]["value"],
            t_max=self.config["model"]["t_max"]["value"],
            log_inflation=pd.DataFrame(
                pd.read_hdf(self.processed_data_path, country_name + "_exogenous_log_inflation")
            ),
            sectoral_growth=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_exogenous_sectoral_growth",
                )
            ),
            unemployment_rate=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_exogenous_unemployment_rate",
                )
            ),
            vacancy_rate=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_exogenous_vacancy_rate",
                )
            ),
            house_price_index=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_exogenous_house_price_index",
                )
            ),
            total_firm_deposits_and_debt=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_exogenous_total_firm_deposits_and_debt",
                )
            ),
            iot_industry_data=pd.DataFrame(
                pd.read_hdf(
                    self.processed_data_path,
                    country_name + "_exogenous_iot_industry_data",
                )
            ),
            exchange_rates_data=self.exchange_rates.historic_exchange_rate_data.loc[[country_name]],
        )

    def _init_countries(self) -> dict[str, Country]:
        countries = {}
        for country_name in self.config["model"]["country_names"]["value"]:
            all_countries = self.config["model"]["country_names"]["value"] + ["ROW"]
            all_countries.remove(country_name)
            exogenous = self._init_exogenous(country_name)
            individuals = self._init_individuals(country_name)
            households = self._init_households(
                country_name=country_name,
                initial_industry_consumption=exogenous.ts.initial("sectoral_household_consumption"),
                individual_ages=individuals.states["Age"],
            )
            firms = self._init_firms(country_name)
            central_gov = self._init_central_government(
                country_name=country_name,
                initial_individual_activity=individuals.states["Activity Status"],
            )
            gov_entities = self._init_gov_entities(country_name)
            banks = self._init_banks(country_name=country_name)
            central_bank = self._init_central_bank(country_name)
            economy = self._init_economy(
                country_name=country_name,
                initial_firm_prices=firms.ts.current("price"),
                initial_individual_activity=individuals.states["Activity Status"],
                initial_cpi_inflation=exogenous.ts.initial("cpi_inflation")[0],
                initial_ppi_inflation=exogenous.ts.initial("ppi_inflation")[0],
                initial_nominal_house_price_index_growth=exogenous.ts.initial("nominal_house_price_index_growth")[0],
                initial_real_rent_paid=households.ts.current("rent"),
                initial_imp_rent_paid=households.ts.current("rent_imputed"),
                initial_rental_income=households.ts.current("income_rental"),
                initial_sectoral_growth=exogenous.ts.initial("sectoral_growth"),
                initial_imports=exogenous.ts.initial("sectoral_imports"),
                initial_imports_by_country={
                    c: exogenous.ts.initial("sectoral_imports_from_" + c) for c in all_countries
                },
                initial_exports=exogenous.ts.initial("sectoral_exports"),
                initial_exports_by_country={c: exogenous.ts.initial("sectoral_exports_to_" + c) for c in all_countries},
                export_taxes=central_gov.states["Export Tax"],
            )
            labour_market = self._init_labour_market(
                country_name=country_name,
                initial_individual_activity=individuals.states["Activity Status"],
                initial_individual_employment_industry=individuals.states["Employment Industry"],
            )
            credit_market = self._init_credit_market(country_name)
            housing_market = self._init_housing_market(country_name)
            country = Country(
                country_name=country_name,
                year=self.config["model"]["year"]["value"],
                t_max=self.t_max,
                scale=self.scale,
                individuals=individuals,
                households=households,
                firms=firms,
                central_government=central_gov,
                government_entities=gov_entities,
                banks=banks,
                central_bank=central_bank,
                economy=economy,
                labour_market=labour_market,
                credit_market=credit_market,
                housing_market=housing_market,
                exogenous=exogenous,
            )
            countries[country_name] = country
        return countries

    def _init_goods_market(self) -> GoodsMarket:
        # Collect agents
        goods_market_participants = {}
        for country_name in self.countries.keys():
            goods_market_participants[country_name] = [
                self.countries[country_name].firms,
                self.countries[country_name].households,
                self.countries[country_name].government_entities,
            ]
        goods_market_participants["ROW"] = [self.row]

        # Create the goods market
        goods_market = GoodsMarket.from_data(
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            trade_proportions=pd.DataFrame(pd.read_hdf(self.processed_data_path, "trade_proportions")),
            config=self.config["goods_market"]["goods_market"],
        )
        goods_market.functions["clearing"].initiate_agents(
            n_industries=self.n_industries,
            goods_market_participants=goods_market_participants,
        )
        goods_market.functions["clearing"].initiate_the_supply_chain(
            initial_supply_chain=None,
        )
        return goods_market

    def _init_rest_of_the_world(self) -> RestOfTheWorld:
        return RestOfTheWorld.from_data(
            country_name="ROW",
            all_country_names=self.config["model"]["country_names"]["value"] + ["ROW"],
            year=self.config["model"]["year"]["value"],
            t_max=self.t_max,
            n_industries=self.n_industries,
            data=pd.DataFrame(pd.read_hdf(self.processed_data_path, "synthetic_rest_of_the_world")),
            row_exports_model=sio.load(self.processed_data_path.parent / "row_exports_model.skops", trusted=True),
            row_imports_model=sio.load(self.processed_data_path.parent / "row_imports_model.skops", trusted=True),
            average_country_ppi_inflation=float(
                np.mean([self.countries[c].economy.ts.current("ppi_inflation")[0] for c in self.countries.keys()])
            ),
            config=self.config["ROW"]["ROW"],
        )
