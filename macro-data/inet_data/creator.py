import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from scipy.optimize import fsolve

from inet_data.processing.synthetic_banks.default_synthetic_banks import (
    SyntheticDefaultBanks,
)
from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_central_bank.default_synthetic_central_bank import (
    SyntheticDefaultCentralBanks,
)
from inet_data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)
from inet_data.processing.synthetic_central_government.default_synthetic_central_government import (
    SyntheticDefaultCentralGovernment,
)
from inet_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)
from inet_data.processing.synthetic_credit_market.default_synthetic_credit_market import (
    DefaultSyntheticCreditMarket,
)
from inet_data.processing.synthetic_credit_market.synthetic_credit_market import (
    SyntheticCreditMarket,
)
from inet_data.processing.synthetic_firms.default_synthetic_firms import (
    SyntheticDefaultFirms,
)
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    SyntheticDefaultGovernmentEntities,
)
from inet_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)
from inet_data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)
from inet_data.processing.synthetic_housing_market.synthetic_housing_market import (
    SyntheticHousingMarket,
)
from inet_data.processing.synthetic_matching.matching_firms_with_banks import (
    match_firms_with_banks,
)
from inet_data.processing.synthetic_matching.matching_households_with_banks import (
    match_households_with_banks,
)
from inet_data.processing.synthetic_matching.matching_households_with_houses import (
    match_households_with_houses,
)
from inet_data.processing.synthetic_matching.matching_individuals_with_firms import (
    match_individuals_with_firms,
)
from inet_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
)
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from inet_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)
from inet_data.readers.handle_readers import get_reader_names, init_readers
from inet_data.readers.util.matching_iot_with_sea import (
    compile_exogenous_industry_data,
    compile_industry_data,
)
from inet_data.util.country_code_map import get_map_long_to_short
from inet_data.util.download import download_data
from inet_data.util.partition import partition_into_quintiles
from inet_data.util.regressions import fit_linear


class Creator:
    def __init__(
        self,
        config_path: Path,
        raw_data_path: Path,
        processed_data_path: Path,
        force_download: bool = False,
        random_seed: int = 0,
        create_exogenous_industry_data: bool = False,
        testing: bool = False,
    ):
        # Parameters
        self.config = self.process_config(config_path)
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

        # Basic model settings
        self.country_names = self.config["model"]["country_names"]["value"]
        self.year = self.config["model"]["year"]["value"]
        self.start_date = self.config["model"]["start_date"]["value"]
        self.scale = self.config["model"]["scale"]["value"]
        self.industries = self.config["model"]["industries"]["value"]

        # Set the random seed
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        # Create directories if they don't already exist
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
        if not os.path.exists(self.processed_data_path.parent):
            os.makedirs(self.processed_data_path.parent)

        # Check if we need to download inet_data
        reader_names = get_reader_names()
        for reader_name in reader_names:
            if not os.path.exists(self.raw_data_path / reader_name):
                force_download = True
                break
        if force_download:
            download_data(raw_data_path=self.raw_data_path)

        # Map to short country names
        self.country_names_short = list(map(get_map_long_to_short(self.raw_data_path).get, self.country_names))

        # Create readers
        self.data_readers = init_readers(
            raw_data_path=self.raw_data_path,
            country_names=self.country_names,
            country_names_short=self.country_names_short,
            year=self.year,
            scale=self.scale,
            industries=self.industries,
            start_date=self.start_date,
            create_exogenous_industry_data=create_exogenous_industry_data,
            testing=testing,
        )

        # Take industry inet_data
        self.industry_data = compile_industry_data(
            current_icio_reader=self.data_readers["icio"][self.year],
            sea_reader=self.data_readers["wiod_sea"],
            econ_reader=self.data_readers["oecd_econ"],
            exchange_rates=self.data_readers["exchange_rates"],
            country_names=self.country_names,
            config=self.config,
        )

        # Take exogenous industry inet_data
        if create_exogenous_industry_data:
            exogenous_industry_data = compile_exogenous_industry_data(
                icio_readers=self.data_readers["icio"],
                exchange_rates=self.data_readers["exchange_rates"],
                country_names=self.country_names,
            )
        else:
            exogenous_industry_data = {}

        # Take exogenous inet_data
        self.exogenous_data = {}
        for country_name in exogenous_industry_data.keys():
            if country_name in self.country_names:
                self.exogenous_data[country_name] = {
                    "log_inflation": self.data_readers["world_bank"].get_log_inflation(country_name),
                    "sectoral_growth": self.data_readers["eurostat"].get_perc_sectoral_growth(country_name),
                    "unemployment_rate": self.data_readers["oecd_econ"].get_unemployment_rate(country_name),
                    "house_price_index": self.data_readers["oecd_econ"].get_house_price_index(country_name),
                    "vacancy_rate": self.data_readers["oecd_econ"].get_vacancy_rate(country_name),
                    "total_firm_deposits_and_debt": self.data_readers["eurostat"].get_total_industry_debt_and_deposits(
                        country_name
                    ),
                }
            else:
                self.exogenous_data[country_name] = {}
            self.exogenous_data[country_name]["iot_industry_data"] = exogenous_industry_data[country_name]
        for country_name in self.country_names:
            if country_name not in self.exogenous_data.keys():
                self.exogenous_data[country_name] = None

        # Agents
        self.synthetic_population: dict[str, SyntheticPopulation] = {}
        self.synthetic_firms: dict[str, SyntheticFirms] = {}
        self.synthetic_banks: dict[str, SyntheticBanks] = {}
        self.synthetic_central_gov: dict[str, SyntheticCentralGovernment] = {}
        self.synthetic_gov_entities: dict[str, SyntheticGovernmentEntities] = {}
        self.synthetic_central_banks: dict[str, SyntheticCentralBank] = {}
        self.synthetic_credit_market: dict[str, SyntheticCreditMarket] = {}
        self.synthetic_housing_market: dict[str, SyntheticHousingMarket] = {}
        self.synthetic_rest_of_the_world = None

    @staticmethod
    def process_config(config_path: Path) -> dict[str, Any]:
        config = yaml.safe_load(open(config_path, "r"))
        config = {
            "model": config["model"].copy(),
            "init": config["init"].copy(),
        }

        # Handle countries
        keys = list(config["init"].keys())
        for key in keys:
            if "&" in key:
                for c in key.split("&"):
                    if c in config["model"]["country_names"]["value"]:
                        config["init"][c] = deepcopy(config["init"][key])
        for key in keys:
            if "&" in key:
                del config["init"][key]

        return config

    def create(self, save_output: bool = True) -> None:
        # Create agents
        self.create_synthetic_central_gov()
        self.create_synthetic_gov_entities()
        self.create_synthetic_central_bank()
        self.create_synthetic_population()
        self.create_synthetic_firms()
        self.create_synthetic_banks()
        self.create_synthetic_rest_of_the_world()
        self.create_synthetic_housing_market()

        # Match agents
        self.match_firms_with_individuals()
        self.match_firms_with_banks()
        self.match_households_with_banks()
        self.match_households_with_houses()

        # Set other initial conditions related to housing
        self.set_housing_initial_conditions()

        # Compute household wealth and income
        self.compute_household_wealth_and_income()

        # Compute household saving rates
        self.set_household_saving_rates()

        # Match IOTs
        self.normalise_household_consumption()
        self.match_consumption_weights_by_income()

        # Compute some stuff after matching
        self.compute_bank_initial_conditions()
        self.set_initial_loans()
        self.compute_firm_profits()
        self.update_central_gov_fields()

        # Save the output
        if save_output:
            self.save()

    def create_synthetic_banks(self) -> None:
        for country_name in self.country_names:
            self.synthetic_banks[country_name] = SyntheticDefaultBanks(
                country_name=country_name,
                year=self.year,
                number_of_banks=(
                    1
                    if self.config["model"]["single_bank"]["value"]
                    else max(
                        1,
                        int(
                            self.data_readers["oecd_econ"].read_number_of_bank_branches(country_name, self.year)
                            / self.scale
                        ),
                    )
                ),
            )
            self.synthetic_banks[country_name].create(
                bank_equity=self.data_readers["eurostat"].get_total_bank_equity(country_name, self.year)
            )

    def create_synthetic_central_gov(self, year_range: int = 10) -> None:
        if self.start_date is not None:
            # If start date is not None, we need to make sure that we don't go back too far
            year_range = min(year_range, self.year - self.start_date)
        for country_name in self.country_names:
            # Initiate the agent
            self.synthetic_central_gov[country_name] = SyntheticDefaultCentralGovernment(
                country_name=country_name,
                year=self.year,
            )

            # Adding inet_data
            unemp = [
                self.data_readers["oecd_econ"].unemployment_benefits_gdp_pct(country_name, year)
                * self.data_readers["world_bank"].get_current_monthly_gdp(country_name, year)
                for year in range(self.year - year_range, self.year)
            ]
            other = [
                self.data_readers["oecd_econ"].all_benefits_gdp_pct(country_name, year)
                * self.data_readers["world_bank"].get_current_monthly_gdp(country_name, year)
                - self.data_readers["oecd_econ"].unemployment_benefits_gdp_pct(country_name, year)
                * self.data_readers["world_bank"].get_current_monthly_gdp(country_name, year)
                for year in range(self.year - year_range, self.year)
            ]
            benefits_data = pd.DataFrame(
                data={"Unemployment Benefits": unemp, "Other Total Benefits": other},
                index=pd.DatetimeIndex(
                    pd.date_range(
                        start=str(self.year - year_range) + "-01-01",
                        end=str(self.year) + "-01-01",
                        freq="Y",
                    )
                ),
            )
            self.synthetic_central_gov[country_name].create(
                central_gov_debt=self.data_readers["oecd_econ"].general_gov_debt(country_name, self.year),
                benefits_data=benefits_data,
                exogenous_data=self.exogenous_data[country_name],
            )

    def create_synthetic_gov_entities(self) -> None:
        for country_name in self.country_names:
            # Initiate the agent
            self.synthetic_gov_entities[country_name] = SyntheticDefaultGovernmentEntities(
                country_name=country_name,
                year=self.year,
            )

            # Create it
            if self.exogenous_data[country_name] is None:
                total_gov_consumption_growth = None
            else:
                total_gov_consumption = (
                    self.exogenous_data[country_name]["iot_industry_data"]
                    .xs("Government Consumption in USD", axis=1, level=0)
                    .sum(axis=1)
                )
                total_gov_consumption = total_gov_consumption.loc[
                    total_gov_consumption.index < pd.Timestamp(self.year, 1, 1)
                ]
                total_gov_consumption_growth = (total_gov_consumption / total_gov_consumption.shift(1)).values
            self.synthetic_gov_entities[country_name].create(
                single_government_entity=self.config["model"]["single_government_entity"]["value"],
                monthly_govt_consumption_in_usd=self.industry_data[country_name]["industry_vectors"][
                    "Government Consumption in USD"
                ].values,
                monthly_govt_consumption_in_lcu=self.industry_data[country_name]["industry_vectors"][
                    "Government Consumption in LCU"
                ].values,
                total_monthly_value_added_in_lcu=self.industry_data[country_name]["industry_vectors"][
                    "Value Added in LCU"
                ].sum(),
                total_number_of_firms=int(
                    self.data_readers["oecd_econ"]
                    .read_business_demography(
                        country=country_name,
                        output=pd.Series(self.industry_data[country_name]["industry_vectors"]["Output"].values),
                        year=self.year,
                    )
                    .sum()
                ),
                total_gov_consumption_growth=total_gov_consumption_growth,
            )

    def create_synthetic_central_bank(self) -> None:
        for country_name in self.country_names:
            self.synthetic_central_banks[country_name] = SyntheticDefaultCentralBanks(
                country_name=country_name,
                year=self.year,
            )
            self.synthetic_central_banks[country_name].create(
                initial_policy_rate=self.data_readers["policy_rates"].cb_policy_rate(country_name, self.year)
            )

    def create_synthetic_population(self) -> None:
        for ind, country_name in enumerate(self.country_names):
            # Create the population
            self.synthetic_population[country_name] = SyntheticHFCSPopulation(
                country_name=country_name,
                country_name_short=self.country_names_short[ind],
                scale=self.scale,
                year=self.year,
                industries=self.industries,
            )
            self.synthetic_population[country_name].create(
                hfcs_reader=self.data_readers["hfcs"][country_name],
                econ_reader=self.data_readers["oecd_econ"],
                wb_reader=self.data_readers["world_bank"],
                n_households=int(
                    self.data_readers["eurostat"].number_of_households(country_name, self.year) / self.scale
                ),
                number_of_firms_by_industry=self.industry_data[country_name]["industry_vectors"][
                    "Number of Firms"
                ].values,
                total_unemployment_benefits=self.synthetic_central_gov[country_name]
                .central_gov_data["Total Unemployment Benefits"]
                .values[0],
                rent_as_fraction_of_unemployment_rate=self.config["init"][country_name]["central_government"][
                    "functions"
                ]["social_housing"]["parameters"]["rent_as_fraction_of_unemployment_rate"]["value"],
            )
            self.synthetic_population[country_name].set_consumption_weights(
                consumption_weights=self.industry_data[country_name]["industry_vectors"][
                    "Household Consumption Weights"
                ].values
            )

    def create_synthetic_firms(self) -> None:
        for country_name in self.country_names:
            self.synthetic_firms[country_name] = SyntheticDefaultFirms(
                country_name=country_name,
                scale=self.scale,
                year=self.year,
                industries=self.industries,
            )
            self.synthetic_firms[country_name].set_industries(
                number_of_firms_by_industry=self.industry_data[country_name]["industry_vectors"][
                    "Number of Firms"
                ].values,
            )
            self.synthetic_firms[country_name].create(
                total_firm_deposits=self.data_readers["eurostat"].get_total_nonfin_firm_deposits(
                    country_name, self.year
                ),
                total_firm_debt=self.data_readers["eurostat"].get_total_nonfin_firm_debt(country_name, self.year),
                econ_reader=self.data_readers["oecd_econ"],
                ons_reader=self.data_readers["ons"],
                exchange_rates=self.data_readers["exchange_rates"],
                industry_data=self.industry_data[country_name],
                number_of_employees_by_industry=self.synthetic_population[country_name].number_employees_by_industry,
                initial_inventory_to_production_fraction=self.config["init"][country_name]["firms"]["functions"][
                    "target_production"
                ]["parameters"]["target_inventory_to_production_fraction"]["value"],
                intermediate_inputs_utilisation_rate=self.config["init"][country_name]["firms"]["parameters"][
                    "intermediate_inputs_utilisation_rate"
                ]["value"],
                capital_inputs_utilisation_rate=self.config["init"][country_name]["firms"]["parameters"][
                    "capital_inputs_utilisation_rate"
                ]["value"],
                assume_zero_initial_deposits=self.config["init"][country_name]["firms"]["parameters"][
                    "assume_zero_initial_deposits"
                ]["value"],
                assume_zero_initial_debt=self.config["init"][country_name]["firms"]["parameters"][
                    "assume_zero_initial_debt"
                ]["value"],
            )

    def create_synthetic_housing_market(self) -> None:
        for country_name in self.country_names:
            self.synthetic_housing_market[country_name] = DefaultSyntheticHousingMarket(
                country_name=country_name,
                year=self.year,
            )

    def create_synthetic_rest_of_the_world(self) -> None:
        # Initiate agent
        self.synthetic_rest_of_the_world = DefaultSyntheticRestOfTheWorld(self.year)

        # Create
        if "ROW" in self.exogenous_data.keys():
            row_exports_data = (
                self.exogenous_data["ROW"]["iot_industry_data"].xs("Exports in USD", axis=1, level=0).sum(axis=1)
            )
            row_exports_data = row_exports_data.loc[row_exports_data.index < pd.Timestamp(self.year, 1, 1)]
            row_exports_data_growth = (row_exports_data / row_exports_data.shift(1)).values
            row_imports_data = (
                self.exogenous_data["ROW"]["iot_industry_data"].xs("Imports in USD", axis=1, level=0).sum(axis=1)
            )
            row_imports_data = row_imports_data.loc[row_imports_data.index < pd.Timestamp(self.year, 1, 1)]
            row_imports_data_growth = (row_imports_data / row_imports_data.shift(1)).values
        else:
            row_exports_data_growth = None
            row_imports_data_growth = None
        self.synthetic_rest_of_the_world.create(
            row_exports=self.industry_data["ROW"]["industry_vectors"]["Exports in USD"],
            row_imports=self.industry_data["ROW"]["industry_vectors"]["Imports in USD"],
            exchange_rate_usd_to_lcu=self.data_readers["exchange_rates"].from_usd_to_lcu("ROW", self.year),
            row_exports_data_growth=row_exports_data_growth,
            row_imports_data_growth=row_imports_data_growth,
        )

    def set_household_saving_rates(self) -> None:
        for country_name in self.country_names:
            self.synthetic_population[country_name].set_household_saving_rates(
                independents=self.config["init"][country_name]["households"]["functions"]["saving_rates"]["parameters"][
                    "independents"
                ]["value"],
            )

    def normalise_household_consumption(self, positive_saving_rates_only: bool = True) -> None:
        def default_desired_consumption(
            income_: np.ndarray,
            consumption_weights_: np.ndarray,
            saving_rates_: np.ndarray,
            tau_vat_: float,
        ) -> np.ndarray:
            return 1.0 / (1 + tau_vat_) * np.outer(consumption_weights_, (1 - saving_rates_) * income_).T

        for country_name in self.country_names:
            iot_hh_consumption = self.industry_data[country_name]["industry_vectors"]["Household Consumption in LCU"]
            vat = self.data_readers["world_bank"].get_tau_vat(country_name, self.year)
            cons_weights = self.synthetic_population[country_name].consumption_weights
            income = self.synthetic_population[country_name].household_data["Income"].values
            sr = self.synthetic_population[country_name].household_data["Saving Rate"].values
            current_hh_consumption = default_desired_consumption(
                income_=income,
                consumption_weights_=cons_weights,
                saving_rates_=sr,
                tau_vat_=vat,
            )

            # Adjust saving rates
            factor = iot_hh_consumption.sum() / current_hh_consumption.sum()
            self.synthetic_population[country_name].household_data["Saving Rate"] = (
                1 - (1 - self.synthetic_population[country_name].household_data["Saving Rate"]) * factor
            )
            if positive_saving_rates_only:
                sr = self.synthetic_population[country_name].household_data["Saving Rate"].values
                sr[sr < 0] = 0.0
                current_hh_consumption = default_desired_consumption(
                    income_=income,
                    consumption_weights_=cons_weights,
                    saving_rates_=sr,
                    tau_vat_=vat,
                )
                diff = iot_hh_consumption.sum() - current_hh_consumption.sum()
                inc_sr = (1.0 / (1 + vat) * np.outer(cons_weights, sr * income).T).sum()
                factor = 1.0 - diff / inc_sr
                self.synthetic_population[country_name].household_data["Saving Rate"] = factor * sr

            # Overwrite the model
            (
                saving_rates,
                self.synthetic_population[country_name].saving_rates_model,
            ) = fit_linear(
                household_data=self.synthetic_population[country_name].household_data,
                independents=self.config["init"][country_name]["households"]["functions"]["saving_rates"]["parameters"][
                    "independents"
                ]["value"],
                dependent="Saving Rate",
            )
            self.synthetic_population[country_name].household_data["Saving Rate"] = saving_rates

    def match_consumption_weights_by_income(self, consumption_variance: float = 0.1) -> None:
        def default_desired_consumption(
            income_: np.ndarray,
            consumption_weights_: np.ndarray,
            saving_rates_: np.ndarray,
            tau_vat_: float,
        ) -> np.ndarray:
            return 1.0 / (1 + tau_vat_) * np.outer(consumption_weights_, (1 - saving_rates_) * income_).T

        for country_name in self.country_names:
            weights_by_income = (
                self.data_readers["oecd_econ"]
                .get_household_consumption_by_income_quantile(
                    country=country_name,
                    year=self.year,
                )
                .values
            )
            iot_hh_consumption = self.industry_data[country_name]["industry_vectors"]["Household Consumption in LCU"]
            iot_hh_consumption_norm = (iot_hh_consumption / iot_hh_consumption.sum()).values
            quintiles = partition_into_quintiles(data=self.synthetic_population[country_name].household_data["Income"])
            vat = self.data_readers["world_bank"].get_tau_vat(country_name, self.year)
            income = self.synthetic_population[country_name].household_data["Income"].values
            sr = self.synthetic_population[country_name].household_data["Saving Rate"].values

            # Set up new consumption weights
            def func(consumption_scalars):
                i_cons_weights = [[[] for _ in range(18)] for _ in range(5)]
                for _g in range(18):
                    i_cons_weights[0][_g] = iot_hh_consumption_norm[_g] + consumption_scalars[_g] * (
                        weights_by_income[_g, 0] - consumption_scalars[_g + 18]
                    )
                    i_cons_weights[1][_g] = iot_hh_consumption_norm[_g] + consumption_scalars[_g] * (
                        weights_by_income[_g, 1] - consumption_scalars[_g + 18]
                    )
                    i_cons_weights[2][_g] = iot_hh_consumption_norm[_g] + consumption_scalars[_g] * (
                        weights_by_income[_g, 2] - consumption_scalars[_g + 18]
                    )
                    i_cons_weights[3][_g] = iot_hh_consumption_norm[_g] + consumption_scalars[_g] * (
                        weights_by_income[_g, 3] - consumption_scalars[_g + 18]
                    )
                    i_cons_weights[4][_g] = iot_hh_consumption_norm[_g] + consumption_scalars[_g] * (
                        weights_by_income[_g, 4] - consumption_scalars[_g + 18]
                    )
                i_cons_weights = np.array(i_cons_weights)

                # Resulting total consumption
                total_consumption = np.zeros(18, dtype=float)
                for q in range(5):
                    ind = np.where(quintiles == q)[0]
                    total_consumption += (
                        default_desired_consumption(
                            income_=income[ind],
                            consumption_weights_=i_cons_weights[q],
                            saving_rates_=sr[ind],
                            tau_vat_=vat,
                        )
                        .astype(float)
                        .sum(axis=0)
                    )

                # Compile equations
                eqs = []
                for _g in range(18):
                    eqs.append(total_consumption[_g] - iot_hh_consumption[_g])
                eqs.append(np.sum(np.absolute(consumption_scalars[0:18])) - consumption_variance)
                eqs.append(np.sum(i_cons_weights[0]) - 1.0)
                eqs.append(np.sum(i_cons_weights[1]) - 1.0)
                eqs.append(np.sum(i_cons_weights[2]) - 1.0)
                eqs.append(np.sum(i_cons_weights[3]) - 1.0)
                eqs.append(np.sum(i_cons_weights[4]) - 1.0)

                for _ in range(12):
                    eqs.append(1 - 1)

                return eqs

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fsolve(func, np.ones(36))[0:36]
            self.synthetic_population[country_name].consumption_weights_by_income = np.zeros((5, 18))
            for g_ in range(18):
                self.synthetic_population[country_name].consumption_weights_by_income[0][g_] = iot_hh_consumption_norm[
                    g_
                ] + res[g_] * (weights_by_income[g_, 0] - res[g_ + 18])
                self.synthetic_population[country_name].consumption_weights_by_income[1][g_] = iot_hh_consumption_norm[
                    g_
                ] + res[g_] * (weights_by_income[g_, 1] - res[g_ + 18])
                self.synthetic_population[country_name].consumption_weights_by_income[2][g_] = iot_hh_consumption_norm[
                    g_
                ] + res[g_] * (weights_by_income[g_, 2] - res[g_ + 18])
                self.synthetic_population[country_name].consumption_weights_by_income[3][g_] = iot_hh_consumption_norm[
                    g_
                ] + res[g_] * (weights_by_income[g_, 3] - res[g_ + 18])
                self.synthetic_population[country_name].consumption_weights_by_income[4][g_] = iot_hh_consumption_norm[
                    g_
                ] + res[g_] * (weights_by_income[g_, 4] - res[g_ + 18])

    def match_firms_with_individuals(self) -> None:
        for country_name in self.country_names:
            # Match them
            match_individuals_with_firms(
                synthetic_population=self.synthetic_population[country_name],
                synthetic_firms=self.synthetic_firms[country_name],
                employee_social_contribution_taxes=self.data_readers["oecd_econ"].read_tau_siw(country_name, self.year),
                income_taxes=self.data_readers["oecd_econ"].read_tau_income(country_name, self.year),
            )

            # Set individual labour inputs
            self.synthetic_population[country_name].set_individual_labour_inputs(
                firm_production=self.synthetic_firms[country_name].firm_data["Production"],
                firm_employees=self.synthetic_firms[country_name].firm_data["Employees ID"],
            )

    def match_firms_with_banks(self) -> None:
        for country_name in self.country_names:
            match_firms_with_banks(
                synthetic_firms=self.synthetic_firms[country_name],
                synthetic_banks=self.synthetic_banks[country_name],
            )

    def match_households_with_banks(self) -> None:
        for country_name in self.country_names:
            match_households_with_banks(
                synthetic_population=self.synthetic_population[country_name],
                synthetic_banks=self.synthetic_banks[country_name],
            )

    def match_households_with_houses(self) -> None:
        for country_name in self.country_names:
            match_households_with_houses(
                synthetic_population=self.synthetic_population[country_name],
                synthetic_housing_market=self.synthetic_housing_market[country_name],
                rental_income_taxes=self.data_readers["oecd_econ"].read_tau_income(country_name, self.year),
                social_housing_rent=self.synthetic_population[country_name].social_housing_rent,
                total_imputed_rent=self.data_readers["icio"][self.year].imputed_rents[country_name],
            )

    def set_housing_initial_conditions(self) -> None:
        for country_name in self.country_names:
            self.synthetic_housing_market[country_name].set_initial_conditions()

    def compute_household_wealth_and_income(self) -> None:
        for country_name in self.country_names:
            self.synthetic_population[country_name].compute_household_wealth(
                wealth_distribution_independents=self.config["init"][country_name]["households"]["functions"]["wealth"][
                    "parameters"
                ]["independents"]["value"],
            )
            self.synthetic_population[country_name].compute_household_income(
                central_gov_config=self.config["init"][country_name]["central_government"],
                total_social_transfers=self.synthetic_central_gov[country_name]
                .central_gov_data["Other Social Benefits"]
                .values[0],
            )

    def compute_bank_initial_conditions(self) -> None:
        for country_name in self.country_names:
            self.synthetic_banks[country_name].set_initial_bank_fields(
                firm_deposits=self.synthetic_firms[country_name].firm_data["Deposits"].values,
                firm_debt=self.synthetic_firms[country_name].firm_data["Debt"].values,
                household_deposits=self.synthetic_population[country_name].household_data["Wealth in Deposits"].values,
                household_mortgage_debt=self.synthetic_population[country_name]
                .household_data["Outstanding Balance of HMR Mortgages"]
                .values
                + self.synthetic_population[country_name]
                .household_data["Outstanding Balance of Mortgages on other Properties"]
                .values,
                household_other_debt=self.synthetic_population[country_name]
                .household_data["Outstanding Balance of other Non-Mortgage Loans"]
                .values,
                cb_policy_rate=self.data_readers["policy_rates"].cb_policy_rate(
                    country=country_name,
                    year=self.year,
                ),
                bank_markup_interest_rate_short_term_firm_loans=self.data_readers["eurostat"].firm_risk_premium(
                    country_name, self.year
                ),
                bank_markup_interest_rate_long_term_firm_loans=self.data_readers["eurostat"].firm_risk_premium(
                    country_name, self.year
                ),
                bank_markup_interest_rate_household_payday_loans=self.data_readers["eurostat"].firm_risk_premium(
                    country_name, self.year
                ),
                bank_markup_interest_rate_household_consumption_loans=self.config["init"][country_name]["banks"][
                    "parameters"
                ]["initial_markup_interest_rate_household_consumption_loans"]["value"],
                bank_markup_interest_rate_mortgages=self.config["init"][country_name]["banks"]["parameters"][
                    "initial_markup_mortgage_interest_rate"
                ]["value"],
                bank_markup_interest_rate_overdraft_firm=self.data_readers["eurostat"].firm_risk_premium(
                    country_name, self.year
                ),
                bank_markup_interest_rate_overdraft_household=self.config["init"][country_name]["banks"]["parameters"][
                    "initial_markup_interest_rate_overdraft_households"
                ]["value"],
            )

    def set_initial_loans(self) -> None:
        for country_name in self.country_names:
            self.synthetic_credit_market[country_name] = DefaultSyntheticCreditMarket(
                country_name=country_name,
                year=self.year,
            )
            self.synthetic_credit_market[country_name].create(
                bank_data=self.synthetic_banks[country_name].bank_data,
                initial_firm_debt=self.synthetic_firms[country_name].firm_data["Debt"].values,
                initial_household_other_debt=self.synthetic_population[country_name]
                .household_data["Outstanding Balance of other Non-Mortgage Loans"]
                .values,
                initial_household_mortgage_debt=self.synthetic_population[country_name]
                .household_data["Outstanding Balance of HMR Mortgages"]
                .values
                + self.synthetic_population[country_name]
                .household_data["Outstanding Balance of Mortgages on other Properties"]
                .values,
                firms_corresponding_bank=self.synthetic_firms[country_name].firm_data["Corresponding Bank ID"],
                households_corresponding_bank=self.synthetic_population[country_name].household_data[
                    "Corresponding Bank ID"
                ],
                initial_firm_loan_maturity=self.config["init"][country_name]["banks"]["parameters"][
                    "long_term_firm_loan_maturity"
                ]["value"],
                household_consumption_loan_maturity=self.config["init"][country_name]["banks"]["parameters"][
                    "household_consumption_expansion_loan_maturity"
                ]["value"],
                mortgage_maturity=self.config["init"][country_name]["banks"]["parameters"]["mortgage_maturity"][
                    "value"
                ],
                assume_zero_initial_firm_debt=self.config["init"][country_name]["firms"]["parameters"][
                    "assume_zero_initial_debt"
                ]["value"],
            )
            self.synthetic_population[country_name].set_debt_installments(
                credit_market_data=self.synthetic_credit_market[country_name].credit_market_data
            )

    def compute_firm_profits(self) -> None:
        for country_name in self.country_names:
            self.synthetic_firms[country_name].set_additional_initial_conditions(
                econ_reader=self.data_readers["oecd_econ"],
                industry_data=self.industry_data[country_name],
                interest_rate_on_firm_deposits=self.synthetic_banks[country_name]
                .bank_data["Interest Rates on Firm Deposits"]
                .values,
                overdraft_rate_on_firm_deposits=self.synthetic_banks[country_name]
                .bank_data["Overdraft Rate on Firm Deposits"]
                .values,
                credit_market_data=self.synthetic_credit_market[country_name].credit_market_data,
            )

    def update_central_gov_fields(self) -> None:
        for country_name in self.country_names:
            self.synthetic_central_gov[country_name].update_fields(
                total_social_housing_rent=self.synthetic_population[country_name].social_housing_rent
                * np.sum(
                    self.synthetic_population[country_name].household_data["Tenure Status of the Main Residence"] == -1
                ),
                firm_taxes_and_subsidies=float(
                    np.sum(self.synthetic_firms[country_name].firm_data["Taxes paid on Production"])
                ),
                firm_corporate_taxes=float(
                    np.sum(self.synthetic_firms[country_name].firm_data["Corporate Taxes Paid"])
                ),
                firm_employer_si_tax=self.data_readers["oecd_econ"].read_tau_sif(country_name, self.year)
                * np.sum(self.synthetic_population[country_name].individual_data["Employee Income"]),
                household_vat=self.data_readers["world_bank"].get_tau_vat(country_name, self.year)
                * self.industry_data[country_name]["industry_vectors"]["Household Consumption in LCU"].sum(),
                export_tax=self.data_readers["world_bank"].get_tau_exp(country_name, self.year)
                * self.industry_data[country_name]["industry_vectors"]["Exports in LCU"].sum(),
                employee_si_tax=self.data_readers["oecd_econ"].read_tau_siw(country_name, self.year)
                * np.sum(self.synthetic_population[country_name].individual_data["Employee Income"]),
                income_tax=self.data_readers["oecd_econ"].read_tau_income(country_name, self.year)
                * (1 - self.data_readers["oecd_econ"].read_tau_siw(country_name, self.year))
                * np.sum(self.synthetic_population[country_name].individual_data["Employee Income"])
                + self.data_readers["oecd_econ"].read_tau_income(country_name, self.year)
                * np.sum(self.synthetic_population[country_name].household_data["Rent Paid"])
                + self.data_readers["oecd_econ"].read_tau_income(country_name, self.year)
                * np.sum(self.synthetic_population[country_name].household_data["Income from Financial Assets"]),
                rental_income_tax=self.data_readers["oecd_econ"].read_tau_income(country_name, self.year)
                * np.sum(self.synthetic_population[country_name].household_data["Rent Paid"]),
                cf_tax=self.data_readers["eurostat"].taxrate_on_capital_formation(country_name, self.year),
            )

    def insert_industry_parameters(
        self,
        parameters: dict[str, dict[str, pd.DataFrame]],
        country_name: str,
    ) -> None:
        for key in self.industry_data[country_name].keys():
            parameters[country_name][key] = self.industry_data[country_name][key]
        parameters[country_name]["industry_vectors"]["Number of Firms"] = self.synthetic_firms[
            country_name
        ].number_of_firms_by_industry
        parameters[country_name]["industry_vectors"]["Number of Employees"] = self.synthetic_population[
            country_name
        ].number_employees_by_industry

    def insert_taxes(
        self,
        parameters: dict[str, dict[str, pd.DataFrame]],
        country_name: str,
    ) -> None:
        parameters[country_name]["Taxes"] = pd.DataFrame(
            data={
                "Value-added Tax": self.data_readers["world_bank"].get_tau_vat(country_name, self.year),
                "Export Tax": self.data_readers["world_bank"].get_tau_exp(country_name, self.year),
                "Employer Social Insurance Tax": self.data_readers["oecd_econ"].read_tau_sif(country_name, self.year),
                "Employee Social Insurance Tax": self.data_readers["oecd_econ"].read_tau_siw(country_name, self.year),
                "Profit Tax": self.data_readers["oecd_econ"].read_tau_firm(country_name, self.year),
                "Income Tax": self.data_readers["oecd_econ"].read_tau_income(country_name, self.year),
                "Capital Formation Tax": self.data_readers["eurostat"].taxrate_on_capital_formation(
                    country_name, self.year
                ),
            },
            index=[self.year],
        )

    def get_parameters(self):
        parameters = {}
        for country_name in self.country_names:
            parameters[country_name] = {}
            self.insert_industry_parameters(parameters, country_name)
            self.insert_taxes(parameters, country_name)

        return parameters

    def save_individuals(self, store: pd.HDFStore, country_name: str) -> None:
        store[country_name + "_synthetic_individuals"] = (
            self.synthetic_population[country_name].individual_data.astype(float)
        ).rename_axis("Individual ID")

    def save_households(self, store: pd.HDFStore, country_name: str) -> None:
        self.synthetic_population[country_name].restrict()
        corr_individuals = self.synthetic_population[country_name].household_data["Corresponding Individuals ID"]
        corr_renters = self.synthetic_population[country_name].household_data["Corresponding Renters"]
        corr_owned_houses = self.synthetic_population[country_name].household_data[
            "Corresponding Additionally Owned Houses ID"
        ]
        household_data_without_lists = self.synthetic_population[country_name].household_data
        del household_data_without_lists["Corresponding Individuals ID"]
        del household_data_without_lists["Corresponding Renters"]
        del household_data_without_lists["Corresponding Additionally Owned Houses ID"]
        store[country_name + "_synthetic_households"] = household_data_without_lists.astype(float).rename_axis(
            "Household ID"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store[country_name + "_synthetic_households_corr_individuals"] = corr_individuals.rename_axis(
                "Household ID"
            )
            store[country_name + "_synthetic_households_corr_renters"] = corr_renters.rename_axis("Household ID")
            store[
                country_name + "_synthetic_households_corr_additionally_owned_houses"
            ] = corr_owned_houses.rename_axis("Household ID")

        # Consumption weights
        store[country_name + "_synthetic_household_consumption_weights"] = pd.DataFrame(
            data=self.synthetic_population[country_name].consumption_weights,
            index=pd.Index(self.industries, name="Industry"),
        )
        store[country_name + "_synthetic_household_consumption_weights_by_income"] = pd.DataFrame(
            data=self.synthetic_population[country_name].consumption_weights_by_income.T,
            index=pd.Index(self.industries, name="Industry"),
            columns=pd.Index(["Q1", "Q2", "Q3", "Q4", "Q5"], name="Quantile"),
        )

        # Saving rates
        sio.dump(
            obj=self.synthetic_population[country_name].saving_rates_model,
            file=self.processed_data_path.parent / "saving_rates_model.skops",
        )

        # Regular social transfers to households
        sio.dump(
            obj=self.synthetic_population[country_name].social_transfers_model,
            file=self.processed_data_path.parent / "social_transfers_model.skops",
        )

        # New wealth distribution
        sio.dump(
            obj=self.synthetic_population[country_name].wealth_distribution_model,
            file=self.processed_data_path.parent / "wealth_distribution_model.skops",
        )

        # Financial assets income multiplier
        store[country_name + "_synthetic_household_coefficient_fa_income"] = pd.DataFrame(
            data=[self.synthetic_population[country_name].coefficient_fa_income],
        )

    def save_firms(self, store: pd.HDFStore, country_name: str) -> None:
        corr_employees = self.synthetic_firms[country_name].firm_data["Employees ID"]
        firm_data_without_employees = self.synthetic_firms[country_name].firm_data
        del firm_data_without_employees["Employees ID"]
        store[country_name + "_synthetic_firms"] = firm_data_without_employees.astype(float).rename_axis("Firm ID")

        # Save the firm employees
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store[country_name + "_synthetic_firms_corr_employees"] = corr_employees

        # Save the intermediate inputs stock
        store[country_name + "_synthetic_firms_intermediate_inputs_stock"] = pd.DataFrame(
            data=self.synthetic_firms[country_name].intermediate_inputs_stock,
            index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
            columns=pd.Index(self.industries, name="Industries"),
        ).rename_axis("Firm ID")
        store[country_name + "_synthetic_firms_used_intermediate_inputs"] = pd.DataFrame(
            data=self.synthetic_firms[country_name].used_intermediate_inputs,
            index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
            columns=pd.Index(self.industries, name="Industries"),
        ).rename_axis("Firm ID")

        # Save the capital input stocks
        store[country_name + "_synthetic_firms_capital_inputs_stock"] = pd.DataFrame(
            data=self.synthetic_firms[country_name].capital_inputs_stock,
            index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
            columns=pd.Index(self.industries, name="Industries"),
        ).rename_axis("Firm ID")
        store[country_name + "_synthetic_firms_used_capital_inputs"] = pd.DataFrame(
            data=self.synthetic_firms[country_name].used_capital_inputs,
            index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
            columns=pd.Index(self.industries, name="Industries"),
        ).rename_axis("Firm ID")

    def save_banks(self, store: pd.HDFStore, country_name: str) -> None:
        corr_firms = self.synthetic_banks[country_name].bank_data["Corresponding Firms ID"]
        corr_households = self.synthetic_banks[country_name].bank_data["Corresponding Households ID"]
        bank_data_without_corr = self.synthetic_banks[country_name].bank_data
        del bank_data_without_corr["Corresponding Firms ID"]
        del bank_data_without_corr["Corresponding Households ID"]
        store[country_name + "_synthetic_banks"] = bank_data_without_corr.astype(float).rename_axis("Bank ID")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Save corresponding firms
            store[country_name + "_synthetic_banks_corr_firms"] = corr_firms.rename_axis("Bank ID")

            # Save corresponding households
            store[country_name + "_synthetic_banks_corr_households"] = corr_households.rename_axis("Bank ID")

        # Dividend payout ratio
        store[country_name + "_dividend_payout_ratio"] = pd.DataFrame(
            data=[self.data_readers["eurostat"].dividend_payout_ratio(country_name, self.year)],
        )

        # Markup on the policy rate to set bank interest rates
        store[country_name + "_policy_rate_markup"] = pd.DataFrame(
            data=[self.data_readers["eurostat"].firm_risk_premium(country_name, self.year)],
        )

        # Long-term bond interest rates
        store[country_name + "_long_term_interest_rates"] = pd.DataFrame(
            data=[
                (1.0 + self.data_readers["oecd_econ"].read_long_term_interest_rates(country_name, self.year))
                ** (1.0 / 12)
                - 1.0
            ],
        )

    def save_central_bank(self, store: pd.HDFStore, country_name: str) -> None:
        store[country_name + "_synthetic_central_bank"] = (
            self.synthetic_central_banks[country_name].central_bank_data.astype(float)
        ).rename_axis("Central Bank ID")

    def save_credit_market(self, store: pd.HDFStore, country_name: str) -> None:
        store[country_name + "_synthetic_credit_market"] = (
            self.synthetic_credit_market[country_name].credit_market_data.astype(float)
        ).rename_axis("Loans")

    def save_housing_market(self, store: pd.HDFStore, country_name: str) -> None:
        store[country_name + "_synthetic_housing_market"] = (
            self.synthetic_housing_market[country_name].housing_market_data.astype(float)
        ).rename_axis("Properties")

    def save_gov_entities(self, store: pd.HDFStore, country_name: str) -> None:
        store[country_name + "_synthetic_gov_entities"] = (
            self.synthetic_gov_entities[country_name].gov_entity_data.astype(float)
        ).rename_axis("Industry")

        # The number of government entities
        store[country_name + "_number_of_gov_entities"] = pd.DataFrame(
            data=[self.synthetic_gov_entities[country_name].number_of_entities],
        )

        # Consumption
        sio.dump(
            obj=self.synthetic_gov_entities[country_name].government_consumption_model,
            file=self.processed_data_path.parent / "government_consumption_model.skops",
        )

    def save_central_gov(self, store: pd.HDFStore, country_name: str) -> None:
        store[country_name + "_synthetic_central_gov"] = (
            self.synthetic_central_gov[country_name].central_gov_data.astype(float)
        ).rename_axis("Central Government ID")

        # Unemployment benefits
        sio.dump(
            obj=self.synthetic_central_gov[country_name].unemployment_benefits_model,
            file=self.processed_data_path.parent / "unemployment_benefits.skops",
        )

        # Other social benefits
        sio.dump(
            obj=self.synthetic_central_gov[country_name].other_benefits_model,
            file=self.processed_data_path.parent / "other_social_benefits.skops",
        )

    def save_exogenous(self, store: pd.HDFStore, country_name: str) -> None:
        if self.exogenous_data[country_name] is None:
            return
        for field in [
            "log_inflation",
            "sectoral_growth",
            "unemployment_rate",
            "vacancy_rate",
            "house_price_index",
            "total_firm_deposits_and_debt",
            "iot_industry_data",
        ]:
            if field in self.exogenous_data[country_name].keys():
                store[country_name + "_exogenous_" + field] = self.exogenous_data[country_name][field]

    def save_rest_of_the_world(self, store: pd.HDFStore) -> None:
        store["synthetic_rest_of_the_world"] = (self.synthetic_rest_of_the_world.row_data.astype(float)).rename_axis(
            "Industry"
        )

        # Models
        sio.dump(
            obj=self.synthetic_rest_of_the_world.exports_model,
            file=self.processed_data_path.parent / "row_exports_model.skops",
        )
        sio.dump(
            obj=self.synthetic_rest_of_the_world.imports_model,
            file=self.processed_data_path.parent / "row_imports_model.skops",
        )

    def save(self) -> None:
        # Create a dataset
        store = pd.HDFStore(str(self.processed_data_path), mode="w")

        # Append the config
        store["config_model"] = pd.DataFrame([str(self.config["model"])])
        store["config_init"] = pd.DataFrame([str(self.config["init"])])

        # Remember the random seed
        store["random_seed"] = pd.DataFrame([self.random_seed])

        # Save exchange rates
        store["exchange_rates"] = self.data_readers["exchange_rates"].df

        # Save the goods criticality matrix
        store["goods_criticality_matrix"] = self.data_readers["goods_criticality"].criticality_matrix

        # Get parameters
        parameters = self.get_parameters()

        # Save initial conditions
        for country_name in self.country_names:
            self.save_individuals(store, country_name)
            self.save_households(store, country_name)
            self.save_firms(store, country_name)
            self.save_banks(store, country_name)
            self.save_central_bank(store, country_name)
            self.save_gov_entities(store, country_name)
            self.save_central_gov(store, country_name)
            self.save_exogenous(store, country_name)
            self.save_credit_market(store, country_name)
            self.save_housing_market(store, country_name)
            self.save_rest_of_the_world(store)

            # Append the parameters
            for param_key in parameters[country_name].keys():
                store[country_name + "_" + param_key] = parameters[country_name][param_key].astype(float)

        # Close
        store.close()
