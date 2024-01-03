import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inet_data.configuration import Configuration
from inet_data.processing import (
    DefaultSyntheticRestOfTheWorld,
    SyntheticCountry,
)
from inet_data.readers import DataReaders, compile_industry_data, create_all_exogenous_data
from inet_data.util import get_map_long_to_short


@dataclass
class Creator:
    """
    This class is used to create all the synthetic data for the INET model.
    Dictionaries have country names as keys and the corresponding synthetic data as values.
    """

    synthetic_countries: dict[str, SyntheticCountry]
    synthetic_rest_of_the_world: DefaultSyntheticRestOfTheWorld
    goods_criticality_matrix: np.ndarray | pd.DataFrame
    exchange_rates: pd.DataFrame
    trade_proportions: pd.DataFrame

    @classmethod
    def default_init(
        cls,
        configuration: Configuration,
        raw_data_path: Path | str,
        random_seed: int = 0,
        create_exogenous_industry_data: bool = True,
        single_hfcs_survey: bool = True,
    ) -> "Creator":
        # ensure that string paths are paths
        if isinstance(raw_data_path, str):
            raw_data_path = Path(raw_data_path)

        np.random.seed(random_seed)

        country_names = configuration.countries
        country_names_short = list(map(get_map_long_to_short(raw_data_path).get, country_names))
        industries = configuration.industries
        year = configuration.year

        scale_dict = {country: configuration.country_configs[country].scale for country in country_names}

        prune_date = configuration.prune_date
        readers = DataReaders.from_raw_data(
            raw_data_path=raw_data_path,
            country_names=country_names,
            country_names_short=country_names_short,
            industries=industries,
            simulation_year=year,
            scale_dict=scale_dict,
            prune_date=prune_date,
            create_exogenous_industry_data=create_exogenous_industry_data,
            force_single_hfcs_survey=single_hfcs_survey,
        )

        single_firm_dict = {
            country: configuration.country_configs[country].single_firm_per_industry for country in country_names
        }

        industry_data = compile_industry_data(
            year=year, readers=readers, country_names=country_names, single_firm_per_industry=single_firm_dict
        )

        year_range = 1 if single_hfcs_survey else 10

        exogenous_data = create_all_exogenous_data(readers, country_names) if create_exogenous_industry_data else None

        # currently only EU countries implemented

        synthetic_countries = {
            country: SyntheticCountry.create_eu_synthetic_country(
                country=country,
                year=year,
                country_configuration=configuration.country_configs[country],
                industries=industries,
                readers=readers,
                exogenous_country_data=exogenous_data.get(country, None) if exogenous_data else None,
                country_industry_data=industry_data[country],
                year_range=year_range,
            )
            for country in country_names
        }

        goods_criticality = readers.goods_criticality.criticality_matrix

        synthetic_row = DefaultSyntheticRestOfTheWorld.from_readers(
            readers=readers,
            year=year,
            exogenous_row_data=exogenous_data.get("ROW", None) if exogenous_data else None,
            row_industry_data=industry_data["ROW"],
        )

        exchange_rates = readers.exchange_rates.df
        trade_proportions = readers.icio[year].get_trade_proportions()

        return cls(
            synthetic_countries=synthetic_countries,
            synthetic_rest_of_the_world=synthetic_row,
            goods_criticality_matrix=goods_criticality,
            exchange_rates=exchange_rates,
            trade_proportions=trade_proportions,
        )

    @classmethod
    def init_from_pickle(cls, path: str | Path) -> "Creator":
        """
        Initialise the creator from a pickle file.

        Args:
            path (str or Path): The path to the pickle file.

        Returns:
            Creator: The creator.
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "rb") as f:
            data = pkl.load(f)

        return cls(**data)

    def save(self, path: str | Path) -> None:
        """
        Save the synthetic data to a pickle file.

        Args:
            path (str or Path): The path to the pickle file.
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "wb") as f:
            pkl.dump(self.__dict__, f)

    # keeping the below legacy code for now, but it should be removed in the future
    #
    # def save_individuals(self, store: pd.HDFStore, country_name: str) -> None:
    #     store[country_name + "_synthetic_individuals"] = (
    #         self.synthetic_population[country_name].individual_data.astype(float)
    #     ).rename_axis("Individual ID")
    #
    # def save_households(self, store: pd.HDFStore, country_name: str, processed_data_path: Path) -> None:
    #     corr_individuals = self.synthetic_population[country_name].household_data["Corresponding Individuals ID"]
    #     corr_renters = self.synthetic_population[country_name].household_data["Corresponding Renters"]
    #     corr_owned_houses = self.synthetic_population[country_name].household_data[
    #         "Corresponding Additionally Owned Houses ID"
    #     ]
    #     household_data_without_lists = self.synthetic_population[country_name].household_data
    #     del household_data_without_lists["Corresponding Individuals ID"]
    #     del household_data_without_lists["Corresponding Renters"]
    #     del household_data_without_lists["Corresponding Additionally Owned Houses ID"]
    #     store[country_name + "_synthetic_households"] = household_data_without_lists.astype(float).rename_axis(
    #         "Household ID"
    #     )
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         store[country_name + "_synthetic_households_corr_individuals"] = corr_individuals.rename_axis(
    #             "Household ID"
    #         )
    #         store[country_name + "_synthetic_households_corr_renters"] = corr_renters.rename_axis("Household ID")
    #         store[
    #             country_name + "_synthetic_households_corr_additionally_owned_houses"
    #         ] = corr_owned_houses.rename_axis("Household ID")
    #
    #     # Consumption weights
    #     store[country_name + "_synthetic_household_consumption_weights"] = pd.DataFrame(
    #         data=self.synthetic_population[country_name].consumption_weights,
    #         index=pd.Index(self.industries, name="Industry"),
    #     )
    #     store[country_name + "_synthetic_household_consumption_weights_by_income"] = pd.DataFrame(
    #         data=self.synthetic_population[country_name].consumption_weights_by_income.T,
    #         index=pd.Index(self.industries, name="Industry"),
    #         columns=pd.Index(["Q1", "Q2", "Q3", "Q4", "Q5"], name="Quantile"),
    #     )
    #
    #     # Saving rates
    #     sio.dump(
    #         obj=self.synthetic_population[country_name].saving_rates_model,
    #         file=processed_data_path.parent / "saving_rates_model.skops",
    #     )
    #
    #     # Regular social transfers to households
    #     sio.dump(
    #         obj=self.synthetic_population[country_name].social_transfers_model,
    #         file=processed_data_path.parent / "social_transfers_model.skops",
    #     )
    #
    #     # New wealth distribution
    #     sio.dump(
    #         obj=self.synthetic_population[country_name].wealth_distribution_model,
    #         file=processed_data_path.parent / "wealth_distribution_model.skops",
    #     )
    #
    #     # Financial assets income multiplier
    #     store[country_name + "_synthetic_household_coefficient_fa_income"] = pd.DataFrame(
    #         data=[self.synthetic_population[country_name].coefficient_fa_income],
    #     )
    #
    # def save_firms(self, store: pd.HDFStore, country_name: str) -> None:
    #     corr_employees = self.synthetic_firms[country_name].firm_data["Employees ID"]
    #     firm_data_without_employees = self.synthetic_firms[country_name].firm_data
    #     del firm_data_without_employees["Employees ID"]
    #     store[country_name + "_synthetic_firms"] = firm_data_without_employees.astype(float).rename_axis("Firm ID")
    #
    #     # Save the firm employees
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         store[country_name + "_synthetic_firms_corr_employees"] = corr_employees
    #
    #     # Save the intermediate inputs stock
    #     store[country_name + "_synthetic_firms_intermediate_inputs_stock"] = pd.DataFrame(
    #         data=self.synthetic_firms[country_name].intermediate_inputs_stock,
    #         index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
    #         columns=pd.Index(self.industries, name="Industries"),
    #     ).rename_axis("Firm ID")
    #     store[country_name + "_synthetic_firms_used_intermediate_inputs"] = pd.DataFrame(
    #         data=self.synthetic_firms[country_name].used_intermediate_inputs,
    #         index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
    #         columns=pd.Index(self.industries, name="Industries"),
    #     ).rename_axis("Firm ID")
    #
    #     # Save the capital input stocks
    #     store[country_name + "_synthetic_firms_capital_inputs_stock"] = pd.DataFrame(
    #         data=self.synthetic_firms[country_name].capital_inputs_stock,
    #         index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
    #         columns=pd.Index(self.industries, name="Industries"),
    #     ).rename_axis("Firm ID")
    #     store[country_name + "_synthetic_firms_used_capital_inputs"] = pd.DataFrame(
    #         data=self.synthetic_firms[country_name].used_capital_inputs,
    #         index=pd.Index(range(len(self.synthetic_firms[country_name].firm_data)), name="Firm ID"),
    #         columns=pd.Index(self.industries, name="Industries"),
    #     ).rename_axis("Firm ID")
    #
    # def save_banks(self, store: pd.HDFStore, country_name: str) -> None:
    #     corr_firms = self.synthetic_banks[country_name].bank_data["Corresponding Firms ID"]
    #     corr_households = self.synthetic_banks[country_name].bank_data["Corresponding Households ID"]
    #     bank_data_without_corr = self.synthetic_banks[country_name].bank_data
    #     del bank_data_without_corr["Corresponding Firms ID"]
    #     del bank_data_without_corr["Corresponding Households ID"]
    #     store[country_name + "_synthetic_banks"] = bank_data_without_corr.astype(float).rename_axis("Bank ID")
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #
    #         # Save corresponding firms
    #         store[country_name + "_synthetic_banks_corr_firms"] = corr_firms.rename_axis("Bank ID")
    #
    #         # Save corresponding households
    #         store[country_name + "_synthetic_banks_corr_households"] = corr_households.rename_axis("Bank ID")
    #
    #     # Dividend payout ratio
    #     store[country_name + "_dividend_payout_ratio"] = pd.DataFrame(
    #         data=[self.data_readers["eurostat"].dividend_payout_ratio(country_name, self.year)],
    #     )
    #
    #     # Markup on the policy rate to set bank interest rates
    #     store[country_name + "_policy_rate_markup"] = pd.DataFrame(
    #         data=[self.data_readers["eurostat"].firm_risk_premium(country_name, self.year)],
    #     )
    #
    #     # Long-term bond interest rates
    #     store[country_name + "_long_term_interest_rates"] = pd.DataFrame(
    #         data=[
    #             (1.0 + self.data_readers["oecd_econ"].read_long_term_interest_rates(country_name, self.year))
    #             ** (1.0 / 12)
    #             - 1.0
    #         ],
    #     )
    #
    # def save_central_bank(self, store: pd.HDFStore, country_name: str) -> None:
    #     store[country_name + "_synthetic_central_bank"] = (
    #         self.synthetic_central_banks[country_name].central_bank_data.astype(float)
    #     ).rename_axis("Central Bank ID")
    #
    # def save_credit_market(self, store: pd.HDFStore, country_name: str) -> None:
    #     store[country_name + "_synthetic_credit_market"] = (
    #         self.synthetic_credit_market[country_name].credit_market_data.astype(float)
    #     ).rename_axis("Loans")
    #
    # def save_housing_market(self, store: pd.HDFStore, country_name: str) -> None:
    #     store[country_name + "_synthetic_housing_market"] = (
    #         self.synthetic_housing_market[country_name].housing_market_data.astype(float)
    #     ).rename_axis("Properties")
    #
    # def save_gov_entities(self, store: pd.HDFStore, country_name: str) -> None:
    #     store[country_name + "_synthetic_gov_entities"] = (
    #         self.synthetic_gov_entities[country_name].gov_entity_data.astype(float)
    #     ).rename_axis("Industry")
    #
    #     # The number of government entities
    #     store[country_name + "_number_of_gov_entities"] = pd.DataFrame(
    #         data=[self.synthetic_gov_entities[country_name].number_of_entities],
    #     )
    #
    #     # Consumption
    #     sio.dump(
    #         obj=self.synthetic_gov_entities[country_name].government_consumption_model,
    #         file=processed_data_path.parent / "government_consumption_model.skops",
    #     )
    #
    # def save_central_gov(self, store: pd.HDFStore, country_name: str) -> None:
    #     store[country_name + "_synthetic_central_gov"] = (
    #         self.synthetic_central_gov[country_name].central_gov_data.astype(float)
    #     ).rename_axis("Central Government ID")
    #
    #     # Unemployment benefits
    #     sio.dump(
    #         obj=self.synthetic_central_gov[country_name].unemployment_benefits_model,
    #         file=processed_data_path.parent / "unemployment_benefits.skops",
    #     )
    #
    #     # Other social benefits
    #     sio.dump(
    #         obj=self.synthetic_central_gov[country_name].other_benefits_model,
    #         file=processed_data_path.parent / "other_social_benefits.skops",
    #     )
    #
    # def save_exogenous(self, store: pd.HDFStore, country_name: str) -> None:
    #     if self.exogenous_data[country_name] is None:
    #         return
    #     for field in [
    #         "log_inflation",
    #         "sectoral_growth",
    #         "unemployment_rate",
    #         "vacancy_rate",
    #         "house_price_index",
    #         "total_firm_deposits_and_debt",
    #         "iot_industry_data",
    #     ]:
    #         if field in self.exogenous_data[country_name].keys():
    #             store[country_name + "_exogenous_" + field] = self.exogenous_data[country_name][field]
    #
    # def save_rest_of_the_world(self, store: pd.HDFStore, processed_data_path: Path) -> None:
    #     store["synthetic_rest_of_the_world"] = (self.synthetic_rest_of_the_world.row_data.astype(float)).rename_axis(
    #         "Industry"
    #     )
    #
    #     # Models
    #     sio.dump(
    #         obj=self.synthetic_rest_of_the_world.exports_model,
    #         file=processed_data_path.parent / "row_exports_model.skops",
    #     )
    #     sio.dump(
    #         obj=self.synthetic_rest_of_the_world.imports_model,
    #         file=processed_data_path.parent / "row_imports_model.skops",
    #     )
    #
    # def save_hdf(self, processed_data_path: Path) -> None:
    #     # Create a dataset
    #     store = pd.HDFStore(str(processed_data_path), mode="w")
    #
    #     # Append the config
    #     store["config_model"] = pd.DataFrame([str(self.config["model"])])
    #     store["config_init"] = pd.DataFrame([str(self.config["init"])])
    #
    #     # Remember the random seed
    #     store["random_seed"] = pd.DataFrame([self.random_seed])
    #
    #     # Save exchange rates
    #     store["exchange_rates"] = self.data_readers["exchange_rates"].df
    #
    #     # Save the goods criticality matrix
    #     store["goods_criticality_matrix"] = self.data_readers["goods_criticality"].criticality_matrix
    #
    #     # Save the trade proportions
    #     store["trade_proportions"] = self.data_readers["icio"][self.year].get_trade_proportions()
    #
    #     # Get parameters
    #     parameters = self.get_parameters()
    #
    #     # Save initial conditions
    #     for country_name in self.country_names:
    #         self.save_individuals(store, country_name)
    #         self.save_households(store, country_name)
    #         self.save_firms(store, country_name)
    #         self.save_banks(store, country_name)
    #         self.save_central_bank(store, country_name)
    #         self.save_gov_entities(store, country_name)
    #         self.save_central_gov(store, country_name)
    #         self.save_exogenous(store, country_name)
    #         self.save_credit_market(store, country_name)
    #         self.save_housing_market(store, country_name)
    #         self.save_rest_of_the_world(store)
    #
    #         # Append the parameters
    #         for param_key in parameters[country_name].keys():
    #             store[country_name + "_" + param_key] = parameters[country_name][param_key].astype(float)
    #
    #     # Close
    #     store.close()
