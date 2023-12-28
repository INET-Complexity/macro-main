from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl

from inet_data.processing import (
    SyntheticPopulation,
    SyntheticFirms,
    SyntheticCreditMarket,
    SyntheticBanks,
    SyntheticCentralBank,
    SyntheticCentralGovernment,
    SyntheticGovernmentEntities,
    SyntheticHousingMarket,
    DefaultSyntheticRestOfTheWorld,
    DefaultSyntheticCGovernment,
    DefaultSyntheticGovernmentEntities,
    DefaultSyntheticCentralBank,
    SyntheticHFCSPopulation,
    DefaultSyntheticFirms,
    DefaultSyntheticBanks,
    DefaultSyntheticHousingMarket,
    match_firms_with_banks,
    match_households_with_banks,
    set_housing_df,
    match_individuals_with_firms_country,
    create_firm_loan_df,
    create_household_loan_df,
    create_mortgage_loan_df,
)

from inet_data.readers import DataReaders, compile_industry_data, create_all_exogenous_data
from inet_data.util import process_config, get_map_long_to_short, initial_interest_rates


@dataclass
class Creator:
    """
    This class is used to create all the synthetic data for the INET model.
    Dictionaries have country names as keys and the corresponding synthetic data as values.
    """

    synthetic_population: dict[str, SyntheticPopulation]
    synthetic_firms: dict[str, SyntheticFirms]
    synthetic_banks: dict[str, SyntheticBanks]
    synthetic_credit_market: dict[str, SyntheticCreditMarket]
    synthetic_central_bank: dict[str, SyntheticCentralBank]
    synthetic_central_government: dict[str, SyntheticCentralGovernment]
    synthetic_government_entities: dict[str, SyntheticGovernmentEntities]
    synthetic_housing_market: dict[str, SyntheticHousingMarket]
    synthetic_rest_of_the_world: DefaultSyntheticRestOfTheWorld
    dividend_payout_ratio: dict[str, float]
    policy_rate_markup: dict[str, float]
    long_term_interest_rates: dict[str, float]
    goods_criticality_matrix: np.ndarray | pd.DataFrame
    exchange_rates: pd.DataFrame
    trade_proportions: pd.DataFrame

    # TODO: chunk this up into smaller functions?
    @classmethod
    def default_init(
        cls,
        configuration: str | Path | dict,
        raw_data_path: Path | str,
        random_seed: int = 0,
        create_exogenous_industry_data: bool = True,
        testing: bool = True,
    ):
        # ensure that string paths are paths
        if isinstance(raw_data_path, str):
            raw_data_path = Path(raw_data_path)

        configuration = process_config(configuration)
        np.random.seed(random_seed)

        country_names = configuration["model"]["country_names"]["value"]
        country_names_short = list(map(get_map_long_to_short(raw_data_path).get, country_names))
        industries = configuration["model"]["industries"]["value"]
        scale = configuration["model"]["scale"]["value"]
        year = configuration["model"]["year"]["value"]
        single_firm_per_industry = configuration["model"]["single_firm_per_industry"]["value"]
        single_government_entity = configuration["model"]["single_government_entity"]["value"]
        assume_zero_initial_deposits = {
            country_name: configuration["init"][country_name]["firms"]["parameters"]["assume_zero_initial_deposits"][
                "value"
            ]
            for country_name in country_names
        }
        assume_zero_initial_debt = {
            country_name: configuration["init"][country_name]["firms"]["parameters"]["assume_zero_initial_debt"][
                "value"
            ]
            for country_name in country_names
        }
        single_bank = configuration["model"]["single_bank"]["value"]

        firm_loan_maturity = {
            country: configuration["init"][country]["banks"]["parameters"]["long_term_firm_loan_maturity"]["value"]
            for country in country_names
        }

        hh_consumption_maturity = {
            country: configuration["init"][country]["banks"]["parameters"][
                "household_consumption_expansion_loan_maturity"
            ]["value"]
            for country in country_names
        }

        mortgage_maturity = {
            country: configuration["init"][country]["banks"]["parameters"]["mortgage_maturity"]["value"]
            for country in country_names
        }

        interest_rate_data = {country: initial_interest_rates(configuration, country) for country in country_names}

        assume_zero_firm_debt = {
            country: configuration["init"][country]["firms"]["parameters"]["assume_zero_initial_debt"]["value"]
            for country in country_names
        }

        prune_date = configuration["model"]["prune_date"]["value"]
        prune_date_format = configuration["model"]["prune_date"]["format"]
        readers = DataReaders.init_default_raw_data_path(
            raw_data_path=raw_data_path,
            country_names=country_names,
            country_names_short=country_names_short,
            industries=industries,
            simulation_year=year,
            scale=scale,
            prune_date=prune_date,
            create_exogenous_industry_data=create_exogenous_industry_data,
            force_single_hfcs_survey=testing,
            prune_date_format=prune_date_format,
        )

        industry_data = compile_industry_data(
            year=year, readers=readers, country_names=country_names, single_firm_per_industry=single_firm_per_industry
        )

        exogenous_data = create_all_exogenous_data(readers, country_names) if create_exogenous_industry_data else None

        year_range = 1 if testing else 10

        synthetic_central_governments = {
            country: DefaultSyntheticCGovernment.create_from_readers(readers, country, year, year_range=year_range)
            for country in country_names
        }

        total_unemployment_benefits = {
            country: synthetic_central_governments[country].central_gov_data["Total Unemployment Benefits"].values[0]
            for country in country_names
        }

        synthetic_gov_entities = {
            country: DefaultSyntheticGovernmentEntities.create_from_readers(
                readers=readers,
                country_name=country,
                year=year,
                exogenous_country_data=exogenous_data.get(country, None) if exogenous_data else None,
                industry_data=industry_data,
                single_government_entity=single_government_entity,
            )
            for country in country_names
        }

        synthetic_central_banks = {
            country: DefaultSyntheticCentralBank.init_from_readers(country, year, readers) for country in country_names
        }

        synthetic_population: dict[str, SyntheticHFCSPopulation] = {
            country: SyntheticHFCSPopulation.create_from_readers(
                readers=readers,
                country_name=country,
                country_name_short=country_short,
                year=year,
                scale=scale,
                industries=industries,
                industry_data=industry_data,
                total_unemployment_benefits=total_unemployment_benefits[country],
            )
            for (country, country_short) in zip(country_names, country_names_short)
        }

        synthetic_firms = {
            country: DefaultSyntheticFirms.init_from_readers(
                readers=readers,
                country_name=country,
                year=year,
                industry_data=industry_data[country],
                assume_zero_initial_deposits=assume_zero_initial_deposits[country],
                assume_zero_initial_debt=assume_zero_initial_debt[country],
                industries=industries,
                n_employees_per_industry=synthetic_population[country].number_employees_by_industry,
                scale=scale,
            )
            for country in country_names
        }

        synthetic_banks = {
            country: DefaultSyntheticBanks.init_from_readers(
                single_bank=single_bank,
                country_name=country,
                year=year,
                readers=readers,
                scale=scale,
            )
            for country in country_names
        }

        synthetic_row = DefaultSyntheticRestOfTheWorld.init_from_readers(
            readers=readers,
            year=year,
            exogenous_row_data=exogenous_data.get("ROW", None) if exogenous_data else None,
            row_industry_data=industry_data["ROW"],
        )

        # housing market data is initialised through the matching
        # where a dictionary is created with country names as keys and housing market data as values
        synthetic_housing_market = {}

        # credit market data is initialised through the matching
        # where a dictionary is created with country names as keys and credit market data as values
        synthetic_credit_market = {}

        for country_name in country_names:
            match_individuals_with_firms_country(
                country_name,
                industries,
                readers,
                synthetic_firms[country_name],
                synthetic_population[country_name],
                year,
            )

            match_firms_with_banks(
                synthetic_firms=synthetic_firms[country_name],
                synthetic_banks=synthetic_banks[country_name],
            )
            match_households_with_banks(
                synthetic_population=synthetic_population[country_name],
                synthetic_banks=synthetic_banks[country_name],
            )

            country_housing_data = set_housing_df(
                synthetic_population[country_name],
                rental_income_taxes=readers.oecd_econ.read_tau_income(country_name, year),
                social_housing_rent=synthetic_population[country_name].social_housing_rent,
                total_imputed_rent=readers.icio[year].imputed_rents[country_name],
            )

            synthetic_housing_market[country_name] = DefaultSyntheticHousingMarket(
                year=year,
                country_name=country_name,
                housing_market_data=country_housing_data,
            )

            # TODO : these functions do things that depend on the function parameters
            # they need to be moved to the model package
            synthetic_population[country_name].compute_household_wealth()
            synthetic_population[country_name].compute_household_income(
                total_social_transfers=synthetic_central_governments[country_name]
                .central_gov_data["Other Social Benefits"]
                .values[0],
            )

            # TODO: same for set household savings rates
            synthetic_population[country_name].set_household_saving_rates()

            iot_hh_consumption = industry_data[country_name]["industry_vectors"]["Household Consumption in LCU"]
            vat = readers.world_bank.get_tau_vat(country_name, year)
            synthetic_population[country_name].normalise_household_consumption(
                iot_hh_consumption=iot_hh_consumption, vat=vat
            )

            weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(
                country=country_name, year=year
            )

            synthetic_population[country_name].match_consumption_weights_by_income(
                weights_by_income=weights_by_income, iot_hh_consumption=iot_hh_consumption, vat=vat
            )

            synthetic_banks[country_name].initialise_deposits_and_loans(
                synthetic_population=synthetic_population[country_name],
                synthetic_firms=synthetic_firms[country_name],
            )

            synthetic_banks[country_name].initialise_rates_profits_liabilities(
                readers, **interest_rate_data[country_name]
            )

            if assume_zero_firm_debt[country_name]:
                firm_loan_df = create_firm_loan_df(
                    synthetic_firms[country_name],
                    synthetic_banks[country_name],
                    firm_loan_maturity=firm_loan_maturity[country_name],
                )
            else:
                firm_loan_df = None

            household_loan_df = create_household_loan_df(
                synthetic_population[country_name],
                synthetic_banks[country_name],
                hh_consumption_maturity[country_name],
            )

            mortgage_loan_df = create_mortgage_loan_df(
                synthetic_population[country_name],
                synthetic_banks[country_name],
                mortgage_maturity[country_name],
            )

            valid_firm_df = (firm_loan_df is not None) and (firm_loan_df.shape[0] > 0)

            if valid_firm_df:
                credit_list = [firm_loan_df, household_loan_df, mortgage_loan_df]
            else:
                credit_list = [household_loan_df, mortgage_loan_df]

            credit_market_data = pd.concat(credit_list, ignore_index=True)

            credit_market_data.index.name = "Loans"
            credit_market_data.columns.name = "Loan Properties"

            synthetic_credit_market[country_name] = SyntheticCreditMarket(
                year=year,
                country_name=country_name,
                credit_market_data=credit_market_data,
            )

            synthetic_population[country_name].set_debt_installments(synthetic_credit_market[country_name])

            synthetic_firms[country_name].set_additional_initial_conditions(
                readers=readers,
                industry_data=industry_data[country_name],
                synthetic_banks=synthetic_banks[country_name],
                synthetic_credit_market=synthetic_credit_market[country_name],
            )

            synthetic_central_governments[country_name].update_fields(
                readers,
                synthetic_population[country_name],
                synthetic_firms[country_name],
                synthetic_banks[country_name],
                industry_data[country_name],
            )

            # don't save all columns for temporary population
            synthetic_population[country_name].restrict()

        dividend_payout_ratio = {
            country: readers.eurostat.dividend_payout_ratio(country, year) for country in country_names
        }
        exchange_rates = readers.exchange_rates.df

        long_term_interest_rates = {
            country: readers.oecd_econ.read_long_term_interest_rates(country, year) for country in country_names
        }

        policy_rate_markup = {country: readers.eurostat.firm_risk_premium(country, year) for country in country_names}

        trade_proportions = readers.icio[year].get_trade_proportions()

        goods_criticality = readers.goods_criticality.criticality_matrix

        return cls(
            synthetic_population=synthetic_population,
            synthetic_firms=synthetic_firms,
            synthetic_banks=synthetic_banks,
            synthetic_credit_market=synthetic_credit_market,
            synthetic_central_bank=synthetic_central_banks,
            synthetic_central_government=synthetic_central_governments,
            synthetic_government_entities=synthetic_gov_entities,
            synthetic_housing_market=synthetic_housing_market,
            synthetic_rest_of_the_world=synthetic_row,
            dividend_payout_ratio=dividend_payout_ratio,
            exchange_rates=exchange_rates,
            long_term_interest_rates=long_term_interest_rates,
            policy_rate_markup=policy_rate_markup,
            trade_proportions=trade_proportions,
            goods_criticality_matrix=goods_criticality,
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
