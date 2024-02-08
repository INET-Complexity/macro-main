import numpy as np
import pandas as pd
import warnings
import h5py
from inet_data import SyntheticPopulation
from typing import Any, Optional

from configurations import HouseholdsConfiguration
from inet_macromodel.agents.agent import Agent
from inet_macromodel.banks.banks import Banks
from inet_macromodel.credit_market.credit_market import CreditMarket
from inet_macromodel.goods_market.value_type import ValueType
from inet_macromodel.households.household_properties import HouseholdType
from inet_macromodel.households.households_ts import create_households_timeseries
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.util.function_mapping import functions_from_model
from inet_macromodel.util.get_histogram import get_histogram
from inet_macromodel.util.property_mapping import map_to_enum


class Households(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        use_consumption_weights_by_income: bool,
    ):
        n_entities = ts.current("n_households")
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_entities,
            n_entities,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.NOMINAL,
                "Seller Value Type": ValueType.NONE,
                "Buyer Priority": 0,
                "Seller Priority": 0,
            },
        )

        self.functions = functions

        # Set initial values
        self.ts["saving_rates_histogram"] = get_histogram(self.get_saving_rates_by_household(), None)

        self.consumption_weights = consumption_weights
        self.consumption_weights_by_income = consumption_weights_by_income

        self.use_consumption_weights_by_income = use_consumption_weights_by_income

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_population: SyntheticPopulation,
        configuration: HouseholdsConfiguration,
        country_name: str,
        all_country_names: list[str],
        industries: list[str],
        initial_consumption_by_industry: np.ndarray,
        value_added_tax: float,
        scale: int,
    ) -> "Households":
        individual_ages = synthetic_population.individual_data["Age"].values

        corr_individuals = synthetic_population.household_data["Corresponding Individuals ID"]
        corr_individuals = corr_individuals.rename_axis("Household ID")

        corr_renters = synthetic_population.household_data["Corresponding Renters"]
        corr_renters = corr_renters.rename_axis("Household ID")

        corr_owned_houses = synthetic_population.household_data["Corresponding Additionally Owned Houses ID"]
        corr_owned_houses = corr_owned_houses.rename_axis("Household ID")

        functions = functions_from_model(model=configuration.functions, loc="inet_macromodel.households")

        hh_data = (
            synthetic_population.household_data.drop(
                columns=[
                    "Corresponding Individuals ID",
                    "Corresponding Renters",
                    "Corresponding Additionally Owned Houses ID",
                ]
            )
            .astype(float)
            .rename_axis("Household ID")
        )

        consumption_weights = synthetic_population.consumption_weights

        consumption_weights_by_income = synthetic_population.consumption_weights_by_income.T

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray] | Any] = {
            "saving_rates_model": synthetic_population.saving_rates_model,
            "social_transfers_model": synthetic_population.social_transfers_model,
            "wealth_distribution_model": synthetic_population.wealth_distribution_model,
            "average_saving_rate": synthetic_population.household_data["Saving Rate"].mean(),
            "coefficient_fa_income": synthetic_population.coefficient_fa_income,
        }

        # Additional states
        for state_name in [
            "Type",
            "Corresponding Bank ID",
            "Corresponding Inhabited House ID",
            "Corresponding Property Owner",
            "Tenure Status of the Main Residence",
        ]:
            if state_name not in hh_data.columns:
                raise ValueError(f"Missing {state_name} from the data for initialising households.")
            if state_name == "Type":
                states[state_name] = hh_data[state_name].values.flatten()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=RuntimeWarning)
                    states[state_name] = hh_data[state_name].values.astype(int).flatten()
                    states[state_name][states[state_name] < 0] = -1

        ts = create_households_timeseries(
            data=hh_data,
            initial_consumption_by_industry=initial_consumption_by_industry,
            scale=scale,
            vat=value_added_tax,
        )

        # Update the household type
        states["Type"] = map_to_enum(states["Type"], HouseholdType)

        # Corresponding individuals
        states["corr_individuals"] = [corr_individuals.values[i][0] for i in range(len(corr_individuals.values))]

        # Number of adults individuals in the household
        states["Number of Adults"] = np.array(
            [
                np.sum(individual_ages[states["corr_individuals"][hh_id]] >= 18)
                for hh_id in range(ts.current("n_households"))
            ]
        )

        # Corresponding renters
        states["corr_renters"] = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_renters]

        use_consumption_weights_by_income = configuration.use_consumption_weights_by_income

        # TODO: corresponding additionally owned houses is not used

        return cls(
            country_name,
            all_country_names,
            len(industries),
            functions,
            ts,
            states,
            consumption_weights,
            consumption_weights_by_income,
            use_consumption_weights_by_income,
        )

    @staticmethod
    def compute_employee_income(individual_income: np.ndarray, corr_households: np.ndarray) -> np.ndarray:
        return np.bincount(corr_households, weights=individual_income)

    def compute_social_transfer_income(
        self,
        total_other_social_transfers: float,
    ) -> np.ndarray:
        inds = ["Income", "Debt"]
        return self.functions["social_transfers"].get_social_transfers(
            n_households=self.ts.current("n_households"),
            total_other_social_transfers=total_other_social_transfers,
            current_independents=(
                np.array([])
                if len(inds) == 0
                else np.stack(
                    [self.ts.current(ind.lower()) for ind in inds],
                    axis=1,
                )
            ),
            initial_independents=(
                np.array([])
                if len(inds) == 0
                else np.stack(
                    [self.ts.initial(ind.lower()) for ind in inds],
                    axis=1,
                )
            ),
            model=self.states["social_transfers_model"],
        )

    def compute_rental_income(
        self,
        housing_data: pd.DataFrame,
        income_taxes: float,
    ) -> np.ndarray:
        housing_data_rented_out = housing_data.loc[
            (housing_data["Is Owner-Occupied"] == 0) & (housing_data["Corresponding Inhabitant Household ID"] != -1)
        ]
        housing_data_rented_out_grouped = housing_data_rented_out.groupby("Corresponding Owner Household ID")[
            "Rent"
        ].sum()
        rental_income = np.zeros(self.ts.current("n_households"))
        rental_income[housing_data_rented_out_grouped.index.values] = (
            1 - income_taxes
        ) * housing_data_rented_out_grouped.values
        return rental_income

    def compute_income_from_financial_assets(self) -> np.ndarray:
        return self.functions["financial_assets"].compute_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def compute_income(self) -> np.ndarray:
        return (
            self.ts.current("income_employee")
            + self.ts.current("income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("income_financial_assets")
        )

    def get_saving_rates_by_household(self, independents: Optional[list[str]] = None) -> np.ndarray:
        if independents is None:
            independents = ["Income", "Debt"]

        return self.functions["saving_rates"].get_saving_rates(
            n_households=self.ts.current("n_households"),
            average_saving_rate=self.states["average_saving_rate"],
            current_independents=np.stack(
                [self.ts.current(ind.lower()) for ind in independents],
                axis=1,
            ),
            initial_independents=np.stack(
                [self.ts.initial(ind.lower()) for ind in independents],
                axis=1,
            ),
            model=self.states["saving_rates_model"],
        )

    def compute_target_consumption_before_ce(
        self,
        per_capita_unemployment_benefits: float,
        tau_vat: float,
    ) -> np.ndarray:
        saving_rates = self.get_saving_rates_by_household()
        self.ts.saving_rates_histogram.append(get_histogram(saving_rates, None))
        return self.functions["consumption"].compute_target_consumption_before_ce(
            saving_rates=saving_rates,
            income=self.ts.current("income"),
            household_benefits=self.states["Number of Adults"] * per_capita_unemployment_benefits
            + self.ts.current("income_social_transfers"),
            historic_consumption=self.ts.historic("consumption"),
            consumption_weights=self.consumption_weights,
            consumption_weights_by_income=self.consumption_weights_by_income,
            take_consumption_weights_by_income_quantile=self.use_consumption_weights_by_income,
            tau_vat=tau_vat,
        )

    def prepare_housing_market_clearing(
        self,
        housing_data: pd.DataFrame,
        observed_fraction_value_price: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
        expected_hpi_growth: float,
        assumed_mortgage_maturity: int,
        rental_income_taxes: float,
    ) -> None:
        # Households make decisions on their demand for properties
        (
            max_price_willing_to_pay,
            max_rent_willing_to_pay,
            households_hoping_to_move,
        ) = self.functions["property"].compute_demand(
            housing_data=housing_data,
            household_residence_tenure_status=self.states["Tenure Status of the Main Residence"],
            household_income=self.ts.current("income"),
            household_financial_wealth=self.ts.current("wealth_financial_assets"),
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=expected_hpi_growth,
            assumed_mortgage_maturity=assumed_mortgage_maturity,
            rental_income_taxes=rental_income_taxes,
        )
        self.ts.max_price_willing_to_pay.append(max_price_willing_to_pay)
        self.ts.max_rent_willing_to_pay.append(max_rent_willing_to_pay)

        # Set price of properties of households that are hoping to move
        ind_mhr_temp_sale = housing_data.loc[
            housing_data["Corresponding Owner Household ID"].isin(households_hoping_to_move)
        ].index
        housing_data["Temporarily for Sale"] = False
        housing_data.loc[ind_mhr_temp_sale, "Temporarily for Sale"] = True
        housing_data["Sale Price"] = np.nan
        housing_data.loc[ind_mhr_temp_sale, "Sale Price"] = housing_data.loc[ind_mhr_temp_sale, "Value"]

        # Set what's up for rent
        prev_up_for_rent = housing_data["Up for Rent"].values
        now_up_for_rent = np.where(np.isnan(housing_data["Corresponding Inhabitant Household ID"].values))[0]
        newly_up_for_rent = [ind for ind in now_up_for_rent if ind not in prev_up_for_rent]
        housing_data["Up for Rent"] = False
        housing_data.loc[now_up_for_rent, "Up for Rent"] = True
        housing_data["Newly on the Rental Market"] = False
        housing_data.loc[newly_up_for_rent, "Newly on the Rental Market"] = True

        # Calculate rent
        ind_hh_newly_on_renting_market = housing_data["Newly on the Rental Market"] == True
        ind_hh_not_newly_on_renting_market = housing_data["Newly on the Rental Market"] == False
        housing_data.loc[ind_hh_newly_on_renting_market, "Rent"] = self.functions[
            "rent"
        ].compute_offered_rent_for_new_properties(
            property_value=housing_data.loc[ind_hh_newly_on_renting_market, "Value"].values,
            observed_fraction_rent_value=observed_fraction_rent_value,
        )
        housing_data.loc[ind_hh_not_newly_on_renting_market, "Rent"] = self.functions[
            "rent"
        ].compute_offered_rent_for_existing_properties(
            current_offered_rent=housing_data.loc[ind_hh_not_newly_on_renting_market, "Rent"].values,
        )

    def update_rent(
        self,
        housing_data: pd.DataFrame,
        historic_inflation: list[np.ndarray],
        exogenous_inflation_before: np.ndarray,
    ) -> None:
        housing_data["Rent"] = self.functions["rent"].compute_rent(
            current_rent=housing_data["Rent"].values,
            historic_inflation=np.concatenate((exogenous_inflation_before, np.array(historic_inflation).flatten())),
        )

    def process_housing_market_clearing(
        self,
        housing_data: pd.DataFrame,
        social_housing_function: Any,
        current_sales: pd.DataFrame,
        current_unemployment_benefits_by_individual: float,
    ) -> None:
        # Calculate rent
        rent_by_household, imputed_rent_by_household = self.compute_rent(
            housing_data=housing_data,
            social_housing_function=social_housing_function,
            current_unemployment_benefits_by_individual=current_unemployment_benefits_by_individual,
        )
        self.ts.rent.append(rent_by_household)
        self.ts.rent_imputed.append(imputed_rent_by_household)

        # Calculate the price paid for property
        price_paid_for_property = np.zeros(self.ts.current("n_households"))
        if len(current_sales) > 0:
            price_paid_for_property[current_sales["buyer_id"].values] = current_sales["price_or_rent"].values
        self.ts.price_paid_for_property.append(price_paid_for_property)

    def compute_rent(
        self,
        housing_data: pd.DataFrame,
        social_housing_function: Any,
        current_unemployment_benefits_by_individual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rent_by_household = np.zeros(self.ts.current("n_households"))
        imputed_rent_by_household = np.zeros(self.ts.current("n_households"))

        # Households in social housing
        ind_social_housing = np.where(self.states["Corresponding Inhabited House ID"] == -1)[0]
        social_housing_rent = social_housing_function.compute_social_housing_rent(
            current_unemployment_benefits_by_individual=current_unemployment_benefits_by_individual,
            current_household_size=self.states["Number of Adults"][ind_social_housing],
        )
        rent_by_household[ind_social_housing] = social_housing_rent
        self.states["Rent paid to Government"] = social_housing_rent.sum()

        # Households renting
        ind_renting = self.states["Tenure Status of the Main Residence"] == 0
        rent = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_renting],
            "Rent",
        ].values
        rent_by_household[ind_renting] = rent

        # Households owning
        ind_owning = self.states["Tenure Status of the Main Residence"] == 1
        rent = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_owning],
            "Rent",
        ].values
        imputed_rent_by_household[ind_owning] = rent

        return rent_by_household, imputed_rent_by_household

    def compute_target_credit(self, current_sales: pd.DataFrame) -> None:
        # Target payday loans to cover immediate financing gaps
        self.ts.target_payday_loans.append(
            self.functions["target_credit"].compute_target_payday_loans(
                target_consumption_before_ce=self.ts.current("target_consumption_before_ce"),
                income=self.ts.current("income"),
                rent=self.ts.current("rent"),
                wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
            )
        )
        self.ts.total_target_payday_loans.append([self.ts.current("target_payday_loans").sum()])

        # Consumption expansion loans to afford additional purchases
        (
            target_consumption_ce,
            target_consumption_expansion_loans,
        ) = self.functions["target_credit"].compute_consumption_expansion_loans(
            current_income=self.ts.current("income"),
            initial_income=self.ts.initial("income"),
            current_wealth_other_real_assets=self.ts.current("wealth_other_real_assets"),
            initial_wealth_other_real_assets=self.ts.initial("wealth_other_real_assets"),
            target_consumption_before_ce=self.ts.current("target_consumption_before_ce"),
            income=self.ts.current("income"),
            rent=self.ts.current("rent"),
            wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
        )
        self.ts.target_consumption_ce.append(target_consumption_ce)
        self.ts.target_consumption_expansion_loans.append(target_consumption_expansion_loans)
        self.ts.total_target_consumption_expansion_loans.append(
            [self.ts.current("target_consumption_expansion_loans").sum()]
        )

        # Mortgages
        target_house_price = np.zeros(self.ts.current("n_households"))
        if len(current_sales) > 0:
            target_house_price[current_sales["buyer_id"].values] = current_sales["price_or_rent"].values
        self.ts.target_mortgage.append(
            self.functions["target_credit"].compute_target_mortgage(
                target_house_price=target_house_price,
                target_consumption_before_ce=self.ts.current("target_consumption_before_ce"),
                income=self.ts.current("income"),
                rent=self.ts.current("rent"),
                wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
            )
        )
        self.ts.total_target_mortgage.append([self.ts.current("target_mortgage").sum()])

    def compute_interest_paid_on_deposits(
        self,
        bank_interest_rate_on_household_deposits: np.ndarray,
        bank_overdraft_rate_on_household_deposits: np.ndarray,
    ) -> np.ndarray:
        return -bank_interest_rate_on_household_deposits[self.states["Corresponding Bank ID"]] * np.maximum(
            0.0, self.ts.current("wealth_deposits")
        ) - bank_overdraft_rate_on_household_deposits[self.states["Corresponding Bank ID"]] * np.minimum(
            0.0, self.ts.current("wealth_deposits")
        )

    def compute_interest_paid(self) -> np.ndarray:
        return self.ts.current("interest_paid_on_loans") + self.ts.current("interest_paid_on_deposits")

    def prepare_goods_market_clearing(self, exchange_rate_usd_to_lcu: float) -> None:
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.ts.target_consumption.append(self.compute_target_consumption())
        self.prepare_buying_goods()
        self.prepare_selling_goods()

    def compute_target_consumption(self) -> np.ndarray:
        return self.functions["consumption"].compute_target_consumption(
            income=self.ts.current("income"),
            target_consumption_before_ce=self.ts.current("target_consumption_before_ce"),
            target_consumption_ce=self.ts.current("target_consumption_ce"),
            target_consumption_expansion_loans=self.ts.current("target_consumption_expansion_loans"),
            received_consumption_expansion_loans=self.ts.current("received_consumption_expansion_loans"),
            consumption_weights=self.consumption_weights,
            consumption_weights_by_income=self.consumption_weights_by_income,
            take_consumption_weights_by_income_quantile=self.use_consumption_weights_by_income,
        )

    def prepare_buying_goods(self) -> None:
        self.set_goods_to_buy(1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("target_consumption"))

    def prepare_selling_goods(self) -> None:
        self.set_goods_to_sell(np.zeros(self.ts.current("n_households")))
        self.set_prices(np.zeros(self.ts.current("n_households")))

    def update_consumption_and_investment(self, tau_vat: float) -> None:
        # Total amount spent
        self.ts.amount_bought.append(self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1))

        # Investment in other real assets
        self.ts.investment_in_other_real_assets.append(
            np.minimum(
                self.ts.current("amount_bought"),
                self.ts.current("target_consumption_ce")
                * np.divide(
                    self.ts.current("received_consumption_expansion_loans"),
                    self.ts.current("target_consumption_expansion_loans"),
                    out=np.ones_like(self.ts.current("received_consumption_expansion_loans")),
                    where=self.ts.current("target_consumption_expansion_loans") != 0.0,
                ),
            )
        )
        self.ts.total_investment_in_other_real_assets.append([self.ts.current("investment_in_other_real_assets").sum()])

        # Consumption
        self.ts.consumption.append(
            (1 + tau_vat) * self.ts.current("amount_bought") - self.ts.current("investment_in_other_real_assets")
        )
        self.ts.total_consumption.append([(1 + tau_vat) * self.ts.current("nominal_amount_spent_in_lcu").sum()])
        self.ts.industry_consumption.append(self.ts.current("nominal_amount_spent_in_lcu").sum(axis=0))

    def update_wealth(self, housing_data: pd.DataFrame, tau_cf: float) -> None:
        # Update real wealth
        self.ts.wealth_main_residence.append(
            self.compute_wealth_of_the_main_residence(
                housing_data=housing_data,
            )
        )
        self.ts.total_wealth_main_residence.append([self.ts.current("wealth_main_residence").sum()])
        self.ts.wealth_other_properties.append(
            self.compute_wealth_of_other_properties(
                housing_data=housing_data,
            )
        )
        self.ts.total_wealth_other_properties.append([self.ts.current("wealth_other_properties").sum()])
        self.ts.wealth_other_real_assets.append(self.compute_wealth_of_other_real_assets())
        self.ts.total_wealth_other_real_assets.append([self.ts.current("wealth_other_real_assets").sum()])
        self.ts.wealth_real_assets.append(
            self.ts.current("wealth_main_residence")
            + self.ts.current("wealth_other_properties")
            + self.ts.current("wealth_other_real_assets")
        )

        # New financial wealth
        new_wealth = np.maximum(
            0.0,
            (
                self.ts.current("income")
                - self.ts.current("rent")
                - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            ),
        )
        (new_wealth_in_deposits, new_wealth_in_other_financial_assets) = self.functions["wealth"].distribute_new_wealth(
            new_wealth=new_wealth,
            model=self.states["wealth_distribution_model"],
            ts=self.ts,
        )

        # Used-up financial wealth
        used_up_wealth = -np.minimum(
            0.0,
            (
                self.ts.current("income")
                - self.ts.current("rent")
                - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            ),
        )
        (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        ) = self.functions["wealth"].use_up_wealth(
            used_up_wealth=used_up_wealth,
            current_wealth_in_deposits=self.ts.current("wealth_deposits"),
            current_wealth_in_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

        # Update other financial assets
        self.ts.wealth_other_financial_assets.append(
            self.compute_wealth_of_other_financial_assets(
                new_wealth_in_other_financial_assets=new_wealth_in_other_financial_assets,
                used_up_wealth_in_other_financial_assets=used_up_wealth_in_other_financial_assets,
            )
        )
        self.ts.total_wealth_other_financial_assets.append([self.ts.current("wealth_other_financial_assets").sum()])

        # Update deposits
        self.ts.wealth_deposits.append(
            self.compute_wealth_in_deposits(
                new_wealth_in_deposits=new_wealth_in_deposits,
                used_up_wealth_in_deposits=used_up_wealth_in_deposits,
                tau_cf=tau_cf,
            )
        )
        self.ts.total_wealth_deposits.append([self.ts.current("wealth_deposits").sum()])

        # Compute total financial assets
        self.ts.wealth_financial_assets.append(
            self.ts.current("wealth_other_financial_assets") + self.ts.current("wealth_deposits")
        )

        # Compute total wealth
        self.ts.wealth.append(self.ts.current("wealth_real_assets") + self.ts.current("wealth_financial_assets"))

    def compute_wealth_of_the_main_residence(self, housing_data: pd.DataFrame) -> np.ndarray:
        wealth_of_the_main_residence = np.zeros(self.ts.current("n_households"))
        ind_owning_mhr = self.states["Tenure Status of the Main Residence"] == 1
        wealth_of_the_main_residence[ind_owning_mhr] = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_owning_mhr], "Value"
        ].values
        return wealth_of_the_main_residence

    def compute_wealth_of_other_properties(self, housing_data: pd.DataFrame) -> np.ndarray:
        wealth_of_other_properties = np.zeros(self.ts.current("n_households"))
        housing_data_not_oo = housing_data.loc[housing_data["Is Owner-Occupied"] == 0]
        housing_data_not_oo_grouped = housing_data_not_oo.groupby("Corresponding Owner Household ID")["Value"].sum()
        wealth_of_other_properties[housing_data_not_oo_grouped.index.values] = housing_data_not_oo_grouped.values
        return wealth_of_other_properties

    def compute_wealth_of_other_real_assets(self) -> np.ndarray:
        return self.functions["wealth"].compute_wealth_in_other_real_assets(
            current_wealth_in_other_real_assets=self.ts.current("wealth_other_real_assets"),
            current_investment_in_other_real_assets=self.ts.current("investment_in_other_real_assets"),
        )

    def compute_wealth_of_other_financial_assets(
        self,
        new_wealth_in_other_financial_assets: float,
        used_up_wealth_in_other_financial_assets: float,
    ) -> np.ndarray:
        return self.functions["wealth"].compute_wealth_in_other_financial_assets(
            current_wealth_in_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
            new_wealth_in_other_financial_assets=new_wealth_in_other_financial_assets,
            used_up_wealth_in_other_financial_assets=used_up_wealth_in_other_financial_assets,
        )

    def compute_wealth_in_deposits(
        self,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        return self.functions["wealth"].compute_wealth_in_deposits(
            current_wealth_in_deposits=self.ts.current("wealth_deposits"),
            new_wealth_in_deposits=new_wealth_in_deposits,
            used_up_wealth_in_deposits=used_up_wealth_in_deposits,
            current_interest_paid=self.ts.current("interest_paid"),
            price_paid_for_property=self.ts.current("price_paid_for_property"),
            debt_installments=self.ts.current("debt_installments"),
            new_loans=self.ts.current("received_payday_loans")
            + self.ts.current("received_consumption_expansion_loans")
            + self.ts.current("received_mortgages"),
            new_real_wealth=self.ts.current("wealth_real_assets") - self.ts.prev("wealth_real_assets"),
            tau_cf=tau_cf,
        )

    def compute_debt(self) -> np.ndarray:
        self.ts.total_payday_loan_debt.append([self.ts.current("payday_loan_debt").sum()])
        self.ts.total_consumption_expansion_loan_debt.append([self.ts.current("consumption_expansion_loan_debt").sum()])
        self.ts.total_mortgage_debt.append([self.ts.current("mortgage_debt").sum()])
        return (
            self.ts.current("payday_loan_debt")
            + self.ts.current("consumption_expansion_loan_debt")
            + self.ts.current("mortgage_debt")
        )

    def compute_net_wealth(self) -> np.ndarray:
        return self.ts.current("wealth") - self.ts.current("debt")

    def handle_insolvency(self, banks: Banks, credit_market: CreditMarket) -> float:
        return self.functions["insolvency"].handle_insolvency(
            households=self,
            banks=banks,
            credit_market=credit_market,
        )

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("households", group)

    def save_consumption_weights(self, group: h5py.Group):
        group.create_dataset("household_consumption_weights_by_income", data=self.consumption_weights.T)
        group["household_consumption_weights_by_income"].attrs["columns"] = list(range(self.n_industries))
