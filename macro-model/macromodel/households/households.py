import h5py
import numpy as np
import pandas as pd
import warnings
from macro_data import SyntheticPopulation, SyntheticCountry
from typing import Any, Tuple

from macromodel.configurations import HouseholdsConfiguration
from macromodel.agents.agent import Agent
from macromodel.banks.banks import Banks
from macromodel.credit_market.credit_market import CreditMarket
from macromodel.goods_market.value_type import ValueType
from macromodel.households.household_properties import HouseholdType
from macromodel.households.households_ts import create_households_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model
from macromodel.util.get_histogram import get_histogram
from macromodel.util.property_mapping import map_to_enum


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
        investment_weights: np.ndarray,
        use_consumption_weights_by_income: bool,
        independents: list[str],
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

        self.independents = independents

        # Set initial values
        self.ts["saving_rates_histogram"] = get_histogram(self.get_saving_rates_by_household(), None)

        self.consumption_weights = consumption_weights
        self.consumption_weights_by_income = consumption_weights_by_income

        self.investment_weights = investment_weights

        self.use_consumption_weights_by_income = use_consumption_weights_by_income

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_population: SyntheticPopulation,
        synthetic_country: SyntheticCountry,
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

        functions = functions_from_model(model=configuration.functions, loc="macromodel.households")

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

        investment_weights = synthetic_population.investment_weights

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray] | Any] = {
            "saving_rates_model": synthetic_population.saving_rates_model,
            "social_transfers_model": synthetic_population.social_transfers_model,
            "wealth_distribution_model": synthetic_population.wealth_distribution_model,
            "average_saving_rate": synthetic_population.household_data["Saving Rate"].mean(),
            "coefficient_fa_income": synthetic_population.coefficient_fa_income,
            "investment_rate": synthetic_population.household_data["Investment Rate"].values,
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

        # TODO: this is set to 0.2 in Sam's code, and transformed somehow into 0.0945. by the time the data is exported
        #  We need to 1. make this a parameters, 2. move this to the macro-data package.
        #  In general, we should think of where to put the piece of code below.

        investment_rate = synthetic_population.household_data["Investment Rate"].values
        # investment_weights = synthetic_country.industry_data["industry_vectors"]["Household Capital Inputs in LCU"]
        # investment_weights = investment_weights.values / investment_weights.values.sum()
        tau_cf = synthetic_country.tax_data.capital_formation_tax
        income = synthetic_population.household_data["Income"].values  # Income is different from Sam's

        initial_investment = pd.DataFrame(
            data=(1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T),
            index=pd.Index(range(len(synthetic_population.household_data))),
            columns=pd.Index(synthetic_country.industries, name="Industry"),
        )

        tau_vat = synthetic_country.tax_data.value_added_tax

        consumption_by_industry_hh = 1 / (1 + tau_vat) * synthetic_population.industry_consumption_before_vat

        ts = create_households_timeseries(
            data=hh_data,
            initial_consumption_by_industry=initial_consumption_by_industry,
            initial_hh_investment=initial_investment.values,
            initial_investment_by_industry=synthetic_population.investment,
            scale=scale,
            vat=value_added_tax,
            tau_cf=tau_cf,
        )

        # Update the household type
        states["Type"] = map_to_enum(states["Type"], HouseholdType)

        # Corresponding individuals
        states["corr_individuals"] = list(corr_individuals.values)

        # Number of adults individuals in the household
        states["Number of Adults"] = np.array(
            [
                np.sum(individual_ages[states["corr_individuals"][hh_id]] >= 18)
                for hh_id in range(ts.current("n_households"))
            ]
        )

        # Corresponding renters
        states["corr_renters"] = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_renters]

        use_consumption_weights_by_income = configuration.take_consumption_weights_by_income_quantile

        independents = configuration.functions.saving_rates.parameters["independents"]

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
            investment_weights,
            use_consumption_weights_by_income,
            independents,
        )

    def compute_employee_income(
        self,
        individual_income: np.ndarray,
        corr_households: np.ndarray,
    ) -> np.ndarray:
        return np.bincount(
            corr_households,
            weights=individual_income,
            minlength=self.ts.current("n_households"),
        )

    def compute_expected_social_transfer_income(
        self,
        total_other_social_transfers: float,
        cpi: float,
        expected_inflation: float,
    ) -> np.ndarray:
        inds = self.independents
        return (
            (1 + expected_inflation)
            * cpi
            * self.functions["social_transfers"].get_social_transfers(
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
        )

    def compute_social_transfer_income(
        self,
        total_other_social_transfers: float,
        cpi: float,
    ) -> np.ndarray:
        inds = self.independents
        return cpi * self.functions["social_transfers"].get_social_transfers(
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
            np.logical_and(
                housing_data["Is Owner-Occupied"] == 0,
                housing_data["Corresponding Inhabitant Household ID"] != -1,
            )
        ]
        housing_data_rented_out_grouped = housing_data_rented_out.groupby("Corresponding Owner Household ID")[
            "Rent"
        ].sum()
        rental_income = np.zeros(self.ts.current("n_households"))
        rental_income[housing_data_rented_out_grouped.index.values] = (
            1 - income_taxes
        ) * housing_data_rented_out_grouped.values
        return rental_income

    def compute_expected_income_from_financial_assets(self) -> np.ndarray:
        return self.functions["financial_assets"].compute_expected_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def compute_income_from_financial_assets(self) -> np.ndarray:
        return self.functions["financial_assets"].compute_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def compute_expected_income(self) -> np.ndarray:
        return (
            self.ts.current("expected_income_employee")
            + self.ts.current("expected_income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("expected_income_financial_assets")
        )

    def compute_income(self) -> np.ndarray:
        return (
            self.ts.current("income_employee")
            + self.ts.current("income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("income_financial_assets")
        )

    def get_saving_rates_by_household(self) -> np.ndarray:
        inds = self.independents
        if len(inds) > 0:
            current_independents = np.stack(
                [self.ts.current(ind.lower()) for ind in inds],
                axis=1,
            )
            initial_independents = np.stack(
                [self.ts.initial(ind.lower()) for ind in inds],
                axis=1,
            )
        else:
            current_independents = np.array([])
            initial_independents = np.array([])
        return self.functions["saving_rates"].get_saving_rates(
            n_households=self.ts.current("n_households"),
            average_saving_rate=self.states["average_saving_rate"],
            current_independents=current_independents,
            initial_independents=initial_independents,
            model=self.states["saving_rates_model"],
        )

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        exogenous_total_consumption: float,
        per_capita_unemployment_benefits: float,
        tau_vat: float,
        assume_zero_growth: bool,
    ) -> np.ndarray:
        saving_rates = self.get_saving_rates_by_household()
        self.ts.saving_rates_histogram.append(get_histogram(saving_rates, None))

        # Target consumption
        if assume_zero_growth:
            return np.outer(
                self.ts.initial("consumption"),
                self.states["consumption_weights_data"],
            ).astype(float)
        else:
            return self.functions["consumption"].compute_target_consumption(
                expected_inflation=expected_inflation,
                current_cpi=current_cpi,
                initial_cpi=initial_cpi,
                historic_consumption_sum=np.array(self.ts.historic("consumption")),
                saving_rates=saving_rates,
                income=self.ts.current("expected_income"),
                household_benefits=self.states["Number of Adults"] * per_capita_unemployment_benefits
                + self.ts.current("expected_income_social_transfers"),
                consumption_weights=self.consumption_weights,
                consumption_weights_by_income=self.consumption_weights_by_income,
                exogenous_total_consumption=exogenous_total_consumption,
                current_time=len(self.ts.historic("total_consumption")),
                take_consumption_weights_by_income_quantile=self.use_consumption_weights_by_income,
                tau_vat=tau_vat,
            )

    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        exogenous_total_investment: float,
        tau_cf: float,
        assume_zero_growth: bool,
    ) -> np.ndarray:
        if assume_zero_growth:
            return self.ts.initial("investment").astype(float)
        else:
            return self.functions["investment"].compute_target_investment(
                expected_inflation=expected_inflation,
                current_cpi=current_cpi,
                initial_cpi=initial_cpi,
                income=self.ts.current("expected_income"),
                exogenous_total_investment=exogenous_total_investment,
                current_time=len(self.ts.historic("total_investment")),
                investment_weights=self.investment_weights,
                investment_rate=self.states["investment_rate"],
                tau_cf=tau_cf,
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
        if len(housing_data) == 0:
            return

        # Households make decisions on their demand for properties
        (
            max_price_willing_to_pay,
            max_rent_willing_to_pay,
            households_hoping_to_move,
        ) = self.functions["property"].compute_demand(
            housing_data=housing_data,
            household_residence_tenure_status=self.states["Tenure Status of the Main Residence"],
            household_income=self.ts.current("expected_income"),
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
        ind_mhr_temp_sale = housing_data["Corresponding Owner Household ID"].isin(households_hoping_to_move)
        housing_data.loc[np.logical_not(ind_mhr_temp_sale), "Sale Price"] = np.nan
        ind_still_on_sale = housing_data["Temporarily for Sale"].copy()
        housing_data["Temporarily for Sale"] = False
        housing_data.loc[ind_mhr_temp_sale, "Temporarily for Sale"] = True
        housing_data.loc[
            np.logical_and(ind_mhr_temp_sale, np.logical_not(ind_still_on_sale)),
            "Sale Price",
        ] = self.functions["property"].compute_initial_sale_price(
            property_values=housing_data.loc[
                np.logical_and(ind_mhr_temp_sale, np.logical_not(ind_still_on_sale)),
                "Value",
            ],
        )
        housing_data.loc[np.logical_and(ind_mhr_temp_sale, ind_still_on_sale), "Sale Price"] = self.functions[
            "property"
        ].compute_updated_sale_price(
            sale_prices=housing_data.loc[
                np.logical_and(ind_mhr_temp_sale, ind_still_on_sale),
                "Sale Price",
            ],
        )

        # Set what's up for rent
        prev_up_for_rent = housing_data["Up for Rent"].values
        now_up_for_rent = np.where(np.isnan(housing_data["Corresponding Inhabitant Household ID"].values))[0]
        newly_up_for_rent = [ind for ind in now_up_for_rent if ind not in prev_up_for_rent]
        housing_data["Up for Rent"] = False
        housing_data.loc[now_up_for_rent, "Up for Rent"] = True
        housing_data["Newly on the Rental Market"] = False
        housing_data.loc[newly_up_for_rent, "Newly on the Rental Market"] = True
        not_newly_up_for_rent = np.logical_and(
            np.logical_not(housing_data["Newly on the Rental Market"]),
            housing_data["Up for Rent"],
        )

        # Calculate rent
        housing_data.loc[housing_data["Newly on the Rental Market"], "Rent"] = self.functions[
            "property"
        ].compute_offered_rent_for_new_properties(
            property_value=housing_data.loc[housing_data["Newly on the Rental Market"], "Value"].values,
            observed_fraction_rent_value=observed_fraction_rent_value,
        )
        housing_data.loc[not_newly_up_for_rent, "Rent"] = self.functions[
            "property"
        ].compute_offered_rent_for_existing_properties(
            current_offered_rent=housing_data.loc[not_newly_up_for_rent, "Rent"].values,
        )

    def update_rent(
        self,
        housing_data: pd.DataFrame,
        historic_inflation: list[np.ndarray],
        exogenous_inflation_before: np.ndarray,
    ) -> None:
        housing_data["Rent"] = self.functions["property"].compute_rent(
            current_rent=housing_data["Rent"].values,
            historic_inflation=np.concatenate(
                (
                    exogenous_inflation_before,
                    np.array(historic_inflation).flatten(),
                )
            ),
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
        # Target consumption loans to cover immediate financing gaps
        self.ts.target_consumption_loans.append(
            self.functions["target_credit"].compute_target_consumption_loans(
                target_consumption=self.ts.current("target_consumption"),
                income=self.ts.current("expected_income"),
                rent=self.ts.current("rent"),
                wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
            )
        )
        self.ts.total_target_consumption_loans.append([self.ts.current("target_consumption_loans").sum()])

        # Mortgages
        target_house_price = np.zeros(self.ts.current("n_households"))
        if len(current_sales) > 0:
            target_house_price[current_sales["buyer_id"].values] = current_sales["price_or_rent"].values
        self.ts.target_mortgage.append(
            self.functions["target_credit"].compute_target_mortgage(
                target_house_price=target_house_price,
                target_consumption=self.ts.current("target_consumption"),
                income=self.ts.current("expected_income"),
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

    def prepare_goods_market_clearing(
        self,
        exchange_rate_usd_to_lcu: float,
    ) -> None:
        # Exchange rates
        self.set_exchange_rate(exchange_rate_usd_to_lcu)

        # Prepare goods market clearing
        self.prepare_buying_goods()
        self.prepare_selling_goods()

    def prepare_buying_goods(self) -> None:
        self.set_goods_to_buy(
            1.0
            / self.exchange_rate_usd_to_lcu
            * (self.ts.current("target_consumption") + self.ts.current("target_investment"))
        )

    def prepare_selling_goods(self) -> None:
        self.set_goods_to_sell(np.zeros(self.ts.current("n_households")))
        self.set_prices(np.zeros(self.ts.current("n_households")))

    def update_consumption_and_investment(self, tau_vat: float, tau_cf: float) -> None:
        # Total amount spent
        self.ts.amount_bought.append(self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1))

        # Distribute
        consumption_by_good = np.minimum(
            self.ts.current("nominal_amount_spent_in_lcu"),
            self.ts.current("target_consumption"),
        )

        # Consumption
        self.ts.consumption.append(consumption_by_good.sum(axis=1))
        self.ts.total_consumption.append([(1 + tau_vat) * self.ts.current("consumption").sum()])
        self.ts.total_consumption_before_vat.append([self.ts.current("consumption").sum()])
        self.ts.industry_consumption.append(consumption_by_good.sum(axis=0))

        # Investment
        self.ts.investment.append(self.ts.current("nominal_amount_spent_in_lcu") - consumption_by_good)
        self.ts.total_investment.append([(1 + tau_cf) * self.ts.current("investment").sum()])
        self.ts.total_investment_before_vat.append([self.ts.current("investment").sum()])
        self.ts.industry_investment.append(self.ts.current("investment").sum(axis=0))

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
        (
            new_wealth_in_deposits,
            new_wealth_in_other_financial_assets,
        ) = self.functions["wealth"].distribute_new_wealth(
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
            self.states["Corresponding Inhabited House ID"][ind_owning_mhr],
            "Value",
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
            current_investment_in_other_real_assets=self.ts.current("investment").sum(axis=1),
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
            new_loans=self.ts.current("received_consumption_loans") + self.ts.current("received_mortgages"),
            new_real_wealth=self.ts.current("investment").sum(axis=1),
            tau_cf=tau_cf,
        )

    def compute_debt(self) -> np.ndarray:
        self.ts.total_consumption_loan_debt.append([self.ts.current("consumption_loan_debt").sum()])
        self.ts.total_mortgage_debt.append([self.ts.current("mortgage_debt").sum()])
        return self.ts.current("consumption_loan_debt") + self.ts.current("mortgage_debt")

    def compute_net_wealth(self) -> np.ndarray:
        return self.ts.current("wealth") - self.ts.current("debt")

    def handle_insolvency(self, banks: Banks, credit_market: CreditMarket) -> Tuple[float, float, float]:
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

    def total_consumption(self) -> np.ndarray:
        return self.ts.get_aggregate("total_consumption")

    def consumption_loan_debt(self) -> np.ndarray:
        return self.ts.get_aggregate("consumption_loan_debt")

    def mortgage_debt(self) -> np.ndarray:
        return self.ts.get_aggregate("mortgage_debt")
