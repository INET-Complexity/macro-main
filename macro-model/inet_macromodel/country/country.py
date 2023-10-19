import logging

import numpy as np

from inet_macromodel.banks.banks import Banks
from inet_macromodel.central_bank.central_bank import CentralBank
from inet_macromodel.central_government.central_government import CentralGovernment
from inet_macromodel.credit_market.credit_market import CreditMarket
from inet_macromodel.housing_market.housing_market import HousingMarket
from inet_macromodel.economy.economy import Economy
from inet_macromodel.exogenous.exogenous import Exogenous
from inet_macromodel.firms.firms import Firms
from inet_macromodel.government_entities.government_entities import GovernmentEntities
from inet_macromodel.households.households import Households
from inet_macromodel.individuals.individuals import Individuals
from inet_macromodel.labour_market.labour_market import LabourMarket

from inet_macromodel.util.get_histogram import get_histogram


class Country:
    def __init__(
        self,
        country_name: str,
        year: int,
        t_max: int,
        scale: int,
        individuals: Individuals,
        households: Households,
        firms: Firms,
        central_government: CentralGovernment,
        government_entities: GovernmentEntities,
        banks: Banks,
        central_bank: CentralBank,
        economy: Economy,
        labour_market: LabourMarket,
        credit_market: CreditMarket,
        housing_market: HousingMarket,
        exogenous: Exogenous,
    ):
        # Parameters
        self.country_name = country_name
        self.year = year
        self.t_max = t_max
        self.scale = scale

        # Agents
        self.individuals = individuals
        self.households = households
        self.firms = firms
        self.central_government = central_government
        self.government_entities = government_entities
        self.banks = banks
        self.central_bank = central_bank

        # The economy
        self.economy = economy

        # Markets
        self.labour_market = labour_market
        self.credit_market = credit_market
        self.housing_market = housing_market

        # Exchange rate
        self.exchange_rate_usd_to_lcu = None

        # Exogenous data
        self.exogenous = exogenous

    def initialisation_phase(self, exchange_rate_usd_to_lcu: float) -> None:
        self.exchange_rate_usd_to_lcu = exchange_rate_usd_to_lcu
        self.firms.update_number_of_firms()

    def estimation_phase(self) -> None:
        self.economy.set_estimates(
            exogenous_log_inflation=self.exogenous.log_inflation_before,
            exogenous_sectoral_growth=self.exogenous.sectoral_growth_before,
            exogenous_hpi_growth=self.exogenous.house_price_index_before,
        )
        self.economy.compute_sectoral_sentiment()
        self.firms.set_estimates(
            previous_average_good_prices=self.economy.ts.current("good_prices"),
            current_estimated_sectoral_growth=self.economy.ts.current("estimated_sectoral_growth"),
        )

    def target_setting_phase(self) -> None:
        # Firms set production targets
        self.firms.set_targets(sectoral_sentiment=self.economy.ts.current("sectoral_sentiment"))

        # Firms determine the wages they're willing to pay new employees
        self.firms.states["offered_wage_function"] = self.firms.compute_offered_wage_function(
            current_individual_labour_inputs=self.individuals.ts.current("labour_inputs"),
            previous_employee_income=self.individuals.ts.current("employee_income"),
            unemployment_benefits_by_individual=self.central_government.ts.current(
                "unemployment_benefits_by_individual"
            )[0],
        )

        # Individuals set reservation wages
        self.individuals.ts.reservation_wages.append(
            self.individuals.compute_reservation_wages(
                unemployment_benefits_by_individual=self.central_government.ts.current(
                    "unemployment_benefits_by_individual"
                )[0],
            )
        )

    def clear_labour_market(self) -> None:
        logging.info("Clearing labour market for %s", self.country_name)
        labour_costs = self.labour_market.clear(
            firms=self.firms,
            households=self.households,
            individuals=self.individuals,
        )
        self.firms.ts.labour_costs.append(labour_costs)

    def update_planning_metrics(self) -> None:
        # The central government updates unemployment benefits paid to individuals and social transfers to households
        self.central_government.update_benefits(
            historic_cpi_inflation=self.economy.ts.historic("cpi_inflation"),
            exogenous_cpi_inflation=self.exogenous.log_inflation_before["Real CPI Inflation"].values,
            current_unemployment_rate=self.economy.ts.current("unemployment_rate")[0],
        )

        # Individuals update their income from unemployment benefits
        self.individuals.ts.income_from_unemployment_benefits.append(
            self.central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=self.individuals.states["Activity Status"],
            )
        )

        # Individual labour inputs
        self.individuals.ts.labour_inputs.append(self.individuals.compute_labour_inputs())

        # Central bank policy rate
        self.central_bank.ts.policy_rate.append([self.central_bank.compute_rate()])

        # Firm labour inputs
        self.firms.ts.labour_inputs.append(
            self.firms.compute_labour_inputs(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
                current_labour_inputs=self.individuals.ts.current("labour_inputs"),
            )
        )

        # Number of employees for each firm
        self.firms.ts.number_of_employees.append(
            self.firms.compute_n_employees(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            )
        )
        self.firms.ts.number_of_employees_histogram.append(
            get_histogram(self.firms.ts.current("number_of_employees"), None)
        )
        self.firms.ts.output_by_employee_histogram.append(
            get_histogram(self.firms.ts.current("production") / self.firms.ts.current("number_of_employees"), None)
        )

        # Firm wages
        self.individuals.ts.employee_income.append(
            self.firms.set_wages(
                current_individual_labour_inputs=self.individuals.ts.current("labour_inputs"),
                previous_employee_income=self.individuals.ts.current("employee_income"),
            )
        )
        self.individuals.ts.employee_income_histogram.append(
            get_histogram(self.individuals.ts.current("employee_income"), self.scale)
        )
        self.firms.ts.total_wage.append(
            self.firms.compute_total_wages_paid(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
                individual_wages=self.individuals.ts.current("employee_income"),
                income_taxes=self.central_government.states["Income Tax"],
                employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
                employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            )
        )

        # Firm production
        self.firms.ts.production.append(self.firms.compute_production())
        self.firms.ts.production_histogram.append(get_histogram(self.firms.ts.current("production"), None))

        # Firm prices
        self.firms.ts.price.append(
            self.firms.compute_price(
                current_estimated_ppi_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
                previous_average_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Firm demand for goods
        self.firms.ts.unconstrained_target_intermediate_inputs.append(
            self.firms.compute_unconstrained_demand_for_intermediate_inputs()
        )
        self.firms.ts.unconstrained_target_intermediate_inputs_costs.append(
            self.firms.compute_unconstrained_demand_for_intermediate_inputs_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.unconstrained_target_capital_inputs.append(
            self.firms.compute_unconstrained_demand_for_capital_inputs()
        )
        self.firms.ts.unconstrained_target_capital_inputs_costs.append(
            self.firms.compute_unconstrained_demand_for_capital_inputs_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Firm intermediate inputs
        self.firms.ts.used_intermediate_inputs.append(self.firms.compute_used_intermediate_inputs())
        self.firms.ts.used_intermediate_inputs_costs.append(
            self.firms.compute_used_intermediate_inputs_costs(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Firm capital inputs
        self.firms.ts.used_capital_inputs.append(self.firms.compute_used_capital_inputs())
        self.firms.ts.used_capital_inputs_costs.append(
            self.firms.compute_used_capital_inputs_costs(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Firm unit costs
        self.firms.ts.unit_costs.append(self.firms.compute_unit_costs())

        # Individual income
        self.individuals.ts.income.append(self.individuals.compute_income())
        self.individuals.ts.income_histogram.append(get_histogram(self.individuals.ts.current("income"), self.scale))

        # Household income
        self.households.ts.income_employee.append(
            self.households.compute_employee_income(
                individual_income=self.individuals.ts.current("income"),
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
        )
        self.households.ts.total_income_employee.append([self.households.ts.current("income_employee").sum()])
        self.households.ts.income_social_transfers.append(
            self.households.compute_social_transfer_income(
                total_other_social_transfers=self.central_government.ts.current("total_other_benefits")[0],
                central_government_init=self.central_government.parameters,
            )
        )
        self.households.ts.total_income_social_transfers.append(
            [self.households.ts.current("income_social_transfers").sum()]
        )
        self.households.ts.income_rental.append(
            self.households.compute_rental_income(
                housing_data=self.housing_market.states,
                income_taxes=self.central_government.states["Income Tax"],
            )
        )
        self.households.ts.total_income_rental.append([self.households.ts.current("income_rental").sum()])
        self.households.ts.income_financial_assets.append(self.households.compute_income_from_financial_assets())
        self.households.ts.total_income_financial_assets.append(
            [self.households.ts.current("income_financial_assets").sum()]
        )
        self.households.ts.income.append(self.households.compute_income())
        self.households.ts.income_histogram.append(get_histogram(self.households.ts.current("income"), self.scale))
        rent_div_income = np.divide(
            self.households.ts.current("rent"),
            self.households.ts.current("income"),
            out=np.zeros_like(self.households.ts.current("rent")),
            where=self.households.ts.current("income") != 0.0,
        )
        self.households.ts.rent_div_income_histogram.append(get_histogram(rent_div_income, None))

        # Household target consumption before any consumption expansion
        self.households.ts.target_consumption_before_ce.append(
            self.households.compute_target_consumption_before_ce(
                per_capita_unemployment_benefits=self.central_government.ts.current(
                    "unemployment_benefits_by_individual"
                )[0],
                tau_vat=self.central_government.states["Value-added Tax"],
            )
        )

    def prepare_housing_market_clearing(self) -> None:
        # Update property values
        self.housing_market.update_property_value()

        # Decide on whether to remain, rent or buy
        self.households.prepare_housing_market_clearing(
            housing_data=self.housing_market.states,
            observed_fraction_value_price=self.housing_market.ts.current("observed_fraction_value_price"),
            observed_fraction_rent_value=self.housing_market.ts.current("observed_fraction_rent_value"),
            expected_hpi_growth=self.economy.ts.current("estimated_nominal_house_price_index_growth")[0],
            assumed_mortgage_maturity=self.banks.parameters["mortgage_maturity"]["value"],
            rental_income_taxes=self.central_government.states["Income Tax"],
        )

        # Set rent
        self.households.update_rent(
            housing_data=self.housing_market.states,
            historic_inflation=self.economy.ts.historic("cpi_inflation"),
            exogenous_inflation_before=self.exogenous.log_inflation_before["Real CPI Inflation"].values,
        )

    def clear_housing_market(self) -> None:
        self.housing_market.clear(
            household_main_residence_tenure_status=self.households.states["Tenure Status of the Main Residence"],
            max_price_willing_to_pay=self.households.ts.current("max_price_willing_to_pay"),
            max_rent_willing_to_pay=self.households.ts.current("max_rent_willing_to_pay"),
        )

    def prepare_credit_market_clearing(self) -> None:
        self.firms.compute_target_credit()
        self.households.compute_target_credit(
            current_sales=self.housing_market.current_sales.loc[
                self.housing_market.current_sales["sales_types"] == "Rental"
            ],
        )
        self.banks.set_interest_rates(
            central_bank_policy_rate=self.central_bank.ts.current("policy_rate")[0],
        )

    def clear_credit_market(self) -> None:
        self.credit_market.clear(
            banks=self.banks,
            firms=self.firms,
            households=self.households,
        )

    def process_housing_market_clearing(self) -> None:
        self.housing_market.ts.observed_fraction_value_price.append(
            self.housing_market.compute_observed_fraction_value_price()
        )
        self.housing_market.ts.observed_fraction_rent_value.append(
            self.housing_market.compute_observed_fraction_rent_value()
        )
        self.housing_market.process_housing_market_clearing(
            household_states=self.households.states,
            household_received_mortgages=self.households.ts.current("received_mortgages"),
            household_financial_wealth=self.households.ts.current("wealth_financial_assets"),
        )

        self.households.process_housing_market_clearing(
            housing_data=self.housing_market.states,
            social_housing_function=self.central_government.functions["social_housing"],
            current_sales=self.housing_market.current_sales.loc[
                self.housing_market.current_sales["sales_types"] == "Sell"
            ],
            current_unemployment_benefits_by_individual=self.central_government.ts.current(
                "unemployment_benefits_by_individual"
            )[0],
        )

    def process_credit_market_clearing(self) -> None:
        # Handle debt installments
        self.firms.ts.debt_installments.append(
            self.credit_market.pay_firm_installments(n_firms=self.firms.ts.current("n_firms"))
        )
        self.firms.ts.total_debt_installments.append([self.firms.ts.current("debt_installments").sum()])
        self.households.ts.debt_installments.append(
            self.credit_market.pay_household_installments(n_households=self.households.ts.current("n_households"))
        )
        self.households.ts.total_debt_installments.append([self.households.ts.current("debt_installments").sum()])
        self.credit_market.remove_repaid_loans()

        # Compute aggregates
        self.credit_market.compute_aggregates()

        # Calculate firm debt
        self.firms.ts.short_term_loan_debt.append(
            self.credit_market.compute_outstanding_short_term_loans_by_firm(n_firms=self.firms.ts.current("n_firms"))
        )
        self.firms.ts.long_term_loan_debt.append(
            self.credit_market.compute_outstanding_short_term_loans_by_firm(n_firms=self.firms.ts.current("n_firms"))
        )
        self.firms.ts.debt.append(self.firms.compute_debt())

        # Calculate the interest on loans paid by firms
        self.firms.ts.interest_paid_on_loans.append(
            self.credit_market.compute_interest_paid_by_firm(n_firms=self.firms.ts.current("n_firms"))
        )

        # Calculate the interest on deposits received/paid by firms
        self.firms.ts.interest_paid_on_deposits.append(
            self.firms.compute_interest_paid_on_deposits(
                bank_interest_rate_on_firm_deposits=self.banks.ts.current("interest_rate_on_firm_deposits"),
                bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
            )
        )

        # Calculate paid interest of firms
        self.firms.ts.interest_paid.append(self.firms.compute_interest_paid())

        # Calculate household debt
        self.households.ts.payday_loan_debt.append(
            self.credit_market.compute_outstanding_payday_loans_by_household(
                n_households=self.households.ts.current("n_households")
            )
        )
        self.households.ts.consumption_expansion_loan_debt.append(
            self.credit_market.compute_outstanding_consumption_expansion_loans_by_household(
                n_households=self.households.ts.current("n_households")
            )
        )
        self.households.ts.mortgage_debt.append(
            self.credit_market.compute_outstanding_mortgages_by_household(
                n_households=self.households.ts.current("n_households")
            )
        )
        self.households.ts.debt.append(self.households.compute_debt())
        self.households.ts.debt_histogram.append(get_histogram(self.households.ts.current("debt"), self.scale))

        # Calculate the interest on loans paid by households
        self.households.ts.interest_paid_on_loans.append(
            self.credit_market.compute_interest_paid_by_household(
                n_households=self.households.ts.current("n_households")
            )
        )

        # Calculate the interest on deposits received/paid by households
        self.households.ts.interest_paid_on_deposits.append(
            self.households.compute_interest_paid_on_deposits(
                bank_interest_rate_on_household_deposits=self.banks.ts.current("interest_rate_on_household_deposits"),
                bank_overdraft_rate_on_household_deposits=self.banks.ts.current("overdraft_rate_on_household_deposits"),
            )
        )

        # Calculate paid interest of households
        self.households.ts.interest_paid.append(self.households.compute_interest_paid())

        # Calculate the interest on loans received by banks
        self.banks.ts.interest_received_on_loans.append(
            self.credit_market.compute_interest_received_by_bank(n_banks=self.banks.ts.current("n_banks"))
        )

    def prepare_goods_market_clearing(self) -> None:
        self.firms.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
        )
        self.households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
        )
        self.government_entities.prepare_goods_market_clearing(
            n_industries=self.economy.n_industries,
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
        )

    def update_realised_metrics(self) -> None:
        # Firms distribute bought goods
        self.firms.distribute_bought_goods()

        # Economic indicators
        self.economy.compute_price_indicators(
            firm_real_amount_bought=self.firms.ts.current("real_amount_bought"),
            firm_nominal_amount_spent=self.firms.ts.current("nominal_amount_spent_in_lcu"),
            household_real_amount_bought=self.households.ts.current("real_amount_bought"),
            household_nominal_amount_spent=self.households.ts.current("nominal_amount_spent_in_lcu"),
            government_real_amount_bought=self.government_entities.ts.current("real_amount_bought"),
            government_nominal_amount_spent=self.government_entities.ts.current("nominal_amount_spent_in_lcu"),
            firms_real_amount_bought_as_capital_goods=self.firms.ts.current("real_amount_bought_as_capital_goods"),
        )
        self.economy.compute_inflation()
        self.economy.compute_growth(
            current_production=self.firms.ts.current("production"),
            prev_production=self.firms.ts.prev("production"),
            industries=self.firms.states["Industry"],
        )
        self.economy.compute_house_price_index(
            current_property_values=self.housing_market.ts.current("property_values"),
            previous_property_values=self.housing_market.ts.prev("property_values"),
        )
        self.economy.compute_labour_market_aggregates(
            current_individual_activity_status=self.individuals.states["Activity Status"],
            current_firm_labour_inputs=self.firms.ts.current("labour_inputs"),
            current_desired_firm_labour_inputs=self.firms.ts.current("desired_labour_inputs"),
            num_ind_employed_before_cleaning=self.labour_market.ts.current("num_employed_individuals_before_clearing")[
                0
            ],
            num_ind_newly_joining=self.labour_market.ts.current("num_individuals_newly_joining")[0],
            num_ind_newly_leaving=self.labour_market.ts.current("num_individuals_newly_leaving")[0],
        )
        self.economy.compute_rental_market_aggregates(
            real_rent_paid=self.households.ts.current("rent"),
            imp_rent_paid=self.households.ts.current("rent_imputed"),
            rental_income=self.households.ts.current("income_rental"),
        )

        # Global trade
        self.economy.record_global_trade(
            firms=self.firms,
            households=self.households,
            government_entities=self.government_entities,
            tau_export=self.central_government.states["Export Tax"],
        )

        # Gross fixed capital formation
        self.firms.ts.gross_fixed_capital_formation.append(
            self.firms.compute_gross_fixed_capital_formation(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Amount spent on intermediate inputs and capital inputs
        self.firms.update_total_newly_bought_costs(
            current_good_prices=self.economy.ts.current("good_prices"),
        )

        # Firm demand
        self.firms.ts.demand.append(self.firms.compute_demand())

        # Firm nominal sales
        self.firms.ts.production_nominal.append(
            self.firms.compute_nominal_production(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Firm inventory and stocks
        self.firms.ts.inventory.append(self.firms.compute_inventory())
        self.firms.ts.inventory_nominal.append(
            self.firms.compute_nominal_inventory(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.intermediate_inputs_stock.append(self.firms.compute_intermediate_inputs_stock())
        self.firms.ts.intermediate_inputs_stock_value.append(
            self.firms.compute_intermediate_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.intermediate_inputs_stock_industry.append(
            self.firms.ts.current("intermediate_inputs_stock").sum(axis=0)
        )
        self.firms.ts.capital_inputs_stock.append(self.firms.compute_capital_inputs_stock())
        self.firms.ts.capital_inputs_stock_value.append(
            self.firms.compute_intermediate_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.capital_inputs_stock_industry.append(self.firms.ts.current("capital_inputs_stock").sum(axis=0))

        # Firm total changes in inventories
        self.firms.ts.total_inventory_change.append(self.firms.compute_total_inventory_change())

        # Firm taxes paid on production
        self.firms.ts.taxes_paid_on_production.append(
            self.firms.compute_taxes_paid_on_production(
                taxes_less_subsidies_rates=self.central_government.states["Taxes Less Subsidies Rates"],
            )
        )

        # Firm total sales
        self.firms.ts.total_sales.append(self.firms.compute_total_sales())

        # Firm profits
        self.firms.ts.profits.append(self.firms.compute_profits())

        # Firm corporate tax payments
        self.firms.ts.corporate_taxes_paid.append(
            self.firms.compute_corporate_taxes_paid(
                tau_firm=self.central_government.states["Profit Tax"],
            )
        )

        # Firm deposits
        self.firms.ts.deposits.append(self.firms.compute_deposits())

        # Firm gross operating surplus and mixed income
        self.firms.ts.gross_operating_surplus_mixed_income.append(
            self.firms.compute_gross_operating_surplus_mixed_income()
        )

        # Handle firm insolvency
        self.firms.handle_insolvency(credit_market=self.credit_market)
        firm_insolvency_rate, num_insolvent_firms_by_sector = self.firms.compute_insolvency_rate()
        self.economy.ts.firm_insolvency_rate.append([firm_insolvency_rate])
        self.economy.ts.num_insolvent_firms_by_sector.append(num_insolvent_firms_by_sector)

        # Compute firm equity
        self.firms.ts.equity.append(
            self.firms.compute_equity(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Firm aggregates for debt and deposits
        self.firms.ts.total_debt.append([self.firms.compute_total_debt()])
        self.firms.ts.total_deposits.append([self.firms.compute_total_deposits()])

        # Household consumption, investment, wealth, and debt
        self.households.update_consumption_and_investment(tau_vat=self.central_government.states["Value-added Tax"])
        self.households.update_wealth(
            housing_data=self.housing_market.states,
            tau_cf=self.central_government.states["Capital Formation Tax"],
        )
        self.households.ts.wealth_histogram.append(get_histogram(self.households.ts.current("wealth"), self.scale))
        self.households.ts.net_wealth.append(self.households.compute_net_wealth())
        self.economy.ts.household_insolvency_rate.append(
            [
                self.households.handle_insolvency(
                    banks=self.banks,
                    credit_market=self.credit_market,
                )
            ]
        )

        # Total government entity consumption
        self.government_entities.record_consumption()

        # Calculate the interest on deposits received/paid by banks
        self.banks.ts.interest_received_on_deposits.append(
            self.banks.compute_interest_received_on_deposits(
                central_bank_policy_rate=self.central_bank.ts.current("policy_rate"),
            )
        )

        # Calculate bank profits
        self.banks.ts.profits.append(self.banks.compute_profits())
        self.banks.ts.profits_histogram.append(get_histogram(self.banks.ts.current("profits"), self.scale))

        # Record current bank deposits and loans from firms and households
        self.banks.update_deposits(
            current_firm_deposits=self.firms.ts.current("deposits"),
            current_household_deposits=self.households.ts.current("wealth_deposits"),
            firm_corresponding_bank=self.firms.states["Corresponding Bank ID"],
            households_corresponding_bank=self.households.states["Corresponding Bank ID"],
        )
        self.banks.update_loans(credit_market=self.credit_market)

        # Bank market share
        self.banks.ts.market_share.append(self.banks.compute_market_share())
        self.banks.ts.market_share_histogram.append(get_histogram(self.banks.ts.current("market_share"), None))

        # Bank equity
        self.banks.ts.equity.append(
            self.banks.compute_equity(
                profit_taxes=self.central_government.states["Profit Tax"],
            )
        )
        self.banks.ts.equity_histogram.append(get_histogram(self.banks.ts.current("equity"), self.scale))

        # Bank liability
        self.banks.ts.liability.append(self.banks.compute_liability())
        self.banks.ts.liability_histogram.append(get_histogram(self.banks.ts.current("liability"), self.scale))

        # Bank deposits
        self.banks.ts.deposits.append(self.banks.compute_deposits())
        self.banks.ts.deposits_histogram.append(get_histogram(self.banks.ts.current("deposits"), self.scale))

        # Handle bank insolvency
        self.central_government.ts.bank_equity_injection.append(
            [self.banks.handle_insolvency(credit_market=self.credit_market)]
        )
        self.economy.ts.bank_insolvency_rate.append([self.banks.compute_insolvency_rate()])

        # Compute taxes collected by the central government
        self.central_government.compute_taxes(
            current_ind_employee_income=self.individuals.ts.current("employee_income"),
            current_total_rent_paid=self.households.ts.current("rent"),
            current_income_financial_assets=self.households.ts.current("income_financial_assets"),
            current_ind_activity=self.individuals.states["Activity Status"],
            current_ind_realised_cons=self.households.ts.current("nominal_amount_spent_in_lcu").sum(axis=1),
            current_bank_profits=self.banks.ts.current("profits"),
            current_firm_production=self.firms.ts.current("production"),
            current_firm_price=self.firms.ts.current("price"),
            current_firm_profits=self.firms.ts.current("profits"),
            current_firm_industries=self.firms.states["Industry"],
            taxes_less_subsidies_rates=self.central_government.states["Taxes Less Subsidies Rates"],
            current_household_new_real_wealth=self.households.ts.current("wealth_real_assets")
            - self.households.ts.prev("wealth_real_assets"),
            current_total_exports=self.economy.ts.current("exports_before_taxes").sum(),
        )

        # General government fields
        self.central_government.ts.taxes_on_products.append([self.central_government.compute_taxes_on_products()])
        self.central_government.ts.revenue.append(
            [
                self.central_government.compute_revenue(
                    household_rent_paid_to_government=self.households.states["Rent paid to Government"]
                )
            ]
        )
        self.central_government.ts.deficit.append(
            self.central_government.compute_deficit(
                current_ind_activity=self.individuals.states["Activity Status"],
                current_household_social_transfers=self.households.ts.current("income_social_transfers"),
                current_government_nominal_amount_spent=self.government_entities.ts.current(
                    "nominal_amount_spent_in_lcu"
                ),
                government_interest_rates=self.banks.ts.current("interest_rate_on_government_debt")[0],
            )
        )
        self.central_government.ts.debt.append(self.central_government.compute_debt())

        # Compute GDP
        self.economy.compute_gdp(
            sales_minus_ii=self.firms.ts.current("total_sales").sum()
            - self.firms.ts.current("used_intermediate_inputs_costs").sum(),
            taxes_on_products=self.central_government.ts.current("taxes_on_products")[0],
            rent_paid=self.economy.ts.current("total_real_rent_paid")[0],
            rent_imputed=self.economy.ts.current("total_imp_rent_paid")[0],
            hh_consumption=self.households.ts.current("total_consumption")[0],
            gov_consumption=self.government_entities.ts.current("total_consumption")[0],
            change_in_firm_stock_inventories=self.firms.ts.current("total_inventory_change").sum()
            + self.firms.ts.current("total_intermediate_inputs_bought_costs").sum()
            - self.firms.ts.current("used_intermediate_inputs_costs").sum()
            + self.firms.ts.current("total_capital_inputs_bought_costs").sum(),
            exports_minus_imports=self.economy.ts.current("exports").sum() - self.economy.ts.current("imports").sum(),
            operating_surplus_plus_wages=self.firms.ts.current("gross_operating_surplus_mixed_income").sum()
            + self.firms.ts.current("total_wage").sum(),
            rent_received=self.economy.ts.current("total_real_rent_rec")[0]
            + self.central_government.ts.current("total_rent_received")[0]
            + self.central_government.ts.current("taxes_rental_income")[0],
        )

    def update_population_structure(self) -> None:
        self.individuals.update_demography()
