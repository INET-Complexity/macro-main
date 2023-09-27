import numpy as np
import pandas as pd

from scipy import special
from functools import reduce

from data.processing.synthetic_firms.synthetic_firms import (
    SyntheticFirms,
)

from data.readers.economic_data.ons_reader import ONSReader
from data.readers.economic_data.oecd_economic_data import OECDEconData
from data.readers.economic_data.exchange_rates import WorldBankRatesReader


class SyntheticDefaultFirms(SyntheticFirms):
    def __init__(
        self,
        country_name: str,
        scale: int,
        year: int,
        industries: list[str],
    ):
        super().__init__(
            country_name,
            scale,
            year,
            industries,
        )

    def create(
        self,
        econ_reader: OECDEconData,
        ons_reader: ONSReader,
        exchange_rates: WorldBankRatesReader,
        total_firm_deposits: float,
        total_firm_debt: float,
        industry_data: dict[str, pd.DataFrame],
        number_of_employees_by_industry: np.ndarray,
        initial_inventory_to_production_fraction: float,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        assume_zero_initial_deposits: bool,
        assume_zero_initial_debt: bool,
    ) -> None:
        self.create_agents(
            econ_reader=econ_reader,
            ons_reader=ons_reader,
            exchange_rates=exchange_rates,
            total_firm_deposits=total_firm_deposits,
            total_firm_debt=total_firm_debt,
            industry_data=industry_data,
            number_of_employees_by_industry=number_of_employees_by_industry,
            initial_inventory_to_production_fraction=initial_inventory_to_production_fraction,
            intermediate_inputs_utilisation_rate=intermediate_inputs_utilisation_rate,
            capital_inputs_utilisation_rate=capital_inputs_utilisation_rate,
            assume_zero_initial_deposits=assume_zero_initial_deposits,
            assume_zero_initial_debt=assume_zero_initial_debt,
        )

    def set_industries(
        self,
        number_of_firms_by_industry: np.ndarray,
    ) -> None:
        self.number_of_firms_by_industry = number_of_firms_by_industry
        self.number_of_firms = np.sum(self.number_of_firms_by_industry)

        # Set firm industries
        self.firm_data["Industry"] = np.array(
            reduce(
                lambda a, b: a + b,
                (
                    [industry] * s
                    for industry, s in zip(
                        range(len(self.industries)),
                        list(self.number_of_firms_by_industry),
                    )
                ),
            )
        )

    def set_firm_sizes(
        self,
        number_of_employees_by_industry: np.ndarray,
        econ_reader: OECDEconData,
        ons_reader: ONSReader,
    ) -> None:
        self.firm_data["Number of Employees"] = np.zeros(self.number_of_firms)

        # Try the OECD, otherwise take ONS data
        firm_size_zetas = econ_reader.read_firm_size_zetas(
            self.country_name,
            self.year,
        )
        if firm_size_zetas is None:
            firm_size_zetas = ons_reader.get_firm_size_zetas()

        for industry in range(len(self.industries)):
            # Sanity check
            if number_of_employees_by_industry[industry] < self.number_of_firms_by_industry[industry]:
                print(
                    "Warning: Fewer Firms than Employees in Sector",
                    industry,
                    number_of_employees_by_industry[industry],
                    self.number_of_firms_by_industry[industry],
                )

            # Draw firm sizes
            sizes = self.draw_firm_sizes(
                industry,
                number_of_employees_by_industry[industry],
                firm_size_zetas[industry],
            )

            # Distribute the remainder
            self.distribute_remainder(
                industry,
                number_of_employees_by_industry,
                sizes,
            )

            # Update the field
            self.firm_data.loc[self.firm_data["Industry"] == industry, "Number of Employees"] = sizes
            self.firm_data["Number of Employees"] = self.firm_data["Number of Employees"].astype(int)

    def draw_firm_sizes(
        self,
        industry: int,
        number_employees: int,
        firm_size_zeta_shape: float,
    ) -> np.ndarray:
        employees_by_industry_range = np.arange(1, number_employees + 1)
        if len(employees_by_industry_range) == 0:
            return np.zeros(self.number_of_firms_by_industry[industry])
        probs = 1 / (employees_by_industry_range**firm_size_zeta_shape * special.zetac(firm_size_zeta_shape))
        sizes_raw = np.random.choice(
            employees_by_industry_range,
            p=probs / sum(probs),
            size=self.number_of_firms_by_industry[industry],
            replace=True,
        )
        n_emp_dist = number_employees - self.number_of_firms_by_industry[industry]

        sizes_raw = sizes_raw / sizes_raw.sum() * n_emp_dist
        sizes = 1 + np.rint(sizes_raw).astype("int")

        while sum(sizes) > number_employees:
            idx = np.random.randint(0, len(sizes))
            if sizes[idx] > 1:
                sizes[idx] = sizes[idx] - 1

        while sum(sizes) < number_employees:
            idx = np.random.randint(0, len(sizes))
            sizes[idx] = sizes[idx] + 1

        assert number_employees == sum(sizes)

        return sizes

    def distribute_remainder(
        self,
        industry: int,
        number_employees_by_industry: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        remainder = number_employees_by_industry[industry] - sizes.sum()
        abs_rem = np.abs(remainder)
        f_choices = np.random.choice(self.number_of_firms_by_industry[industry], size=int(abs_rem))
        for f in f_choices:
            new_size = sizes[f] + np.sign(remainder)
            while new_size <= 1:
                f = np.random.choice(self.number_of_firms_by_industry[industry])
                new_size = sizes[f] + np.sign(remainder)
            sizes[f] = new_size

    def set_firm_wages(
        self,
        number_of_employees_by_industry: np.ndarray,
        labour_compensation: np.ndarray,
        tau_sif: float,
    ) -> None:
        firm_wages = np.zeros(self.number_of_firms)
        for industry in range(len(self.industries)):
            if number_of_employees_by_industry[industry] > 0:
                firm_wages[self.firm_data["Industry"] == industry] = (
                    self.firm_data.loc[
                        self.firm_data["Industry"] == industry,
                        "Number of Employees",
                    ]
                    / number_of_employees_by_industry[industry]
                    * (labour_compensation[industry] / (1 + tau_sif))
                )
        self.firm_data["Total Wages"] = firm_wages
        self.firm_data["Total Wages Paid"] = (1 + tau_sif) * firm_wages

    def set_firm_production(
        self,
        number_of_employees_by_industry: np.ndarray,
        output: np.ndarray,
    ) -> None:
        self.firm_data["Production"] = np.nan
        for industry in range(len(self.industries)):
            self.firm_data.loc[self.firm_data["Industry"] == industry, "Production"] = np.divide(
                self.firm_data.loc[
                    self.firm_data["Industry"] == industry,
                    "Number of Employees",
                ].values
                * output[industry],
                float(number_of_employees_by_industry[industry]),
                out=np.zeros(
                    self.firm_data.loc[
                        self.firm_data["Industry"] == industry,
                        "Number of Employees",
                    ].values.shape
                ),
                where=number_of_employees_by_industry[industry] != 0,
            )

    def set_firm_prices(self, exchange_rates: WorldBankRatesReader) -> None:
        self.firm_data["Price in USD"] = np.ones_like(self.firm_data["Production"].values)
        self.firm_data["Price"] = exchange_rates.from_usd_to_lcu(self.country_name, self.year)

    def set_firm_labour_inputs(self) -> None:
        self.firm_data["Labour Inputs"] = self.firm_data["Production"].copy()

    def set_firm_inventory(self, initial_inventory_to_production_fraction: float) -> None:
        self.firm_data["Inventory"] = initial_inventory_to_production_fraction * self.firm_data["Production"].values

    def set_firm_demand(self) -> None:
        self.firm_data["Demand"] = self.firm_data["Production"].values + self.firm_data["Inventory"].values

    def set_firm_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        initial_utilisation_rate: float,
    ) -> None:
        self.intermediate_inputs_stock = (
            1.0
            / initial_utilisation_rate
            * (
                self.firm_data["Production"].values
                / intermediate_inputs_productivity_matrix[:, self.firm_data["Industry"].values]
            ).T
        )

    def set_firm_used_intermediate_inputs(self, intermediate_inputs_productivity_matrix: np.ndarray) -> None:
        self.used_intermediate_inputs = (
            self.firm_data["Production"].values
            / intermediate_inputs_productivity_matrix[:, self.firm_data["Industry"].values]
        ).T.astype(float)

    def set_firm_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        initial_utilisation_rate: float,
    ) -> None:
        self.capital_inputs_stock = (
            1.0
            / initial_utilisation_rate
            * (
                self.firm_data["Production"].values
                / capital_inputs_productivity_matrix[:, self.firm_data["Industry"].values]
            ).T
        )

    def set_firm_used_capital_inputs(self, capital_inputs_depreciation_matrix: np.ndarray) -> None:
        self.used_capital_inputs = (
            self.firm_data["Production"].values
            * capital_inputs_depreciation_matrix[:, self.firm_data["Industry"].values]
        ).T.astype(float)

    def set_firm_deposits(self, total_firm_deposits: float, assume_zero_initial_deposits: bool) -> None:
        if assume_zero_initial_deposits:
            self.firm_data["Deposits"] = 0.0
        else:
            self.firm_data["Deposits"] = (
                self.firm_data["Production"] / self.firm_data["Production"].sum() * total_firm_deposits
            )

    def set_firm_debt(self, total_firm_debt: float, assume_zero_initial_debt: bool) -> None:
        if assume_zero_initial_debt:
            self.firm_data["Debt"] = 0.0
        else:
            self.firm_data["Debt"] = (
                self.capital_inputs_stock.sum(axis=1) / self.capital_inputs_stock.sum() * total_firm_debt
            )

    def set_firm_equity(self) -> None:
        self.firm_data["Equity"] = (
            self.firm_data["Deposits"]
            + self.firm_data["Price"] * self.firm_data["Inventory"]
            + self.firm_data["Price"] * self.intermediate_inputs_stock.sum(axis=1)
            + self.firm_data["Price"] * self.capital_inputs_stock.sum(axis=1)
            - self.firm_data["Debt"]
        )

    def set_taxes_paid_on_production(self, taxes_less_subsidies_rates: np.ndarray) -> None:
        self.firm_data["Taxes paid on Production"] = (
            taxes_less_subsidies_rates[self.firm_data["Industry"].values]
            * self.firm_data["Production"].values
            * self.firm_data["Price"].values
        )

    def set_interest_paid(
        self,
        interest_rate_on_firm_deposits: np.ndarray,
        overdraft_rate_on_firm_deposits: np.ndarray,
        credit_market_data: pd.DataFrame,
    ) -> None:
        # Interest on deposits
        self.firm_data["Interest paid on deposits"] = -interest_rate_on_firm_deposits[
            self.firm_data["Corresponding Bank ID"].values
        ] * np.maximum(0.0, self.firm_data["Deposits"].values) - overdraft_rate_on_firm_deposits[
            self.firm_data["Corresponding Bank ID"].values
        ] * np.minimum(
            0.0, self.firm_data["Deposits"].values
        )

        # Interest paid on loans
        credit_market_data_firm_loans = credit_market_data.loc[credit_market_data["loan_type"] == 2]
        interest_on_loans = np.zeros(len(self.firm_data))
        for firm_id in range(len(self.firm_data)):
            curr_loans = credit_market_data_firm_loans[credit_market_data_firm_loans["loan_recipient_id"] == firm_id]
            for loan_id in range(len(curr_loans)):
                interest_on_loans[firm_id] += float(
                    curr_loans.iloc[loan_id]["loan_interest_rate"] * curr_loans.iloc[loan_id]["loan_value"]
                )
        self.firm_data["Interest paid on loans"] = interest_on_loans

        # Total interest paid
        self.firm_data["Interest paid"] = (
            self.firm_data["Interest paid on deposits"] + self.firm_data["Interest paid on loans"]
        )

    def set_firm_profits(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        tau_sif: float,
    ) -> None:
        # Sales
        sales = self.firm_data["Production"].values * self.firm_data["Price"].values

        # Labour
        labour_costs = self.firm_data["Total Wages"].values * (1 + tau_sif)

        # Intermediate inputs
        intermediate_inputs_costs = (
            self.firm_data["Production"].values
            / intermediate_inputs_productivity_matrix[:, self.firm_data["Industry"].values]
        ).T.sum(axis=1) * self.firm_data["Price"].values

        # Capital inputs
        capital_inputs_costs = (
            self.firm_data["Production"].values
            * capital_inputs_depreciation_matrix[:, self.firm_data["Industry"].values]
        ).T.sum(axis=1) * self.firm_data["Price"].values

        # Update profits
        self.firm_data["Profits"] = (
            sales
            - labour_costs
            - intermediate_inputs_costs
            - capital_inputs_costs
            - self.firm_data["Taxes paid on Production"].values
            - self.firm_data["Interest paid"].values
        )

        print("Initial profits")
        print(self.firm_data["Profits"])

    def set_unit_costs(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        tau_sif: float,
    ) -> None:
        # Labour
        labour_costs = self.firm_data["Total Wages"].values * (1 + tau_sif)

        # Intermediate inputs
        intermediate_inputs_costs = (
            self.firm_data["Production"].values[:, None]
            / intermediate_inputs_productivity_matrix[:, self.firm_data["Industry"].values].T
        ).sum(axis=1) * self.firm_data["Price"].values

        # Capital inputs
        capital_inputs_costs = (
            self.firm_data["Production"].values[:, None]
            * capital_inputs_depreciation_matrix[:, self.firm_data["Industry"].values].T
        ).sum(axis=1) * self.firm_data["Price"].values

        # Update unit costs
        self.firm_data["Unit Costs"] = (
            labour_costs
            + intermediate_inputs_costs
            + capital_inputs_costs
            + self.firm_data["Taxes paid on Production"].values
        ) / self.firm_data["Production"].values

    def set_corporate_taxes_paid(self, tau_firm: float) -> None:
        self.firm_data["Corporate Taxes Paid"] = tau_firm * np.maximum(0.0, self.firm_data["Profits"])

    def set_firm_debt_installments(self, credit_market_data: pd.DataFrame) -> None:
        credit_market_data_firm_loans = credit_market_data.loc[credit_market_data["loan_type"] == 2]
        debt_installments = np.zeros(len(self.firm_data))
        for firm_id in range(len(self.firm_data)):
            curr_loans = credit_market_data_firm_loans[credit_market_data_firm_loans["loan_recipient_id"] == firm_id]
            for loan_id in range(len(curr_loans)):
                debt_installments[firm_id] += float(
                    curr_loans.iloc[loan_id]["loan_value"] / curr_loans.iloc[loan_id]["loan_maturity"]
                )
        self.firm_data["Debt Installments"] = debt_installments
