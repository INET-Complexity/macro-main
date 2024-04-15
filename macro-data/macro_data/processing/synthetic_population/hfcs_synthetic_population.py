import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.hfcs_household_tools import (
    set_household_types,
    set_household_housing_data,
)
from macro_data.processing.synthetic_population.hfcs_individual_tools import process_individual_data
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.exogenous_data import ExogenousCountryData
from macro_data.util.clean_data import remove_outliers
from macro_data.util.imputation import apply_iterative_imputer
from macro_data.util.regressions import fit_linear

RESTRICT_COLS = [
    "Type",
    "Corresponding Individuals ID",
    "Corresponding Bank ID",
    "Corresponding Inhabited House ID",
    "Corresponding Renters",
    "Corresponding Property Owner",
    "Corresponding Additionally Owned Houses ID",
    "Income",
    "Investment",
    "Employee Income",
    "Regular Social Transfers",
    "Rental Income from Real Estate",
    "Income from Financial Assets",
    "Saving Rate",
    "Rent Paid",
    "Rent Imputed",
    "Wealth",
    "Net Wealth",
    "Wealth in Real Assets",
    "Value of the Main Residence",
    "Value of other Properties",
    "Wealth Other Real Assets",
    "Wealth in Deposits",
    "Wealth in Other Financial Assets",
    "Wealth in Financial Assets",
    "Outstanding Balance of HMR Mortgages",
    "Outstanding Balance of Mortgages on other Properties",
    "Outstanding Balance of other Non-Mortgage Loans",
    "Debt",
    "Debt Installments",
    "Tenure Status of the Main Residence",
    "Number of Properties other than Household Main Residence",
]

CONVERT_HH_COLS = [
    "Rent Paid",
    "Amount spent on Consumption of Goods and Services",
    "Rental Income from Real Estate",
    "Income from Financial Assets",
    "Income from Pensions",
    "Regular Social Transfers",
    "Income",
    "Value of the Main Residence",
    "Value of other Properties",
    "Value of Household Vehicles",
    "Value of Household Valuables",
    "Value of Self-Employment Businesses",
    "Wealth in Deposits",
    "Mutual Funds",
    "Bonds",
    "Value of Private Businesses",
    "Shares",
    "Money owed to Households",
    "Other Assets",
    "Voluntary Pension",
    "Outstanding Balance of HMR Mortgages",
    "Outstanding Balance of Mortgages on other Properties",
    "Outstanding Balance of Credit Line",
    "Outstanding Balance of Credit Card Debt",
    "Outstanding Balance of other Non-Mortgage Loans",
]

CONVERT_IND_COLS = ["Employee Income", "Income from Unemployment Benefits", "Income"]


class SyntheticHFCSPopulation(SyntheticPopulation):
    """
    A class representing a synthetic population generated from HFCSPopulation data.

    Attributes:
        country_name (str): The name of the country.
        country_name_short (str): The short name of the country.
        scale (int): The scale of the population.
        year (int): The year of the population data.
        industries (list[str]): The list of industries.
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        household_data (pd.DataFrame): The DataFrame containing household data.
        social_housing_rent (float): The rent for social housing.
        coefficient_fa_income (float): The coefficient for FA income.
        consumption_weights (np.ndarray): The array of consumption weights.
        consumption_weights_by_income (np.ndarray): The array of consumption weights by income.

    Methods:
        create_from_readers(cls, readers, country_name, country_name_short, scale, year, industry_data, industries, total_unemployment_benefits, rent_as_fraction_of_unemployment_rate, n_quantiles=5):
            Creates a SyntheticHFCSPopulation instance from data readers.

        restrict(self):
            Restricts the household data to selected columns.

        compute_household_wealth(self):
            Computes the wealth-related fields in the household data.

        compute_household_income(self, total_social_transfers):
            Computes the income-related fields in the household data.

        set_household_housing_data(self, rent_as_fraction_of_unemployment_rate, unemployment_benefits_by_capita):
            Sets the housing-related fields in the household data.
    """

    def __init__(
        self,
        country_name: str,
        country_name_short: str,
        scale: int,
        year: int,
        industries: list[str],
        individual_data: pd.DataFrame,
        household_data: pd.DataFrame,
        social_housing_rent: float,
        coefficient_fa_income: float,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
    ):
        saving_rates_model = LinearRegression()
        social_transfers_model = LinearRegression()
        wealth_distribution_model = LinearRegression()
        super().__init__(
            country_name,
            country_name_short,
            scale,
            year,
            industries,
            individual_data,
            household_data,
            social_housing_rent,
            coefficient_fa_income,
            consumption_weights,
            consumption_weights_by_income,
            saving_rates_model,
            social_transfers_model,
            wealth_distribution_model,
        )

    # TODO rent as fraction of unemployment rate seems to be a parameter of government functions
    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        country_name_short: str,
        scale: int,
        year: int,
        quarter: int,
        industry_data: dict[str, pd.DataFrame],
        industries: list[str],
        total_unemployment_benefits: float,
        exogenous_data: ExogenousCountryData,
        rent_as_fraction_of_unemployment_rate: float = 0.25,
        n_quantiles: int = 5,
        population_ratio: float = 1.0,
        exch_rate: float = 1.0,
        proxied_country: str | Country = None,
        yearly_factor: float = 4.0,
    ) -> "SyntheticHFCSPopulation":
        """
        Creates a synthetic population from data readers.

        Args:
            cls: The class object.
            readers (DataReaders): The data readers object.
            country_name (str): The name of the country.
            country_name_short (str): The short name of the country.
            scale (int): The scaling factor.
            year (int): The year.
            quarter (int): The quarter.
            industry_data (dict[str, dict]): The industry data.
            industries (list[str]): The list of industries.
            total_unemployment_benefits (float): The total unemployment benefits.
            exogenous_data (ExogenousCountryData): The exogenous data for the Country.
            rent_as_fraction_of_unemployment_rate (float): The rent as a fraction of the unemployment rate.
            n_quantiles (int, optional): The number of quantiles. Defaults to 5.
            population_ratio (float, optional): The population ratio. Defaults to 1.0. This is used in case
                                                 the HFCS population is used as a proxy for another country.
            exch_rate (float, optional): The exchange rate. Defaults to 1.0.
            proxied_country (str, optional): The name of the proxied country. Defaults to None.
            yearly_factor (float, optional): The yearly factor. Defaults to 4.0 for 4 quarters.

        Returns:
            cls: The synthetic population object.
        """

        n_households = int(readers.eurostat.number_of_households(country_name, year) * population_ratio / scale)
        hfcs_individuals_data = readers.hfcs[country_name].individuals_df
        hfcs_households_data = readers.hfcs[country_name].households_df

        household_data, individual_data = sample_households(hfcs_households_data, hfcs_individuals_data, n_households)

        unemployment_rate = exogenous_data.labour_stats.loc[f"{year}-Q{quarter}", "Unemployment Rate (Value)"].iloc[0]
        participation_rate = exogenous_data.labour_stats.loc[f"{year}-Q{quarter}", "Participation Rate (Value)"].iloc[0]

        n_firms_by_industry = industry_data["industry_vectors"]["Number of Firms"].values

        individual_data = process_individual_data(
            individual_data,
            industries,
            scale,
            total_unemployment_benefits,
            unemployment_rate,
            participation_rate,
            n_firms_by_industry,
        )
        n_unemployed = np.sum(individual_data["Activity Status"] == 2)

        household_data = remove_outliers(
            data=household_data,
            cols=[
                "Rent Paid",
                "Income",
                "Consumption of Consumer Goods/Services as a Share of Income",
            ],
            use_logpdf=False,
        )

        if exch_rate != 1.0:
            household_data.loc[:, CONVERT_HH_COLS] *= exch_rate
            individual_data.loc[:, CONVERT_IND_COLS] *= exch_rate

        household_data = household_data.loc[household_data["Corresponding Individuals ID"].notna()]

        household_data = set_household_types(household_data, individual_data)

        # DANGER if we don't have total unemployment benefits
        # we set them to 0, must be checked!!!
        if total_unemployment_benefits is None:
            total_unemployment_benefits = 0.0
            logging.warning("Total unemployment benefits not provided, setting them to 0.0")

        household_data = set_household_housing_data(
            household_data,
            scale,
            rent_as_fraction_of_unemployment_rate,
            unemployment_benefits_by_capita=total_unemployment_benefits / n_unemployed,
        )

        # initialise fields to nans, will be filled later when computing wealth
        non_initialised_fields = [
            "Wealth Other Real Assets",
            "Wealth in Real Assets",
            "Wealth in Other Financial Assets",
            "Wealth in Financial Assets",
            "Wealth",
            "Debt",
            "Net Wealth",
            "Employee Income",
        ]
        for field in non_initialised_fields:
            household_data[field] = np.nan

        social_housing_rent = rent_as_fraction_of_unemployment_rate * total_unemployment_benefits / n_unemployed
        consumption_weights = industry_data["industry_vectors"]["Household Consumption Weights"].values
        consumption_weights_by_income = np.zeros((n_quantiles, len(consumption_weights)))
        for i in range(n_quantiles):
            consumption_weights_by_income[i] = consumption_weights

        individual_data["Corresponding Firm ID"] = np.nan

        return cls(
            country_name=country_name,
            country_name_short=country_name_short,
            scale=scale,
            year=year,
            industries=industries,
            household_data=household_data,
            individual_data=individual_data,
            social_housing_rent=social_housing_rent,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            coefficient_fa_income=0.0,
        )

    def restrict(self) -> None:
        self.household_data = self.household_data[RESTRICT_COLS]

    def compute_household_wealth(self, independents: Optional[list[str]] = None) -> None:
        """
        Computes the household wealth based on the given independent variables.

        Args:
            independents (Optional[list[str]]): The list of independent variables. Defaults to ["Income", "Debt"].
        """

        if independents is None:
            independents = ["Income", "Debt"]
        self.set_household_other_real_assets_wealth()
        self.set_household_total_real_assets()
        self.set_household_deposits()
        self.set_household_other_financial_assets()
        self.set_household_financial_assets()
        self.set_household_wealth()
        self.set_household_mortgage_debt()
        self.set_household_other_debt()
        self.set_household_debt()
        self.set_household_net_wealth()
        # TODO: move to the model package
        self.set_wealth_distribution_function(independents=independents)

    def compute_household_income(
        self,
        total_social_transfers: float,
        independents: Optional[list[str]] = None,
    ) -> None:
        """
        Computes the household income based on the given total social transfers.

        Args:
            total_social_transfers (float): The total amount of social transfers.
            independents (Optional[list[str]]): The list of independent variables. Defaults to ["Income", "Debt"].

        Returns:
            None
        """

        if independents is None:
            independents = ["Income", "Debt"]

        self.set_household_social_transfers(
            total_social_transfers=total_social_transfers,
            independents=independents,
        )
        self.set_household_employee_income()
        self.set_household_income_from_financial_assets()
        self.set_household_income()

    def set_household_other_real_assets_wealth(self) -> None:
        self.household_data.loc[
            self.household_data["Value of Household Vehicles"].isna(),
            "Value of Household Vehicles",
        ] = 0.0
        self.household_data.loc[
            self.household_data["Value of Household Valuables"].isna(),
            "Value of Household Valuables",
        ] = 0.0
        self.household_data.loc[
            self.household_data["Value of Self-Employment Businesses"].isna(),
            "Value of Self-Employment Businesses",
        ] = 0.0

        self.household_data["Wealth Other Real Assets"] = (
            self.household_data["Value of Household Vehicles"]
            + self.household_data["Value of Household Valuables"]
            + self.household_data["Value of Self-Employment Businesses"]
        )
        self.household_data.loc[:, "Wealth Other Real Assets"] *= self.scale

    def set_household_total_real_assets(self) -> None:
        self.household_data["Wealth in Real Assets"] = (
            self.household_data["Value of the Main Residence"]
            + self.household_data["Value of other Properties"]
            + self.household_data["Wealth Other Real Assets"]
        )

    def set_household_deposits(self) -> None:
        self.household_data.loc[self.household_data["Wealth in Deposits"].isna(), "Wealth in Deposits"] = 0.0
        self.household_data.loc[:, "Outstanding Balance of Credit Line"] = 0.0
        self.household_data.loc[:, "Outstanding Balance of Credit Card Debt"] = 0.0
        self.household_data.loc[:, "Wealth in Deposits"] *= self.scale

    def set_household_other_financial_assets(self) -> None:
        self.household_data.loc[self.household_data["Mutual Funds"].isna(), "Mutual Funds"] = 0.0
        self.household_data.loc[self.household_data["Bonds"].isna(), "Bonds"] = 0.0
        self.household_data.loc[
            self.household_data["Value of Private Businesses"].isna(),
            "Value of Private Businesses",
        ] = 0.0
        self.household_data.loc[self.household_data["Shares"].isna(), "Shares"] = 0.0
        self.household_data.loc[self.household_data["Managed Accounts"].isna(), "Managed Accounts"] = 0.0
        self.household_data.loc[
            self.household_data["Money owed to Households"].isna(),
            "Money owed to Households",
        ] = 0.0
        self.household_data.loc[self.household_data["Other Assets"].isna(), "Other Assets"] = 0.0
        self.household_data.loc[self.household_data["Voluntary Pension"].isna(), "Voluntary Pension"] = 0.0

        self.household_data["Wealth in Other Financial Assets"] = (
            self.household_data["Mutual Funds"]
            + self.household_data["Bonds"]
            + self.household_data["Value of Private Businesses"]
            + self.household_data["Shares"]
            + self.household_data["Managed Accounts"]
            + self.household_data["Money owed to Households"]
            + self.household_data["Other Assets"]
            + self.household_data["Voluntary Pension"]
        )
        self.household_data.loc[:, "Wealth in Other Financial Assets"] *= self.scale

    def set_household_financial_assets(self) -> None:
        self.household_data["Wealth in Financial Assets"] = (
            self.household_data["Wealth in Deposits"] + self.household_data["Wealth in Other Financial Assets"]
        )

    def set_household_wealth(self) -> None:
        self.household_data["Wealth"] = (
            self.household_data["Wealth in Real Assets"] + self.household_data["Wealth in Financial Assets"]
        )

    def set_household_mortgage_debt(self) -> None:
        self.household_data.loc[
            self.household_data["Outstanding Balance of HMR Mortgages"].isna(),
            "Outstanding Balance of HMR Mortgages",
        ] = 0.0
        self.household_data.loc[
            self.household_data["Outstanding Balance of Mortgages on other Properties"].isna(),
            "Outstanding Balance of Mortgages on other Properties",
        ] = 0.0
        self.household_data.loc[:, "Outstanding Balance of HMR Mortgages"] *= self.scale
        self.household_data.loc[:, "Outstanding Balance of Mortgages on other Properties"] *= self.scale

    def set_household_other_debt(self) -> None:
        self.household_data.loc[
            self.household_data["Outstanding Balance of other Non-Mortgage Loans"].isna(),
            "Outstanding Balance of other Non-Mortgage Loans",
        ] = 0.0
        self.household_data.loc[:, "Outstanding Balance of other Non-Mortgage Loans"] *= self.scale

    def set_household_debt(self) -> None:
        self.household_data["Debt"] = (
            self.household_data["Outstanding Balance of HMR Mortgages"]
            + self.household_data["Outstanding Balance of Mortgages on other Properties"]
            + self.household_data["Outstanding Balance of other Non-Mortgage Loans"]
        )

    def set_debt_installments(self, credit_market_data: pd.DataFrame) -> None:
        """
        Sets the debt installments for each household based on the credit market data.

        Args:
            credit_market_data (DataFrame): The credit market data.

        Returns:
            None
        """
        credit_market_data_household_loans = credit_market_data.loc[credit_market_data["loan_type"].isin([4, 5])]
        debt_installments = np.zeros(len(self.household_data))
        for household_id in range(len(self.household_data)):
            curr_loans = credit_market_data_household_loans[
                credit_market_data_household_loans["loan_recipient_id"] == household_id
            ]
            for loan_id in range(len(curr_loans)):
                debt_installments[household_id] += float(
                    curr_loans.iloc[loan_id]["loan_value"] / curr_loans.iloc[loan_id]["loan_maturity"]
                )
        self.household_data["Debt Installments"] = debt_installments

    def set_household_net_wealth(self) -> None:
        """
        Calculates and sets the net wealth of each household.

        The net wealth is calculated by subtracting the household's debt from its total wealth.

        Returns:
            None
        """
        self.household_data["Net Wealth"] = self.household_data["Wealth"] - self.household_data["Debt"]

    def set_wealth_distribution_function(self, independents: Optional[list[str]] = None) -> None:
        """
        Sets the wealth distribution function based on the given independent variables.

        Parameters:
            independents (list[str]): List of independent variables.

        Returns:
            None
        """

        if independents is None:
            independents = ["Income", "Debt"]
        self.household_data["Fraction Deposits / Total Financial Wealth"] = np.divide(
            self.household_data["Wealth in Deposits"].values.astype(float),
            self.household_data["Wealth in Financial Assets"].values.astype(float),
            out=np.ones_like(self.household_data["Wealth in Deposits"].values),
            where=self.household_data["Wealth in Financial Assets"].values.astype(float) != 0.0,
        )
        fit_linear(
            data=self.household_data,
            independents=independents,
            dependent="Fraction Deposits / Total Financial Wealth",
            model=self.wealth_distribution_model,
        )

    def set_household_employee_income(self) -> None:
        self.household_data["Employee Income"] = [
            self.individual_data.loc[self.household_data["Corresponding Individuals ID"][i], "Income"].sum()
            for i in range(len(self.household_data))
        ]

    def set_household_social_transfers(
        self, total_social_transfers: float, independents: Optional[list[str]] = None
    ) -> None:
        """
        Sets the household social transfers based on the total social transfers and independent variables.

        Parameters:
            total_social_transfers (float): The total amount of social transfers.
            independents (Optional[list[str]]): The list of independent variables. Defaults to None.

        Returns:
            None
        """
        if independents is None:
            independents = ["Income", "Debt"]
        # Household regular social transfers and pensions, impute missing values
        self.household_data = apply_iterative_imputer(
            self.household_data,
            [
                "Type",
                "Net Wealth",
                "Regular Social Transfers",
                "Income from Pensions",
            ],
            min_value=0,
        )

        # Aggregate
        self.household_data["Regular Social Transfers"] += self.household_data["Income from Pensions"].values

        self.household_data["Income from Pensions"] = 0.0

        # Social transfers for each household group
        self.household_data["Regular Social Transfers"] /= self.household_data["Regular Social Transfers"].sum()
        social_transfers = fit_linear(
            data=self.household_data,
            independents=independents,
            dependent="Regular Social Transfers",
            model=self.social_transfers_model,
        )
        social_transfers[social_transfers < 0] = 0.0
        self.household_data["Regular Social Transfers"] = social_transfers

        # Rescale them
        self.household_data["Regular Social Transfers"] *= (
            total_social_transfers / self.household_data["Regular Social Transfers"].sum()
        )

    def set_household_income_from_financial_assets(self) -> None:
        """
        Sets the household monthly income from financial assets based on the wealth in other financial assets.

        This method calculates the coefficient of income from financial assets based on the sum of income from financial
        assets divided by the sum of wealth in other financial assets. It then multiplies the coefficient with the wealth
        in other financial assets to determine the income from financial assets for each household.

        Returns:
            None
        """
        fa_mask = np.logical_not(np.isnan(self.household_data["Income from Financial Assets"].values.astype(float)))
        self.household_data["Income from Financial Assets"] *= self.scale
        self.coefficient_fa_income = (
            self.household_data["Income from Financial Assets"].values.astype(float)[fa_mask].sum() / self.yearly_factor
        ) / (self.household_data["Wealth in Other Financial Assets"].values.astype(float)[fa_mask]).sum()
        self.household_data["Income from Financial Assets"] = (
            self.coefficient_fa_income * self.household_data["Wealth in Other Financial Assets"].values
        )

    def set_household_income(self) -> None:
        """
        Calculates the total household income by summing up different income sources.

        Returns:
            None
        """
        self.household_data["Income"] = (
            self.household_data["Employee Income"]
            + self.household_data["Regular Social Transfers"]
            + self.household_data["Rental Income from Real Estate"]
            + self.household_data["Income from Financial Assets"]
        )

    def set_household_investment_rates(self, investment_rates: np.ndarray | float = 0.2) -> None:
        """
        Sets the investment rates for each household based on the given investment rates.

        Parameters:
            investment_rates (np.ndarray): The investment rates.

        Returns:
            None
        """
        self.household_data["Investment Rate"] = investment_rates

    def set_household_saving_rates(self, independents: Optional[list[str]] = None) -> None:
        """
        Sets the saving rates for each household based on the given independent variables.
        This method imputes missing values in household data, and then defines the saving rates as 1 minus the consumption share.

        Parameters:
            independents (Optional[list[str]]): List of independent variables to consider. If not provided, defaults to ["Income", "Debt"].

        Returns:
            None
        """
        # TODO: does this make sense? average consumption of goods/services is 36% (median similar too)
        if independents is None:
            independents = ["Income", "Debt"]
        # Some obvious cleaning
        self.household_data.loc[
            self.household_data["Consumption of Consumer Goods/Services as a Share of Income"].isin(["A"]),
            "Consumption of Consumer Goods/Services as a Share of Income",
        ] = np.nan
        self.household_data.loc[:, "Consumption of Consumer Goods/Services as a Share of Income"] = self.household_data[
            "Consumption of Consumer Goods/Services as a Share of Income"
        ].astype(float)
        self.household_data.loc[
            self.household_data["Consumption of Consumer Goods/Services as a Share of Income"] > 1.0,
            "Consumption of Consumer Goods/Services as a Share of Income",
        ] = np.nan

        self.household_data["Consumption of Consumer Goods/Services as a Share of Income"] = self.household_data[
            "Consumption of Consumer Goods/Services as a Share of Income"
        ].astype(float)

        # Impute missing values
        imputed_consumption_share = IterativeImputer(min_value=0, max_value=1).fit_transform(
            self.household_data[
                [
                    "Type",
                    "Income",
                    "Wealth",
                    "Debt",
                    "Consumption of Consumer Goods/Services as a Share of Income",
                ]
            ].values
        )[:, 4]
        self.household_data["Saving Rate"] = 1.0 - imputed_consumption_share

        # Fit a model
        saving_rates = fit_linear(
            data=self.household_data,
            independents=independents,
            dependent="Saving Rate",
            model=self.saving_rates_model,
        )

        self.household_data["Saving Rate"] = saving_rates

    def normalise_household_investment(
        self, tau_cf: float, iot_hh_investment: np.ndarray | pd.Series, positive_investment_rates: bool = True
    ):
        inv_weights = iot_hh_investment / iot_hh_investment.sum()
        income = self.household_data["Income"].values
        investment_rate = self.household_data["Investment Rate"].values

        current_hh_investment = default_target_investment(
            income_=income, investment_rate=investment_rate, tau_cf_=tau_cf, investment_weights_=inv_weights
        )

        factor = iot_hh_investment.sum() / current_hh_investment.sum()

        self.household_data["Investment Rate"] = factor * investment_rate

        # set initial investment
        self.household_data["Investment"] = (self.household_data["Income"] * self.household_data["Investment Rate"]) / (
            1 + tau_cf
        )

    def normalise_household_consumption(
        self,
        iot_hh_consumption: np.ndarray | pd.Series,
        vat: float,
        positive_saving_rates_only: bool = True,
        independents: Optional[list[str]] = None,
    ) -> None:
        """
        Normalizes the household consumption based on the input parameters.
        This is necessary because we need to match the household consumption to the IOT data.

        We first adjust the savings rate of the households by multiplying it with a factor that ensures that the total
        consumption of the households across all sectors and income quantiles matches the aggregate household consumption.

        Next, we need to make sure that the different consumption shares of the different quantiles match the aggregate household consumption
        for each sector. We do this by solving a constrained optimisation problem, where the constraints are that the consumptions match,
        and that is set up so that the consumption shares are a weighted average of the consumption shares of the quantiles
        (from the consumption_weights attribute) and of the averaged consumption shares over all quantiles.

        This attempts to get consumption shares that are not too far from the quantile data, but that also match the aggregate consumption.

        Args:
            iot_hh_consumption (np.ndarray | pd.Series): The household consumption to be normalized.
            vat (float): The value-added tax rate.
            positive_saving_rates_only (bool, optional): Flag indicating whether to enforce positive saving rates only. Defaults to True.
            independents (Optional[list[str]], optional): List of independent variables. Defaults to None.
        """
        if independents is None:
            independents = ["Income", "Debt"]
        cons_weights = self.consumption_weights
        income = self.household_data["Income"].values
        sr = self.household_data["Saving Rate"].values
        current_hh_consumption = default_desired_consumption(
            income_=income,
            consumption_weights_=cons_weights,
            saving_rates_=sr,
            tau_vat_=vat,
        )

        # Adjust saving rates
        factor = iot_hh_consumption.sum() / current_hh_consumption.sum()
        self.household_data["Saving Rate"] = 1 - (1 - self.household_data["Saving Rate"]) * factor
        if positive_saving_rates_only:
            sr = self.household_data["Saving Rate"].values
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
            self.household_data["Saving Rate"] = factor * sr

        self.household_data["Consumption"] = (
            1 / (1 + vat) * (1 - self.household_data["Saving Rate"]) * self.household_data["Income"]
        )

        # Overwrite the model
        fit_linear(
            data=self.household_data,
            independents=independents,
            dependent="Saving Rate",
            model=self.saving_rates_model,
        )

    def match_consumption_weights_by_income(
        self,
        weights_by_income: np.ndarray | pd.DataFrame,
        iot_hh_consumption: pd.Series,
        vat: float,
        consumption_variance: float = 0.1,
    ) -> None:
        n_quantiles = weights_by_income.shape[1]
        if isinstance(iot_hh_consumption, pd.Series):
            iot_hh_consumption_norm = (iot_hh_consumption / iot_hh_consumption.sum()).values
            iot_hh_consumption = iot_hh_consumption.values
        else:
            iot_hh_consumption_norm = iot_hh_consumption / iot_hh_consumption.sum(axis=0)
        if isinstance(weights_by_income, pd.DataFrame):
            weights_by_income = weights_by_income.values
        quintiles = pd.qcut(self.household_data["Income"], n_quantiles, labels=False)
        disposable_income_by_quantile = (
            self.household_data.groupby(quintiles).apply(lambda x: ((1 - x["Saving Rate"]) * x["Income"]).sum()).values
        )

        # set up optimisation

        homogeneous_weights = np.outer(iot_hh_consumption_norm, np.ones(n_quantiles))
        consumption_difference = (
            lambda alpha: np.dot(
                homogeneous_weights * alpha + weights_by_income * (1 - alpha), disposable_income_by_quantile
            )
            / (1 + vat)
            - iot_hh_consumption
        )

        # cost_fct = lambda alpha: np.sum((consumption_difference(alpha) / iot_hh_consumption.sum()) ** 2)
        def cost_fct(alpha):
            diff = consumption_difference(alpha)
            return np.dot(diff, diff) / iot_hh_consumption.sum()

        initial_guess = np.zeros(n_quantiles)  # Initial guess for alphas
        bounds = [(0, 1 - consumption_variance)] * n_quantiles  # Each alpha[q] is between 0 and 1-consumption_variance

        result = minimize(
            cost_fct,
            initial_guess,
            bounds=bounds,
            method="SLSQP",
        )

        # Apply the correction factors
        alphas_solution = result.x
        corrected_weights = homogeneous_weights * alphas_solution + weights_by_income * (1 - alphas_solution)
        # Set the consumption weights
        # note that the weights are transposed, [quantile, industry]

        # TODO in practice this doesn't change much; weights seem to always saturate close to 1, except for poor qs
        self.consumption_weights_by_income = corrected_weights.T


def split_array(array_to_split: np.ndarray) -> list[list[int]]:
    """
    Splits an array into subarrays based on the values in the input array.

    Args:
        array_to_split (np.ndarray): The array to be split.

    Returns:
        list[list[int]]: A list of subarrays, where each subarray contains consecutive elements with the same value.

    Example:
        >>> array = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
        >>> split_array(array)
        [[1, 1], [2, 2, 2], [3, 3, 3, 3]]
    """
    result = []
    current_sum = 0
    for num in array_to_split:
        result.append(list(range(current_sum, current_sum + num)))
        current_sum += num
    return result


def sample_households(
    hfcs_households_data: pd.DataFrame, hfcs_individuals_data: pd.DataFrame, n_households: int
) -> (pd.DataFrame, pd.DataFrame):
    """
    This function samples a specified number of households from the given HFCS households data,
    and also selects the corresponding individuals from the HFCS individuals data.

    Households are first sampled with replacement. Corresponding individuals are selected, and can be duplicated if
    their corresponding household was sampled more than once.

    Parameters:
    hfcs_households_data (pd.DataFrame): A DataFrame containing HFCS households data.
    hfcs_individuals_data (pd.DataFrame): A DataFrame containing HFCS individuals data.
    n_households (int): The number of households to sample.

    Returns:
    tuple: A tuple containing two DataFrames. The first DataFrame contains the selected households
    and the second DataFrame contains the individuals corresponding to the selected households.

    The function works as follows:
    1. Randomly selects a number of households from the hfcs_households_data DataFrame.
    2. Selects the corresponding individuals from the hfcs_individuals_data DataFrame.
    3. Adds the list of individuals per household to the selected households DataFrame.
    4. Resets the indices of the returned DataFrames for consistency.
    """
    # Step 1: Sample households with replacement
    weights = hfcs_households_data["Weight"] / hfcs_households_data["Weight"].sum()
    sampled_household_indices = np.random.choice(hfcs_households_data.index, n_households, p=weights, replace=True)

    # Create a list of DataFrames for each sampled household with a new unique ID
    household_dataframes = []
    individual_dataframes = []
    new_individual_id = 0
    for new_household_id, old_household_id in enumerate(sampled_household_indices):
        # Add household with new ID
        household = hfcs_households_data.loc[old_household_id].copy()
        household["New Household ID"] = new_household_id
        household_dataframes.append(household)

        # Duplicate corresponding individuals and assign new IDs
        individuals = hfcs_individuals_data[
            hfcs_individuals_data["Corresponding Household ID"] == old_household_id
        ].copy()
        individuals["New Household ID"] = new_household_id
        individuals["New Individual ID"] = range(new_individual_id, new_individual_id + len(individuals))
        new_individual_id += len(individuals)
        individual_dataframes.append(individuals)

    # Concatenate the lists into single DataFrames
    household_selection = pd.DataFrame(household_dataframes).reset_index(drop=True)
    individual_selection = pd.concat(individual_dataframes).reset_index(drop=True)

    # Step 4: Update the household_selection DataFrame
    corresponding_individuals = individual_selection.groupby("New Household ID")["New Individual ID"].apply(list)
    household_selection.set_index("New Household ID", inplace=True)
    household_selection["Corresponding Individuals ID"] = corresponding_individuals

    individual_selection["Corresponding Household ID"] = individual_selection["New Household ID"]

    return household_selection, individual_selection


def default_desired_consumption(
    income_: np.ndarray,
    consumption_weights_: np.ndarray,
    saving_rates_: np.ndarray,
    tau_vat_: float,
) -> np.ndarray:
    return 1.0 / (1 + tau_vat_) * np.outer(consumption_weights_, (1 - saving_rates_) * income_).T


def default_target_investment(
    income_: np.ndarray,
    investment_weights_: np.ndarray,
    investment_rate: np.ndarray,
    tau_cf_: float,
) -> np.ndarray:
    return 1.0 / (1 + tau_cf_) * np.outer(investment_weights_, investment_rate * income_).T
