import numpy as np
import pandas as pd

from abc import abstractmethod, ABC

from data.readers.economic_data.ons_reader import ONSReader
from data.readers.economic_data.oecd_economic_data import OECDEconData
from data.readers.economic_data.exchange_rates import WorldBankRatesReader


class SyntheticFirms(ABC):
    @abstractmethod
    def __init__(
        self,
        country_name: str,
        scale: int,
        year: int,
        industries: list[str],
    ):
        self.country_name = country_name
        self.scale = scale
        self.year = year
        self.industries = industries

        # Firm data
        self.number_of_firms = None
        self.number_of_firms_by_industry = None
        self.firm_data = pd.DataFrame()
        self.intermediate_inputs_stock = None
        self.used_intermediate_inputs = None
        self.capital_inputs_stock = None
        self.used_capital_inputs = None
        self.employees = None

    @abstractmethod
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
        pass

    def create_agents(
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
        self.set_firm_sizes(
            number_of_employees_by_industry=number_of_employees_by_industry,
            econ_reader=econ_reader,
            ons_reader=ons_reader,
        )
        self.set_firm_wages(
            number_of_employees_by_industry=number_of_employees_by_industry,
            labour_compensation=industry_data["industry_vectors"]["Labour Compensation in LCU"].values,
            tau_sif=econ_reader.read_tau_sif(self.country_name, self.year),
        )
        self.set_firm_production(
            number_of_employees_by_industry=number_of_employees_by_industry,
            output=industry_data["industry_vectors"]["Output"].values,
        )
        self.set_firm_prices(exchange_rates=exchange_rates)
        self.set_firm_labour_inputs()
        self.set_firm_inventory(initial_inventory_to_production_fraction=initial_inventory_to_production_fraction)
        self.set_firm_demand()
        self.set_firm_intermediate_inputs_stock(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            initial_utilisation_rate=intermediate_inputs_utilisation_rate,
        )
        self.set_firm_used_intermediate_inputs(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values
        )
        self.set_firm_capital_inputs_stock(
            capital_inputs_productivity_matrix=industry_data["capital_inputs_productivity_matrix"].values,
            initial_utilisation_rate=capital_inputs_utilisation_rate,
        )
        self.set_firm_used_capital_inputs(
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
        )
        self.set_firm_deposits(
            total_firm_deposits=total_firm_deposits,
            assume_zero_initial_deposits=assume_zero_initial_deposits,
        )
        self.set_firm_debt(
            total_firm_debt=total_firm_debt,
            assume_zero_initial_debt=assume_zero_initial_debt,
        )
        self.set_firm_equity()

    @abstractmethod
    def set_industries(
        self,
        number_of_firms_by_industry: np.ndarray,
    ) -> None:
        pass

    def set_additional_initial_conditions(
        self,
        econ_reader: OECDEconData,
        industry_data: dict[str, pd.DataFrame],
        interest_rate_on_firm_deposits: np.ndarray,
        overdraft_rate_on_firm_deposits: np.ndarray,
        credit_market_data: pd.DataFrame,
    ) -> None:
        self.set_taxes_paid_on_production(
            taxes_less_subsidies_rates=industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values,
        )
        self.set_interest_paid(
            interest_rate_on_firm_deposits=interest_rate_on_firm_deposits,
            overdraft_rate_on_firm_deposits=overdraft_rate_on_firm_deposits,
            credit_market_data=credit_market_data,
        )
        self.set_firm_profits(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
            tau_sif=econ_reader.read_tau_sif(self.country_name, self.year),
        )
        self.set_unit_costs(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
            tau_sif=econ_reader.read_tau_sif(self.country_name, self.year),
        )
        self.set_corporate_taxes_paid(
            tau_firm=econ_reader.read_tau_firm(self.country_name, self.year),
        )
        self.set_firm_debt_installments(credit_market_data=credit_market_data)

    @abstractmethod
    def set_firm_sizes(
        self,
        number_of_employees_by_industry: np.ndarray,
        econ_reader: OECDEconData,
        ons_reader: ONSReader,
    ) -> None:
        pass

    @abstractmethod
    def set_firm_wages(
        self,
        number_of_employees_by_industry: np.ndarray,
        labour_compensation: np.ndarray,
        tau_sif: float,
    ) -> None:
        pass

    @abstractmethod
    def set_firm_production(
        self,
        number_of_employees_by_industry: np.ndarray,
        output: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def set_firm_prices(self, exchange_rates: WorldBankRatesReader) -> None:
        pass

    @abstractmethod
    def set_firm_demand(self) -> None:
        pass

    @abstractmethod
    def set_firm_labour_inputs(self) -> None:
        pass

    @abstractmethod
    def set_firm_inventory(self, initial_inventory_to_production_fraction: float) -> None:
        pass

    @abstractmethod
    def set_firm_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        initial_utilisation_rate: float,
    ) -> None:
        pass

    @abstractmethod
    def set_firm_used_intermediate_inputs(self, intermediate_inputs_productivity_matrix: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_firm_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        initial_utilisation_rate: float,
    ) -> None:
        pass

    @abstractmethod
    def set_firm_used_capital_inputs(self, capital_inputs_depreciation_matrix: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_firm_deposits(self, total_firm_deposits: float, assume_zero_initial_deposits: float) -> None:
        pass

    @abstractmethod
    def set_firm_debt(self, total_firm_debt: float, assume_zero_initial_debt: float) -> None:
        pass

    @abstractmethod
    def set_firm_equity(self) -> None:
        pass

    @abstractmethod
    def set_taxes_paid_on_production(
        self,
        taxes_less_subsidies_rates: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def set_interest_paid(
        self,
        interest_rate_on_firm_deposits: np.ndarray,
        overdraft_rate_on_firm_deposits: np.ndarray,
        credit_market_data: pd.DataFrame,
    ) -> None:
        pass

    @abstractmethod
    def set_firm_profits(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        tau_sif: float,
    ) -> None:
        pass

    @abstractmethod
    def set_unit_costs(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        tau_sif: float,
    ) -> None:
        pass

    @abstractmethod
    def set_corporate_taxes_paid(self, tau_firm: float) -> None:
        pass

    @abstractmethod
    def set_firm_debt_installments(self, credit_market_data: pd.DataFrame) -> None:
        pass
