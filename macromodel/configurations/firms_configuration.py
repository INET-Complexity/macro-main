from typing import Any, Literal

from pydantic import BaseModel, Field


class BoughtGoodsDistributor(BaseModel):
    """
    The function used by firms to distribute bought goods between intermediate inputs and capital goods.
    Options: BoughtGoodsDistributorEvenly, BoughtGoodsDistributorIIPrio
    """

    name: Literal["BoughtGoodsDistributorEvenly", "BoughtGoodsDistributorIIPrio"] = "BoughtGoodsDistributorIIPrio"
    path_name: str = "bought_goods_distributor"
    parameters: dict[str, Any] = {}


class ExcessDemand(BaseModel):
    """
    The function used by firms to calculate their excess demand for goods.
    Options: DefaultExcessDemandSetter, ZeroExcessDemandSetter
    """

    name: Literal["ConstrainedExcessDemandSetter",] = "ConstrainedExcessDemandSetter"
    path_name: str = "excess_demand"
    parameters: dict[str, Any] = {
        "consider_intermediate_inputs": 0.0,
        "consider_capital_inputs": 1.0,
        "consider_labour_inputs": 0.0,
    }


class LabourProductivity(BaseModel):
    """
    The function used to calculate the labour productivity of firms.
    Options: WorkEffortLabourProductivitySetter
    """

    name: Literal["WorkEffortLabourProductivitySetter"] = "WorkEffortLabourProductivitySetter"
    path_name: str = "labour_productivity"
    parameters: dict[str, Any] = {
        "max_increase_in_work_effort": 1.5,
        "consider_intermediate_inputs": True,
        "consider_capital_inputs": True,
        "work_effort_increase_speed": 1.0,
    }


class DemandEstimator(BaseModel):
    """
    The function used by firms to estimate their future demand for goods.
    Options: DefaultDemandEstimator
    """

    name: Literal["DefaultDemandEstimator"] = "DefaultDemandEstimator"
    path_name: str = "demand_estimator"
    parameters: dict[str, Any] = {
        "firm_growth_adjustment_speed": 0.0,
        "sectoral_growth_adjustment_speed": 0.0,
    }


class ProfitEstimator(BaseModel):
    """
    The function used by firms to estimate their future profits.
    Options: DefaultProfitEstimator
    """

    name: Literal["DefaultFirmProfitsSetter"] = "DefaultFirmProfitsSetter"
    path_name: str = "profit_estimator"
    parameters: dict[str, Any] = {}


class DemandForGoods(BaseModel):
    """
    The function used by firms to calculate their demand for goods.
    Options: DefaultDemandSetter, DemandExcessSetter
    """

    name: Literal["DefaultDemandSetter"] = "DefaultDemandSetter"
    path_name: str = "demand_for_goods"
    parameters: dict[str, Any] = {}


class Demography(BaseModel):
    """
    The function used for handling firm entry and exit.
    Options: NoFirmDemography, DefaultFirmDemography
    """

    name: Literal["NoFirmDemography", "DefaultFirmDemography"] = "DefaultFirmDemography"
    path_name: str = "demography"
    parameters: dict[str, Any] = {}


class DesiredLabour(BaseModel):
    """
    The function used to calculate the desired amount of labour for each firm.
    Options: DefaultDesiredLabourSetter
    """

    name: Literal["DefaultDesiredLabourSetter"] = "DefaultDesiredLabourSetter"
    path_name: str = "desired_labour"
    parameters: dict[str, Any] = {
        "consider_intermediate_inputs": False,
        "consider_capital_inputs": 1.0,
    }


class GrowthEstimator(BaseModel):
    """
    The function used to estimate growth for each firm.
    Options: ZeroGrowthEstimator, DefaultGrowthEstimator
    """

    name: Literal["ZeroGrowthEstimator", "DefaultGrowthEstimator"] = "DefaultGrowthEstimator"
    path_name: str = "growth_estimator"
    parameters: dict[str, Any] = {}


class OfferedWageSetter(BaseModel):
    """
    The function used to set the offered wage to individuals by each firm.
    Options: DefaultOfferedWageSetter
    """

    name: Literal["DefaultOfferedWageSetter"] = "DefaultOfferedWageSetter"
    path_name: str = "offered_wage_setter"
    parameters: dict[str, Any] = {"labour_market_tightness_markup_scale": 0.5, "markup_time_span": 4}


class Prices(BaseModel):
    """
    The function used to set prices.
    Options: ConstantPriceSetter, SupplyDemandPriceSetter, CANVASPriceSetter
    """

    name: Literal["DefaultPriceSetter", "ExogenousPriceSetter"] = "DefaultPriceSetter"
    path_name: str = "prices"
    parameters: dict[str, Any] = {
        "price_setting_noise_std": 0.05,
        "price_setting_speed_gf": 1.0,
        "price_setting_speed_dp": 0.0,
        "price_setting_speed_cp": 0.0,
    }


class Production(BaseModel):
    """
    The function used to produce goods.
    Options: PureLeontief, CriticalAndImportantLeontief, CriticalLeontief, Linear
    """

    name: Literal["PureLeontief", "CriticalAndImportantLeontief", "CriticalLeontief", "Linear"] = "PureLeontief"
    path_name: str = "production"
    parameters: dict[str, Any] = {}


class TargetCapitalInputs(BaseModel):
    """
    The function used to setting firms desired capital inputs.
    Options: UnconstrainedTargetCapitalInputsSetter, FinancialTargetCapitalInputsSetter
    """

    name: Literal["FinancialTargetCapitalInputsSetter"] = "FinancialTargetCapitalInputsSetter"
    path_name: str = "target_capital_inputs"
    parameters: dict[str, Any] = {"target_capital_inputs_fraction": 0.0, "credit_gap_fraction": 0.0}


class TargetCredit(BaseModel):
    """
    The function for setting the firms target loans.
    Options: DefaultTargetCreditSetter
    """

    name: Literal["DefaultTargetCreditSetter", "SimpleTargetCreditSetter"] = "DefaultTargetCreditSetter"
    path_name: str = "target_credit"
    parameters: dict[str, Any] = {}


class TargetIntermediateInputs(BaseModel):
    """
    The function for setting the firms target intermediate inputs.
    Options: UnconstrainedTargetIntermediateInputsSetter, FinancialTargetIntermediateInputsSetter
    """

    name: Literal["FinancialTargetIntermediateInputsSetter"] = "FinancialTargetIntermediateInputsSetter"
    path_name: str = "target_intermediate_inputs"
    parameters: dict[str, Any] = {"target_intermediate_inputs_fraction": 0.0, "credit_gap_fraction": 0.0}


class WageSetter(BaseModel):
    """
    The function for setting the wages paid to employed individuals.
    Options: DefaultFirmWageSetter
    """

    name: Literal["WorkEffortFirmWageSetter"] = "WorkEffortFirmWageSetter"
    path_name: str = "wage_setter"
    parameters: dict[str, Any] = {
        "labour_market_tightness_markup_scale": 0.0,
        "markup_time_span": 4,
        "max_increase_in_work_effort": 1.5,
    }


class TargetProduction(BaseModel):
    """
    The function for setting the firms target production.
    Options: DefaultTargetProductionSetter
    """

    name: Literal["DefaultTargetProductionSetter"] = "DefaultTargetProductionSetter"
    path_name: str = "target_production"
    parameters: dict[str, Any] = {
        "existing_inventory_fraction": 0.0,
        "maximum_debt_to_equity_ratio": 2.0,
        "target_inventory_to_production_fraction": 0.0,
        "financial_constrains_fraction": 0.0,
        "intermediate_inputs_target_considers_labour_inputs": 0.0,
        "intermediate_inputs_target_considers_intermediate_inputs": 0.0,
        "intermediate_inputs_target_considers_capital_inputs": 1.0,
        "capital_inputs_target_considers_labour_inputs": 0.0,
        "capital_inputs_target_considers_intermediate_inputs": 0.0,
        "capital_inputs_target_considers_capital_inputs": 1.0,
    }


class FirmsFunctions(BaseModel):
    bought_goods_distributor: BoughtGoodsDistributor = BoughtGoodsDistributor()
    demand_estimator: DemandEstimator = DemandEstimator()
    demand_for_goods: DemandForGoods = DemandForGoods()
    demography: Demography = Demography()
    desired_labour: DesiredLabour = DesiredLabour()
    growth_estimator: GrowthEstimator = GrowthEstimator()
    offered_wage_setter: OfferedWageSetter = OfferedWageSetter()
    prices: Prices = Prices()
    production: Production = Production()
    target_capital_inputs: TargetCapitalInputs = TargetCapitalInputs()
    target_credit: TargetCredit = TargetCredit()
    target_intermediate_inputs: TargetIntermediateInputs = TargetIntermediateInputs()
    wage_setter: WageSetter = WageSetter()
    target_production: TargetProduction = TargetProduction()
    excess_demand: ExcessDemand = ExcessDemand()
    labour_productivity: LabourProductivity = LabourProductivity()
    profit_estimator: ProfitEstimator = ProfitEstimator()


class FirmsParameters(BaseModel):
    """Parameters for firm behavior configuration.

    Defines operational parameters that control firm production and investment through:
    - Capital input timing and depreciation
    - Resource utilization rates
    - Production capacity constraints
    - Investment behavior settings

    Attributes:
        capital_inputs_delay (list[int]): Delays in capital input availability by industry
        depreciation_rates (list[float]): Asset depreciation rates by industry
        capital_inputs_utilisation_rate (float): Capacity utilization for capital
        intermediate_inputs_utilisation_rate (float): Capacity utilization for inputs
    """

    capital_inputs_delay: list[int] = [0 for _ in range(18)]
    depreciation_rates: list[float] = [0.0 for _ in range(18)]
    capital_inputs_utilisation_rate: float = Field(1.0, ge=0.0, le=1.0)
    intermediate_inputs_utilisation_rate: float = Field(1.0, ge=0.0, le=1.0)

    @classmethod
    def disaggregated_industries_default(cls, n_industries: int):
        return {
            "capital_inputs_delay": [0 for _ in range(n_industries)],
            "depreciation_rates": [0.0 for _ in range(n_industries)],
            "capital_inputs_utilisation_rate": 1.0,
            "intermediate_inputs_utilisation_rate": 1.0,
        }


class FirmsConfiguration(BaseModel):
    """Configuration for firm behavior and operations.

    Defines the complete configuration for firms through:
    - Operational parameters
    - Functional components
    - Calculation settings

    Attributes:
        parameters (FirmsParameters): Operational parameter settings
        functions (FirmsFunctions): Function implementations
        calculate_hill_exponent (bool): Whether to calculate Hill exponent
    """

    parameters: FirmsParameters = FirmsParameters()
    functions: FirmsFunctions = FirmsFunctions()
    calculate_hill_exponent: bool = True

    @property
    def reset_params(self):
        inventory_frac = self.functions.target_production.parameters["existing_inventory_fraction"]
        values = {
            "capital_inputs_utilisation_rate": self.parameters.capital_inputs_utilisation_rate,
            "intermediate_inputs_utilisation_rate": self.parameters.intermediate_inputs_utilisation_rate,
            "initial_inventory_to_input_fraction": inventory_frac,
        }
        return values

    @classmethod
    def n_industries_default(cls, n_industries: int):
        return cls(
            parameters=FirmsParameters.disaggregated_industries_default(n_industries),
            functions=FirmsFunctions(),
            calculate_hill_exponent=True,
        )
