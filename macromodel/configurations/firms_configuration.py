from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


def create_good_bundle(n_industries: int, bundles: Optional[list[list[int]]] = None) -> list:
    """Assign bundle indices to industries based on substitution groups.

    For a given number of industries, assign each industry to a bundle index.
    Industries listed together in a bundle share the same index. Industries not
    listed in any bundle are assigned unique bundle indices individually.

    After assignment, bundle indices are relabeled to ensure dense, increasing
    numbering based on first appearance.

    Args:
        n_industries (int): Total number of industries.
        bundles (List[List[int]]): List of substitution bundles, where each
            bundle is a list of industry indices.

    Returns:
        np.ndarray: Array of shape (n_industries,) mapping each industry to its bundle index.
    """
    if bundles is None:
        bundles = []

    good_bundle = [-1] * n_industries
    bundle_idx = 0

    # Assign bundle indices to industries included in bundles
    for bundle in bundles:
        for industry in bundle:
            good_bundle[industry] = bundle_idx
        bundle_idx += 1

    # Assign remaining industries that are not in any bundle
    for i in range(n_industries):
        if good_bundle[i] == -1:
            good_bundle[i] = bundle_idx
            bundle_idx += 1

    # Relabel to ensure increasing order
    seen = {}
    new_labels = []
    for x in good_bundle:
        if x not in seen:
            seen[x] = len(seen)
        new_labels.append(seen[x])

    good_bundle = new_labels

    return good_bundle


DEFAULT_BUNDLE = create_good_bundle(n_industries=18)


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

    name: Literal["PureLeontief", "CriticalAndImportantLeontief", "CriticalLeontief", "Linear", "BundledLeontief"] = (
        "PureLeontief"
    )
    path_name: str = "production"
    parameters: dict[str, Any] = {}


class TargetCapitalInputs(BaseModel):
    """
    The function used to setting firms desired capital inputs.
    Options: UnconstrainedTargetCapitalInputsSetter, FinancialTargetCapitalInputsSetter
    """

    name: Literal["FinancialTargetCapitalInputsSetter", "BundleWeightedTargetCapitalInputsSetter"] = (
        "FinancialTargetCapitalInputsSetter"
    )
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

    name: Literal["FinancialTargetIntermediateInputsSetter", "BundleWeightedTargetIntermediateInputsSetter"] = (
        "FinancialTargetIntermediateInputsSetter"
    )
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


class ProductivityGrowth(BaseModel):
    """
    The function for computing TFP growth.
    Options: NoOpTFPGrowth, SimpleTFPGrowth, StochasticTFPGrowth, SectoralTFPGrowth
    """

    name: Literal["NoOpTFPGrowth", "SimpleTFPGrowth", "StochasticTFPGrowth", "SectoralTFPGrowth"] = "NoOpTFPGrowth"
    path_name: str = "productivity_growth"
    parameters: dict[str, Any] = {}


class ProductivityInvestmentPlanner(BaseModel):
    """
    The function for planning productivity investments.
    Options: NoProductivityInvestmentPlanner, SimpleProductivityInvestmentPlanner, OptimalProductivityInvestmentPlanner

    All parameters can be specified as either:
    - float: uniform value applied to all firms
    - list[float]: heterogeneous values, one per firm (length must match n_firms)

    Note: n_firms is automatically injected during instantiation and does not need to be specified here.
    """

    name: Literal[
        "NoProductivityInvestmentPlanner", "SimpleProductivityInvestmentPlanner", "OptimalProductivityInvestmentPlanner"
    ] = "NoProductivityInvestmentPlanner"
    path_name: str = "productivity_investment_planner"
    parameters: dict[str, Any] = {
        "n_firms": 18,  # Number of firms (default for aggregate model)
        # TFP parameters (existing)
        "hurdle_rate": 0.15,  # Can be float or list[float]
        "max_investment_fraction": 0.1,  # Can be float or list[float]
        "investment_effectiveness": 0.1,  # Can be float or list[float]
        "investment_elasticity": 0.3,  # Can be float or list[float]
        # Technical coefficient parameters (new)
        "tfp_investment_share": 0.4,  # Can be float or list[float]
        "technical_investment_effectiveness": 0.15,  # Can be float or list[float]
        "technical_diminishing_returns": 0.5,  # Can be float or list[float]
        "price_weight": 0.4,  # Can be float or list[float]
        "usage_weight": 0.3,  # Can be float or list[float]
        "potential_weight": 0.3,  # Can be float or list[float]
    }


class TechnicalCoefficientsGrowth(BaseModel):
    """
    The function for computing technical coefficients growth.
    Options: NoOpTechnicalGrowth, SimpleTechnicalGrowth
    """

    name: Literal["NoOpTechnicalGrowth", "SimpleTechnicalGrowth"] = "NoOpTechnicalGrowth"
    path_name: str = "technical_coefficients_growth"
    parameters: dict[str, Any] = {
        "investment_effectiveness": 0.15,
        "diminishing_returns_factor": 0.5,
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
    productivity_growth: ProductivityGrowth = ProductivityGrowth()  # Defaults to NoOpTFPGrowth
    productivity_investment_planner: ProductivityInvestmentPlanner = (
        ProductivityInvestmentPlanner()
    )  # Defaults to NoProductivityInvestmentPlanner
    target_capital_inputs: TargetCapitalInputs = TargetCapitalInputs()
    target_credit: TargetCredit = TargetCredit()
    target_intermediate_inputs: TargetIntermediateInputs = TargetIntermediateInputs()
    wage_setter: WageSetter = WageSetter()
    target_production: TargetProduction = TargetProduction()
    excess_demand: ExcessDemand = ExcessDemand()
    labour_productivity: LabourProductivity = LabourProductivity()
    profit_estimator: ProfitEstimator = ProfitEstimator()
    technical_coefficients_growth: TechnicalCoefficientsGrowth = TechnicalCoefficientsGrowth()


class FirmsParameters(BaseModel):
    """Parameters for firm behavior configuration.

    Defines operational parameters that control firm production and investment through:
    - Capital input timing and depreciation
    - Resource utilization rates
    - Production capacity constraints
    - Investment behavior settings

    Attributes:
        capital_inputs_delay (list[int]): Delays in capital input availability by firm
        depreciation_rates (list[float]): Asset depreciation rates by firm
        capital_inputs_utilisation_rate (float): Capacity utilization for capital
        intermediate_inputs_utilisation_rate (float): Capacity utilization for inputs

    Note:
        The length of capital_inputs_delay and depreciation_rates determines n_firms.
    """

    capital_inputs_delay: list[int] = [0 for _ in range(18)]
    depreciation_rates: list[float] = [0.0 for _ in range(18)]
    capital_inputs_utilisation_rate: float = Field(1.0, ge=0.0, le=1.0)
    intermediate_inputs_utilisation_rate: float = Field(1.0, ge=0.0, le=1.0)
    tfp_base_growth_rate: float = Field(0.0025, ge=0.0, le=0.1, description="Base TFP growth rate (quarterly)")
    tfp_investment_elasticity: float = Field(0.3, ge=0.0, le=1.0, description="Returns to scale for TFP investment")

    # Productivity investment allocation parameters
    max_productivity_investment_fraction: float = Field(
        0.15, ge=0.0, le=1.0, description="Max productivity investment as fraction of output value"
    )
    max_productivity_cash_fraction: float = Field(
        0.3, ge=0.0, le=1.0, description="Max productivity investment as fraction of available cash"
    )
    tfp_investment_share: float = Field(
        0.4, ge=0.0, le=1.0, description="Share of productivity budget allocated to TFP (vs technical)"
    )

    # Technical coefficient investment parameters
    technical_investment_effectiveness: float = Field(
        0.15, ge=0.0, le=1.0, description="Effectiveness of technical coefficient investment"
    )
    technical_diminishing_returns: float = Field(
        0.5, ge=0.0, le=2.0, description="Diminishing returns factor for technical improvements"
    )

    # Investment targeting weights
    price_weight: float = Field(
        0.4, ge=0.0, le=1.0, description="Weight for price-based targeting in technical investment"
    )
    usage_weight: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight for usage-based targeting in technical investment"
    )
    potential_weight: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight for improvement potential in technical investment"
    )

    # Bundle arbitrage parameters
    enable_bundle_arbitrage: bool = Field(True, description="Enable bundle-aware arbitrage in technical investment")
    bundle_significance_threshold: float = Field(
        0.1, ge=0.0, le=1.0, description="Min fraction of spending for bundle arbitrage to apply"
    )
    arbitrage_intensity: float = Field(2.0, ge=0.0, le=5.0, description="Intensity of bundle arbitrage effects")

    @classmethod
    def disaggregated_industries_default(
        cls,
        n_industries: int,
        tfp_base_growth_rate: float = 0.0025,
        tfp_investment_elasticity: float = 0.3,
        max_productivity_investment_fraction: float = 0.15,
        max_productivity_cash_fraction: float = 0.3,
        tfp_investment_share: float = 0.4,
        technical_investment_effectiveness: float = 0.15,
        technical_diminishing_returns: float = 0.5,
        price_weight: float = 0.4,
        usage_weight: float = 0.3,
        potential_weight: float = 0.3,
        enable_bundle_arbitrage: bool = False,
        arbitrage_intensity: float = 2.0,
    ) -> "FirmsParameters":
        return cls(
            **{
                "capital_inputs_delay": [0 for _ in range(n_industries)],
                "depreciation_rates": [0.0 for _ in range(n_industries)],
                "capital_inputs_utilisation_rate": 1.0,
                "intermediate_inputs_utilisation_rate": 1.0,
                "tfp_base_growth_rate": tfp_base_growth_rate,
                "tfp_investment_elasticity": tfp_investment_elasticity,
                # Use defaults for new productivity investment parameters
                "max_productivity_investment_fraction": max_productivity_investment_fraction,
                "max_productivity_cash_fraction": max_productivity_cash_fraction,
                "tfp_investment_share": tfp_investment_share,
                "technical_investment_effectiveness": technical_investment_effectiveness,
                "technical_diminishing_returns": technical_diminishing_returns,
                "price_weight": price_weight,
                "usage_weight": usage_weight,
                "potential_weight": potential_weight,
                "enable_bundle_arbitrage": enable_bundle_arbitrage,
                "arbitrage_intensity": arbitrage_intensity,
            }
        )


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

    parameters: FirmsParameters = FirmsParameters.disaggregated_industries_default(n_industries=18)
    functions: FirmsFunctions = FirmsFunctions()
    calculate_hill_exponent: bool = True
    substitution_bundles: list = DEFAULT_BUNDLE

    @property
    def n_firms(self) -> int:
        """Get number of firms from parameters.

        The number of firms is inferred from the length of capital_inputs_delay
        or depreciation_rates lists in FirmsParameters.

        Returns:
            int: Number of firms configured
        """
        return len(self.parameters.capital_inputs_delay)

    @model_validator(mode="after")
    def validate_n_firms_consistency(self) -> "FirmsConfiguration":
        """Validate that n_firms is consistent across all parameters that use it."""
        # Get n_firms from parameters
        n_firms_from_delay = len(self.parameters.capital_inputs_delay)
        n_firms_from_depreciation = len(self.parameters.depreciation_rates)

        # Ensure capital_inputs_delay and depreciation_rates have same length
        if n_firms_from_delay != n_firms_from_depreciation:
            raise ValueError(
                f"capital_inputs_delay length ({n_firms_from_delay}) must match "
                f"depreciation_rates length ({n_firms_from_depreciation})"
            )

        n_firms = n_firms_from_delay

        # Always set n_firms in productivity investment planner (even if not currently active)
        # This ensures it's correct if the user later activates the planner
        self.functions.productivity_investment_planner.parameters["n_firms"] = n_firms

        # Only validate list parameter lengths if planner is active
        if self.functions.productivity_investment_planner.name != "NoProductivityInvestmentPlanner":
            # Validate that list parameters in productivity investment planner have correct length
            for param_name, param_value in self.functions.productivity_investment_planner.parameters.items():
                if param_name == "n_firms":
                    continue
                if isinstance(param_value, list):
                    if len(param_value) != n_firms:
                        raise ValueError(
                            f"Productivity investment parameter '{param_name}' has length {len(param_value)} "
                            f"but n_firms is {n_firms}. "
                            f"When using list parameters, they must match the number of firms."
                        )

        return self

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
    def n_industries_default(
        cls,
        n_industries: int,
        bundles: Optional[list[list[int]]] = None,
        tfp_base_growth_rate: float = 0.0025,
        tfp_investment_elasticity: float = 0.3,
        enable_bundle_arbitrage: bool = False,
    ) -> "FirmsConfiguration":
        if bundles is None:
            bundles = []

        if len(bundles) == 0 and enable_bundle_arbitrage:
            raise ValueError("Bundle arbitrage cannot be enabled without defining bundles.")

        if len(bundles) > 0:
            functions = FirmsFunctions(
                target_capital_inputs=TargetCapitalInputs(name="BundleWeightedTargetCapitalInputsSetter"),
                target_intermediate_inputs=TargetIntermediateInputs(
                    name="BundleWeightedTargetIntermediateInputsSetter"
                ),
                production=Production(name="BundledLeontief"),
            )
        else:
            functions = FirmsFunctions()

        bundles_grouped = create_good_bundle(n_industries=n_industries, bundles=bundles)
        return cls(
            parameters=FirmsParameters.disaggregated_industries_default(
                n_industries,
                tfp_base_growth_rate=tfp_base_growth_rate,
                tfp_investment_elasticity=tfp_investment_elasticity,
            ),
            functions=functions,
            calculate_hill_exponent=True,
            substitution_bundles=bundles_grouped,
        )
