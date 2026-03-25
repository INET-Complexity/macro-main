from typing import Any, Literal, Optional

from pydantic import BaseModel


def create_household_bundle(n_industries: int, bundles: Optional[list[list[int]]] = None) -> list:
    """Assign bundle indices to industries based on substitution groups for households.

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
        list: Array of length n_industries mapping each industry to its bundle index.
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


DEFAULT_HOUSEHOLD_BUNDLE = create_household_bundle(n_industries=18)


class HouseholdsParameters(BaseModel):
    """Parameters for household behavior configuration.

    Defines behavioral parameters that control household economic decisions through:
    - Consumption pattern determination
    - Income-based consumption weights
    - Behavioral preference settings

    Attributes:
        take_consumption_weights_by_income_quantile (bool): Whether to use income-based
            consumption weights. Defaults to False.
    """

    take_consumption_weights_by_income_quantile: bool = False


class FinancialAssetsFunction(BaseModel):
    """
    The function used to set household income from financial assets.
    """

    path_name: str = "financial_assets"
    name: Literal["DefaultFinancialAssets"] = "DefaultFinancialAssets"
    parameters: dict[str, Any] = {"income_from_fa_noise_std": 0.0}


class ConsumptionFunction(BaseModel):
    """
    The function used to set household consumption.
    """

    path_name: str = "consumption"
    name: Literal["DefaultHouseholdConsumption", "ExogenousHouseholdConsumption", "CESHouseholdConsumption"] = (
        "DefaultHouseholdConsumption"
    )
    parameters: dict[str, Any] = {
        "consumption_smoothing_fraction": 0.0,
        "consumption_smoothing_window": 12,
        "minimum_consumption_fraction": 1.0,
        "elasticity_of_substitution": 1.0,
    }


class InvestmentFunction(BaseModel):
    """
    The function used for setting household investment.
    """

    path_name: str = "investment"
    name: Literal["DefaultHouseholdInvestment", "NoHouseholdInvestment"] = "DefaultHouseholdInvestment"
    parameters: dict[str, Any] = {}


class InsolvencyFunction(BaseModel):
    """
    The function used for handling household insolvency
    """

    path_name: str = "insolvency"
    name: Literal["DefaultHouseholdInsolvencyHandler"] = "DefaultHouseholdInsolvencyHandler"
    parameters: dict[str, Any] = {}


class PropertyFunction(BaseModel):
    """
    The function used for calculating household demand for property.
    """

    path_name: str = "property"
    name: Literal["DefaultHouseholdDemandForProperty"] = "DefaultHouseholdDemandForProperty"
    parameters: dict[str, Any] = {
        "cost_comparison_temperature": 1.0,
        "maximum_price_income_coefficient": 5.0,
        "maximum_price_income_exponent": 1.0,
        "maximum_price_noise_variance": 0.3,
        "probability_stay_in_owned_property": 0.5,
        "probability_stay_in_rented_property": 0.5,
        "psychological_pressure_of_renting": 0.1,
        "price_initial_markup": 0.0,
        "price_decrease_probability": 0.0,
        "price_decrease_mean": 0.0,
        "price_decrease_variance": 0.0,
        "rent_initial_markup": 0.0,
        "rent_decrease_probability": 0.0,
        "rent_decrease_mean": 0.0,
        "rent_decrease_variance": 0.0,
        "maximum_price_noise_mean": 0.0,
        "maximum_rent_income_coefficient": 0.2,
        "maximum_rent_income_exponent": 0.0,
        "partial_rent_inflation_indexation": 0.0,
        "partial_rent_inflation_delay": 1,
    }


class SavingRatesFunction(BaseModel):
    """
    The function used to set household saving rates.
    """

    path_name: str = "saving_rates"
    name: Literal["AverageSavingRatesSetter", "ConstantSavingRatesSetter", "DefaultSavingRatesSetter"] = (
        "AverageSavingRatesSetter"
    )
    parameters: dict[str, Any] = {
        "independents": ["Income", "Debt"],
    }


class RentFunction(BaseModel):
    """
    The function used to set rent.
    """

    path_name: str = "rent"
    name: Literal["ConstantRentSetter", "DefaultRentSetter"] = "ConstantRentSetter"
    parameters: dict[str, Any] = {
        "new_property_rent_markup": 0.1,
        "offered_rent_decrease": 0.03,
        "partial_rent_inflation_indexation": 0.0,
    }


class TargetCreditFunction(BaseModel):
    """
    The function for setting household target credit.
    """

    path_name: str = "target_credit"
    name: Literal["DefaultHouseholdTargetCredit"] = "DefaultHouseholdTargetCredit"
    parameters: dict[str, Any] = {
        "down_payment_fraction": 1.0,
    }


class WealthFunction(BaseModel):
    """
    The function used for updating household wealth.
    """

    path_name: str = "wealth"
    name: Literal["DefaultWealthSetter"] = "DefaultWealthSetter"
    parameters: dict[str, Any] = {
        "other_real_assets_depreciation_rate": 0.05,
    }


class SocialTransfersFunction(BaseModel):
    """The function used to set household social transfer allocation.

    Defines how social transfers are distributed among households through:
    - Equal distribution mechanisms
    - Income-based allocation
    - Model-driven predictions
    - Household characteristics consideration

    Attributes:
        path_name (str): Module path for social transfer functions
        name (Literal): Selected transfer allocation strategy
        parameters (dict): Configuration parameters including independent variables
    """

    path_name: str = "social_transfers"
    name: Literal["EqualSocialTransfersSetter", "ConstantSocialTransfersSetter", "DefaultSocialTransfersSetter"] = (
        "EqualSocialTransfersSetter"
    )
    parameters: dict[str, Any] = {
        "independents": ["Income", "Debt"],
    }


class HouseholdsFunctions(BaseModel):
    """
    Wrapper for the functions used by households.
    """

    financial_assets: FinancialAssetsFunction = FinancialAssetsFunction()
    consumption: ConsumptionFunction = ConsumptionFunction()
    insolvency: InsolvencyFunction = InsolvencyFunction()
    property: PropertyFunction = PropertyFunction()
    rent: RentFunction = RentFunction()
    target_credit: TargetCreditFunction = TargetCreditFunction()
    wealth: WealthFunction = WealthFunction()
    social_transfers: SocialTransfersFunction = SocialTransfersFunction()
    saving_rates: SavingRatesFunction = SavingRatesFunction()
    investment: InvestmentFunction = InvestmentFunction()


class HouseholdsConfiguration(BaseModel):
    """
    Configuration for households.
    """

    functions: HouseholdsFunctions = HouseholdsFunctions()
    parameters: HouseholdsParameters = HouseholdsParameters()
    take_consumption_weights_by_income_quantile: bool = False
    substitution_bundles: list = DEFAULT_HOUSEHOLD_BUNDLE

    @classmethod
    def n_industries_default(cls, n_industries: int, bundles: Optional[list[list[int]]] = None):
        """Create households configuration with specified number of industries and substitution bundles.

        Args:
            n_industries (int): Number of industries in the economy
            bundles (Optional[list[list[int]]]): Substitution bundles for consumption.
                If provided, automatically configures CES consumption function.

        Returns:
            HouseholdsConfiguration: Configured instance with appropriate substitution settings
        """
        if bundles is None:
            bundles = []

        if len(bundles) > 0:
            functions = HouseholdsFunctions(consumption=ConsumptionFunction(name="CESHouseholdConsumption"))
        else:
            functions = HouseholdsFunctions()

        bundles_grouped = create_household_bundle(n_industries=n_industries, bundles=bundles)
        return cls(
            functions=functions,
            parameters=HouseholdsParameters(),
            take_consumption_weights_by_income_quantile=False,
            substitution_bundles=bundles_grouped,
        )
