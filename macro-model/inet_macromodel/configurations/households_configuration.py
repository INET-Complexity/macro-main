from pydantic import BaseModel
from typing import Any, Optional, Union, Literal


class HouseholdsParameters(BaseModel):
    consumption_weights_by_income_quantile: bool = False


class FinancialAssetsFunction(BaseModel):
    """
    The function used to set household income from financial assets.
    """

    path_name: str = "financial_assets"
    name: Literal["ConstantFinancialAssets", "DefaultFinancialAssets"] = "ConstantFinancialAssets"
    parameters: dict[str, Any] = {}


class ConsumptionFunction(BaseModel):
    """
    The function used to set household consumption.
    """

    path_name: str = "consumption"
    name: Literal["DefaultHouseholdConsumption"] = "DefaultHouseholdConsumption"
    parameters: dict[str, Any] = {
        "consumption_smoothing_fraction": 0.0,
        "consumption_smoothing_window": 12,
    }


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
        "maximum_price_noise_std": 0.3,
        "probability_stay_in_owned_property": 0.5,
        "probability_stay_in_rented_property": 0.5,
        "psychological_pressure_of_renting": 0.1,
        "rental_yield_btl_temperature": 1.0,
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
        "consumption_expansion_quantile": 0.0,
    }


class WealthFunction(BaseModel):
    """
    The function used for updating household wealth.
    """

    path_name: str = "wealth"
    name: Literal["DefaultWealthSetter"] = "DefaultWealthSetter"
    parameters: dict[str, Any] = {
        "other_real_assets_depreciation_rate": 0.05,
        "independents": ["Income", "Debt"],
    }


class SocialTransfersFunction(BaseModel):
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


class HouseholdsConfiguration(BaseModel):
    """
    Configuration for households.
    """

    functions: HouseholdsFunctions = HouseholdsFunctions()
    parameters: HouseholdsParameters = HouseholdsParameters()
    use_consumption_weights_by_income: bool = False
