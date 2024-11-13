from typing import Any, Literal

from pydantic import BaseModel


class HouseholdsParameters(BaseModel):
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
    name: Literal["DefaultHouseholdConsumption", "ExogenousHouseholdConsumption"] = "DefaultHouseholdConsumption"
    parameters: dict[str, Any] = {
        "consumption_smoothing_fraction": 0.0,
        "consumption_smoothing_window": 12,
        "minimum_consumption_fraction": 1.0,
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
