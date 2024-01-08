from typing import Any, Literal

from pydantic import BaseModel


class DemographyFunction(BaseModel):
    """
    The function for setting individual demography.
    """

    path_name: str = "demography"
    name: Literal["NoAging"] = "NoAging"
    parameters: dict[str, Any] = {}


class IncomeFunction(BaseModel):
    """
    The function for setting individual income.
    """

    path_name: str = "income"
    name: Literal["DefaultIncomeSetter"] = "DefaultIncomeSetter"
    parameters: dict[str, Any] = {}


class LabourInputsFunction(BaseModel):
    """
    The function for updating individual productivity.
    """

    path_name: str = "labour_inputs"
    name: Literal[
        "ScaledIndividualsProductivitySetter", "ConstantIndividualsLabourInputsSetter"
    ] = "ScaledIndividualsProductivitySetter"
    parameters: dict[str, Any] = {"decrease_unemployed": 0.0, "increase_employed": 0.0}


class ReservationWagesFunction(BaseModel):
    """
    The function for setting individual reservation wages.
    """

    path_name: str = "reservation_wages"
    name: Literal["DefaultReservationWageSetter"] = "DefaultReservationWageSetter"
    parameters: dict[str, Any] = {"unemployed_reservation_wage_timespan": 0.0}


class IndividualsFunctions(BaseModel):
    """
    The functions used by individuals.
    """

    demography: DemographyFunction = DemographyFunction()
    income: IncomeFunction = IncomeFunction()
    labour_inputs: LabourInputsFunction = LabourInputsFunction()
    reservation_wages: ReservationWagesFunction = ReservationWagesFunction()


class IndividualsConfiguration(BaseModel):
    """
    Represents the configuration of individuals.

    Attributes:
        functions (IndividualsFunctions): The functions used by individuals.
    """

    functions: IndividualsFunctions = IndividualsFunctions()
