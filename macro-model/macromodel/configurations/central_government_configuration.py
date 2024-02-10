from pydantic import BaseModel
from typing import Literal


class SocialBenefits(BaseModel):
    name: Literal["ConstantSocialBenefitsSetter", "DefaultSocialBenefitsSetter"] = "DefaultSocialBenefitsSetter"
    path_name: str = "social_benefits"
    parameters: dict = {}


class SocialHousing(BaseModel):
    name: Literal["DefaultSocialHousing"] = "DefaultSocialHousing"
    path_name: str = "social_housing"
    parameters: dict = {"rent_as_fraction_of_unemployment_rate": 0.25}


class CentralGovernmentFunctions(BaseModel):
    social_benefits: SocialBenefits = SocialBenefits()
    social_housing: SocialHousing = SocialHousing()


class CentralGovernmentConfiguration(BaseModel):
    functions: CentralGovernmentFunctions = CentralGovernmentFunctions()
