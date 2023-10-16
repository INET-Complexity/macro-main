import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from inet_data.readers.population_data.hfcs_reader import HFCSReader
from inet_data.readers.economic_data.oecd_economic_data import OECDEconData
from inet_data.readers.economic_data.world_bank_reader import WorldBankReader
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)

from inet_data.util.regressions import fit_linear
from inet_data.util.clean_data import remove_outliers

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from typing import Any


class SyntheticHFCSPopulation(SyntheticPopulation):
    def __init__(
        self,
        country_name: str,
        country_name_short: str,
        scale: int,
        year: int,
        industries: list[str],
    ):
        super().__init__(
            country_name,
            country_name_short,
            scale,
            year,
            industries,
        )

    def create(
        self,
        hfcs_reader: HFCSReader,
        econ_reader: OECDEconData,
        wb_reader: WorldBankReader,
        n_households: int,
        number_of_firms_by_industry: np.ndarray,
        total_unemployment_benefits: float,
        rent_as_fraction_of_unemployment_rate: float,
    ) -> None:
        # Fetch HFCS data
        hfcs_individuals_data = hfcs_reader.individuals_df
        hfcs_households_data = hfcs_reader.households_df

        # Create a mapping between households and individuals and vice-versa
        ind_hh_map = dict(
            zip(
                hfcs_individuals_data.index,
                hfcs_individuals_data["Corresponding Household ID"],
            )
        )
        hh_ind_map = defaultdict(list)
        for key, value in ind_hh_map.items():
            hh_ind_map[value].append(key)
        hh_ind_map = dict(hh_ind_map)

        # Draw households at random
        household_inds = np.random.choice(
            range(len(hfcs_households_data)),
            n_households,
            p=hfcs_households_data["Weight"] / hfcs_households_data["Weight"].sum(),
            replace=True,
        )

        # Compile new dataframes
        self.compile_new_dataframes(
            hfcs_households_data=hfcs_households_data,
            hfcs_individuals_data=hfcs_individuals_data,
            hh_ind_map=hh_ind_map,
            household_inds=household_inds,
        )

        # Remove outliers
        self.household_data = remove_outliers(
            data=self.household_data,
            cols=[
                "Rent Paid",
                "Income",
                "Consumption of Consumer Goods/Services as a Share of Income",
            ],
        )
        self.individual_data = remove_outliers(
            data=self.individual_data,
            cols=["Employee Income", "Gender", "Age", "Education", "Labour Status"],
        )

        # Create individuals and households
        self.create_individuals(
            econ_reader,
            wb_reader,
            number_of_firms_by_industry,
            total_unemployment_benefits,
        )
        self.create_households(
            rent_as_fraction_of_unemployment_rate=rent_as_fraction_of_unemployment_rate,
            unemployment_benefits_by_capita=total_unemployment_benefits
            / np.sum(self.individual_data["Activity Status"] == 2),
        )
        self.number_employees_by_industry = self.get_number_employees_by_industry()
        self.social_housing_rent = (
            rent_as_fraction_of_unemployment_rate
            * total_unemployment_benefits
            / np.sum(self.individual_data["Activity Status"] == 2)
        )

    def compile_new_dataframes(
        self,
        hfcs_households_data: pd.DataFrame,
        hfcs_individuals_data: pd.DataFrame,
        hh_ind_map: dict,
        household_inds: np.ndarray,
    ) -> None:
        household_inds_updated, hh_ind_map_updated = [], []
        ind_rows = []
        curr_ind_num = 0
        for i, household_ind in tqdm(
            enumerate(household_inds),
            desc="Compiling population data frames for " + self.country_name,
            total=len(household_inds),
        ):
            household_id = hfcs_households_data.index[household_ind]
            ind_row = hfcs_individuals_data.loc[hh_ind_map[household_id]]
            if len(ind_row) > 0:
                ind_row["Corresponding Household ID"] = i
                ind_rows.append(ind_row)
                household_inds_updated.append(household_id)
                hh_ind_map_updated.append(list(range(curr_ind_num, curr_ind_num + len(ind_row))))
                curr_ind_num += len(ind_row)
        household_inds_updated = np.array(household_inds_updated)
        self.individual_data = pd.concat(ind_rows, axis=0)
        self.individual_data.index = range(len(self.individual_data))
        self.household_data = hfcs_households_data.loc[household_inds_updated].copy()
        self.household_data.index = range(len(self.household_data))
        self.household_data["Corresponding Individuals ID"] = [
            hh_ind_map_updated[hid] for hid in self.household_data.index
        ]
        self.household_data.drop("Weight", axis=1, inplace=True)

    def create_individuals(
        self,
        econ_reader: OECDEconData,
        wb_reader: WorldBankReader,
        number_of_firms_by_industry: np.ndarray,
        total_unemployment_benefits: float,
    ) -> None:
        self.set_individual_gender()
        self.set_individual_age()
        self.set_individual_education()
        self.set_individual_labour_status()
        self.set_individual_activity_status(
            unemployment_rate=wb_reader.get_unemployment_rate(
                country=self.country_name,
                year=self.year,
            ),
            participation_rate=wb_reader.get_participation_rate(
                country=self.country_name,
                year=self.year,
            ),
        )
        self.set_individual_employment_nace()
        self.set_individual_employee_income(
            unemployment_benefits_by_individual=total_unemployment_benefits
            / np.sum(self.individual_data["Activity Status"] == 2),
        )
        self.set_individual_unemployed_income(
            unemployment_benefits_by_individual=total_unemployment_benefits
            / np.sum(self.individual_data["Activity Status"] == 2),
        )
        self.set_income()

        # Keep what we need
        self.individual_data = self.individual_data[
            [
                "Gender",
                "Age",
                "Education",
                "Activity Status",
                "Employment Industry",
                "Employee Income",
                "Income from Unemployment Benefits",
                "Income",
                "Corresponding Household ID",
            ]
        ]

    def set_individual_gender(self) -> None:
        # Proportionally fill-in gender
        missing_genders = self.individual_data["Gender"].isna()
        p_male = (self.individual_data["Gender"] == 1).mean()
        p_female = (self.individual_data["Gender"] == 2).mean()
        p_total = p_male + p_female
        p_male /= p_total
        p_female /= p_total
        self.individual_data.loc[missing_genders, "Gender"] = np.random.choice(
            [1, 2],
            missing_genders.sum(),
            p=[
                p_male,
                p_female,
            ],
            replace=True,
        )

    def set_individual_age(self) -> None:
        is_student = self.individual_data["Labour Status"] == 4
        self.individual_data.loc[
            is_student,
            ["Gender", "Age"],
        ] = IterativeImputer(min_value=6, max_value=18).fit_transform(
            self.individual_data.loc[
                is_student,
                ["Gender", "Age"],
            ].values
        )
        self.individual_data[["Gender", "Age"]] = IterativeImputer(min_value=0).fit_transform(
            self.individual_data[["Gender", "Age"]].values
        )

    def set_individual_education(self) -> None:
        # Impute education
        self.individual_data[["Gender", "Age", "Education"]] = IterativeImputer().fit_transform(
            self.individual_data[["Gender", "Age", "Education"]].values
        )
        self.individual_data["Education"] = self.individual_data["Education"].astype(int)

    def set_individual_labour_status(self) -> None:
        self.individual_data.loc[self.individual_data["Age"] < 16, "Labour Status"] = 4  # student
        self.individual_data.loc[self.individual_data["Labour Status"].isna()] = 2  # unemployed

    def set_individual_activity_status(
        self,
        unemployment_rate: float,
        participation_rate: float,
    ) -> None:
        # Turn the labour status into an activity status
        self.individual_data["Activity Status"] = self.individual_data["Labour Status"].map(
            self.convert_labour_status_to_activity_status,
        )

        # Adjust according to the participation rate
        current_participation_rate = np.sum(
            np.logical_and(
                self.individual_data["Activity Status"] != 3,
                self.individual_data["Age"] >= 16,
            )
        ) / np.sum(self.individual_data["Age"] >= 16)
        if participation_rate >= current_participation_rate:
            n_additional_unemployed = int(
                participation_rate * np.sum(self.individual_data["Age"] >= 16)
                - np.sum(
                    np.logical_and(
                        self.individual_data["Activity Status"] != 3,
                        self.individual_data["Age"] >= 16,
                    )
                )
            )
            ind_nea = np.where(
                np.logical_and(
                    self.individual_data["Activity Status"] == 3,
                    self.individual_data["Age"] >= 16,
                )
            )[0]
            rnd_ind = np.random.choice(
                ind_nea,
                np.min(n_additional_unemployed, len(ind_nea)),
                replace=False,
            )
            self.individual_data.loc[rnd_ind, "Activity Status"] = 2
        else:
            n_additional_nea = int(
                np.sum(
                    np.logical_and(
                        self.individual_data["Activity Status"] != 3,
                        self.individual_data["Age"] >= 16,
                    )
                )
                - participation_rate * np.sum(self.individual_data["Age"] >= 16)
            )
            ind_unemp = np.where(
                np.logical_and(
                    self.individual_data["Activity Status"] == 2,
                    self.individual_data["Age"] >= 16,
                )
            )[0]
            rnd_ind = np.random.choice(
                ind_unemp,
                min(n_additional_nea, len(ind_unemp)),
                replace=False,
            )
            self.individual_data.loc[rnd_ind, "Activity Status"] = 3
        self.individual_data.loc[
            rnd_ind,
            [
                "Employment Industry",
                "Employee Income",
                "Income from Unemployment Benefits",
                "Income",
            ],
        ] = np.nan

        # Adjust according to the unemployment rate
        current_unemployment_rate = np.sum(self.individual_data["Activity Status"] == 2) / (
            np.sum(self.individual_data["Activity Status"] == 1) + np.sum(self.individual_data["Activity Status"] == 2)
        )
        if unemployment_rate >= current_unemployment_rate:
            n_additional_unemployed = int(
                unemployment_rate
                * (
                    np.sum(self.individual_data["Activity Status"] == 1)
                    + np.sum(self.individual_data["Activity Status"] == 2)
                )
                - np.sum(self.individual_data["Activity Status"] == 2)
            )
            rnd_ind = np.random.choice(
                np.where(self.individual_data["Activity Status"] == 1)[0],
                n_additional_unemployed,
                replace=False,
            )
            self.individual_data.loc[rnd_ind, "Activity Status"] = 2
        else:
            n_additional_employed = int(
                np.sum(self.individual_data["Activity Status"] == 2)
                - unemployment_rate
                * (
                    np.sum(self.individual_data["Activity Status"] == 1)
                    + np.sum(self.individual_data["Activity Status"] == 2)
                )
            )
            rnd_ind = np.random.choice(
                np.where(
                    self.individual_data["Activity Status"] == 2,
                )[0],
                n_additional_employed,
                replace=False,
            )
            self.individual_data.loc[rnd_ind, "Activity Status"] = 1
        self.individual_data.loc[
            rnd_ind,
            [
                "Employment Industry",
                "Employee Income",
                "Income from Unemployment Benefits",
                "Income",
            ],
        ] = np.nan

    def set_individual_employment_nace(self) -> None:
        self.individual_data.loc[
            self.individual_data["Employment Industry"].isin(["R", "S"]),
            "Employment Industry",
        ] = "R_S"

        # Convert to numbers
        industry_map = dict(zip(self.industries, range(len(self.industries))))
        self.individual_data["Employment Industry"] = self.individual_data["Employment Industry"].map(industry_map)

        # Clean
        self.individual_data.loc[
            self.individual_data["Employment Industry"] == "-1",
            "Employment Industry",
        ] = np.nan
        self.individual_data.loc[
            self.individual_data["Employment Industry"] == "-2",
            "Employment Industry",
        ] = np.nan

        # Get current frequency of NACE employments
        frequencies = np.array(
            [np.sum(self.individual_data["Employment Industry"] == ind) for ind in range(len(self.industries))]
        ).astype(float)
        frequencies /= np.sum(frequencies)

        # Fill-in missing sectors
        sectors_missing = np.where(frequencies == 0)[0]
        individuals_missing_industry = np.where(
            np.logical_and(
                self.individual_data["Activity Status"] < 3,
                pd.isnull(self.individual_data["Employment Industry"]).values,
            )
        )[0]
        self.individual_data.loc[individuals_missing_industry, "Employment Industry"] = np.random.choice(
            sectors_missing,
            len(individuals_missing_industry),
            replace=True,
        )

        """
        # Distribute proportionally
        individuals_missing_industry = np.where(
            np.logical_and(
                self.individual_data["Activity Status"] < 3,
                pd.isnull(self.individual_data["Employment Industry"]).values,
            )
        )[0]
        self.individual_data.loc[individuals_missing_industry, "Employment Industry"] = np.random.choice(
            np.arange(len(self.industries)),
            len(individuals_missing_industry),
            p=frequencies,
            replace=True,
        )
        """

        # Assumption
        self.individual_data.loc[self.individual_data["Activity Status"] == 3, "Employment Industry"] = np.nan

    def set_individual_employee_income(
        self,
        unemployment_benefits_by_individual: float,
    ) -> None:
        is_employed = self.individual_data["Activity Status"] == 1

        # We're not explicitly modelling this
        self.individual_data["Employee Income"] += self.individual_data["Self-Employment Income"].fillna(0.0).values
        self.individual_data.loc[
            self.individual_data["Employee Income"] < 0,
            "Employee Income",
        ] = 0.0
        no_income = self.individual_data["Employee Income"] == 0.0
        self.individual_data.loc[is_employed & no_income, "Employee Income"] = np.nan
        self.individual_data.loc[
            is_employed,
            ["Employee Income", "Age", "Education"],
        ] = IterativeImputer(min_value=0).fit_transform(
            self.individual_data.loc[
                is_employed,
                [
                    "Employee Income",
                    "Age",
                    "Education",
                ],
            ].values
        )

        # Only employed individuals receive employee income
        self.individual_data.loc[self.individual_data["Activity Status"] != 1, "Employee Income"] = 0.0

        # Rescale that
        self.individual_data.loc[:, "Employee Income"] *= self.scale

        # Monthly!
        self.individual_data.loc[:, "Employee Income"] /= 12.0

        # Employee income is at least the unemployment rate
        is_employed = self.individual_data["Activity Status"] == 1
        self.individual_data.loc[
            np.logical_and(
                is_employed,
                self.individual_data["Employee Income"] < unemployment_benefits_by_individual,
            ),
            "Employee Income",
        ] = unemployment_benefits_by_individual

    def set_individual_unemployed_income(
        self,
        unemployment_benefits_by_individual: float,
    ) -> None:
        is_unemployed = self.individual_data["Activity Status"] == 2
        not_unemployed = self.individual_data["Activity Status"] != 2

        self.individual_data["Income from Unemployment Benefits"] = 0.0
        self.individual_data.loc[
            is_unemployed, "Income from Unemployment Benefits"
        ] = unemployment_benefits_by_individual

        # Only unemployed individuals receive unemployment income
        self.individual_data.loc[not_unemployed, "Income from Unemployment Benefits"] = 0.0

    def set_income(self) -> None:
        self.individual_data["Income"] = (
            self.individual_data["Employee Income"].fillna(0.0).values
            + self.individual_data["Income from Unemployment Benefits"].fillna(0.0).values
        )

    def create_households(
        self,
        rent_as_fraction_of_unemployment_rate: float,
        unemployment_benefits_by_capita: float,
    ) -> None:
        # Remove households without associated individuals
        self.household_data = self.household_data.loc[self.household_data["Corresponding Individuals ID"].notna()]

        # Determine the household type
        self.set_household_types()

        # Set housing inet_data
        self.set_household_housing_data(
            rent_as_fraction_of_unemployment_rate=rent_as_fraction_of_unemployment_rate,
            unemployment_benefits_by_capita=unemployment_benefits_by_capita,
        )

    def restrict(self) -> None:
        self.household_data = self.household_data[
            [
                "Type",
                "Corresponding Individuals ID",
                "Corresponding Bank ID",
                "Corresponding Inhabited House ID",
                "Corresponding Renters",
                "Corresponding Property Owner",
                "Corresponding Additionally Owned Houses ID",
                #
                "Income",
                "Employee Income",
                "Regular Social Transfers",
                "Rental Income from Real Estate",
                "Income from Financial Assets",
                #
                "Saving Rate",
                #
                "Rent Paid",
                "Rent Imputed",
                #
                "Wealth",
                "Net Wealth",
                "Wealth in Real Assets",
                "Value of the Main Residence",
                "Value of other Properties",
                "Wealth Other Real Assets",
                "Wealth in Deposits",
                "Wealth in Other Financial Assets",
                "Wealth in Financial Assets",
                #
                "Outstanding Balance of HMR Mortgages",
                "Outstanding Balance of Mortgages on other Properties",
                "Outstanding Balance of other Non-Mortgage Loans",
                "Debt",
                "Debt Installments",
                #
                "Tenure Status of the Main Residence",
                "Number of Properties other than Household Main Residence",
            ]
        ]

    def compute_household_wealth(self, wealth_distribution_independents: list[str]) -> None:
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
        self.set_wealth_distribution_function(independents=wealth_distribution_independents)

    def compute_household_income(
        self,
        central_gov_config: dict[str, Any],
        total_social_transfers: float,
    ) -> None:
        self.set_household_social_transfers(
            independents=central_gov_config["functions"]["household_social_transfers"]["parameters"]["independents"][
                "value"
            ],
            total_social_transfers=total_social_transfers,
        )
        self.set_household_employee_income()
        self.set_household_income_from_financial_assets()
        self.set_household_income()

    # Converting HFCS labour status to model activity status
    @staticmethod
    def convert_labour_status_to_activity_status(ls: int) -> int:
        individual_activity_status_map = {
            1: 1,  # REGULAR_WORK -> EMPLOYED
            2: 2,  # ON_LEAVE -> UNEMPLOYED
            3: 2,  # UNEMPLOYED -> UNEMPLOYED
            4: 3,  # STUDENT -> NOT_ECONOMICALLY_ACTIVE
            5: 3,  # RETIREE -> NOT_ECONOMICALLY_ACTIVE
            6: 3,  # DISABLED -> NOT_ECONOMICALLY_ACTIVE
            7: 1,  # MILITARY_SOCIAL_SERVICE -> EMPLOYED
            8: 2,  # DOMESTIC_TASKS -> UNEMPLOYED
            9: 2,  # OTHER_NOT_FOR_PAY -> UNEMPLOYED
        }
        return individual_activity_status_map[ls]

    # Naive determination of the household type based on the set of ages in the household
    @staticmethod
    def get_household_type(ages: np.ndarray) -> int:
        ages = np.sort(ages)
        match len(ages):
            case 1:
                if ages[0] < 64:
                    return 51  # ONE_ADULT_YOUNGER_THAN_64
                else:
                    return 52  # ONE_ADULT_OLDER_THAN_65
            case 2:
                if 18 <= ages[0] < 64 and 18 <= ages[1] < 64:
                    return 6  # TWO_ADULTS_YOUNGER_THAN_65
                elif (18 <= ages[0] < 64 < ages[1]) or (ages[0] > 64 > ages[1] >= 18):
                    return 7  # TWO_ADULTS_ONE_AT_LEAST_65
                elif ages[0] < 18 and ages[1] < 18:
                    raise ValueError("Children living together?")
                else:
                    return 9  # SINGLE_PARENT_WITH_CHILDREN
            case 3:
                if ages[0] < 18 <= ages[1] and ages[2] >= 18:
                    return 10  # TWO_ADULTS_WITH_ONE_CHILD
                elif np.all(ages) >= 18:
                    return 8  # THREE_OR_MORE_ADULTS
                else:
                    return 9  # SINGLE_PARENT_WITH_CHILDREN
            case _:
                if len(ages) == 4:
                    if ages[0] < 18 <= ages[2] and ages[1] < 18 <= ages[3]:
                        return 11  # TWO_ADULTS_WITH_TWO_CHILDREN
                else:
                    if ages[-1] >= 18 and ages[-2] >= 18 and np.all(ages[0:-2] < 18):
                        return 12  # TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN

                return 13  # THREE_OR_MORE_ADULTS_WITH_CHILDREN

    def set_household_types(self) -> None:
        i_natypes = np.where(self.household_data["Type"].isna())[0]
        for i_natype in i_natypes:
            corr_inds = np.array(self.household_data["Corresponding Individuals ID"][i_natype])
            self.household_data.loc[i_natype, "Type"] = self.get_household_type(
                self.individual_data.loc[corr_inds, "Age"].values
            )

    def set_household_housing_data(
        self,
        rent_as_fraction_of_unemployment_rate: float,
        unemployment_benefits_by_capita: float,
    ) -> None:
        # Whether the household owns or rents
        self.household_data.loc[
            self.household_data["Tenure Status of the Main Residence"] == 2,
            "Tenure Status of the Main Residence",
        ] = 1
        self.household_data.loc[
            self.household_data["Tenure Status of the Main Residence"] == 4,
            "Tenure Status of the Main Residence",
        ] = 1
        self.household_data.loc[
            self.household_data["Tenure Status of the Main Residence"] == 3,
            "Tenure Status of the Main Residence",
        ] = 0
        households_renting = self.household_data["Tenure Status of the Main Residence"] == 0
        households_owning = self.household_data["Tenure Status of the Main Residence"] == 1

        # Rent paid and value of the household main residence
        self.household_data.loc[:, "Rent Paid"] *= self.scale
        self.household_data.loc[:, "Value of the Main Residence"] *= self.scale
        self.household_data.loc[
            np.logical_and(households_renting, self.household_data["Rent Paid"] == 0.0),
            "Rent Paid",
        ] = np.nan
        self.household_data.loc[
            np.logical_and(
                households_owning,
                self.household_data["Value of the Main Residence"] == 0.0,
            ),
            "Value of the Main Residence",
        ] = np.nan
        self.household_data.loc[
            :,
            [
                "Type",
                "Rent Paid",
                "Value of the Main Residence",
            ],
        ] = IterativeImputer().fit_transform(
            self.household_data[
                [
                    "Type",
                    "Rent Paid",
                    "Value of the Main Residence",
                ]
            ].values
        )
        social_housing_rent = rent_as_fraction_of_unemployment_rate * unemployment_benefits_by_capita
        self.household_data.loc[
            self.household_data["Rent Paid"] < social_housing_rent, "Rent Paid"
        ] = social_housing_rent
        self.household_data.loc[
            households_owning,
            "Rent Paid",
        ] = 0.0

        # Number of additional properties
        self.household_data.loc[
            self.household_data["Number of Properties other than Household Main Residence"].isna(),
            "Number of Properties other than Household Main Residence",
        ] = 0
        self.household_data.loc[:, "Number of Properties other than Household Main Residence"] = self.household_data[
            "Number of Properties other than Household Main Residence"
        ].astype(int)

        # Value of other properties
        self.household_data.loc[:, "Value of other Properties"] *= self.scale
        household_without_additional_properties = (
            self.household_data["Number of Properties other than Household Main Residence"] == 0
        )
        self.household_data.loc[household_without_additional_properties, "Value of other Properties"] = 0.0
        self.household_data.loc[
            np.logical_and(
                np.logical_not(household_without_additional_properties),
                self.household_data["Value of other Properties"] == 0.0,
            ),
            "Value of other Properties",
        ] = np.nan
        self.household_data.loc[
            :,
            [
                "Type",
                "Number of Properties other than Household Main Residence",
                "Value of the Main Residence",
                "Value of other Properties",
            ],
        ] = IterativeImputer().fit_transform(
            self.household_data[
                [
                    "Type",
                    "Number of Properties other than Household Main Residence",
                    "Value of the Main Residence",
                    "Value of other Properties",
                ]
            ].values
        )

        # Rent received
        self.household_data.loc[:, "Rental Income from Real Estate"] *= self.scale
        self.household_data.loc[:, "Rental Income from Real Estate"] /= 12.0
        self.household_data.loc[
            self.household_data["Rental Income from Real Estate"] < social_housing_rent,
            "Rental Income from Real Estate",
        ] = social_housing_rent
        self.household_data.loc[household_without_additional_properties, "Rental Income from Real Estate"] = 0.0
        self.household_data.loc[
            np.logical_and(
                np.logical_not(household_without_additional_properties),
                self.household_data["Rental Income from Real Estate"] == 0.0,
            ),
            "Rental Income from Real Estate",
        ] = np.nan
        self.household_data.loc[
            :,
            [
                "Type",
                "Value of other Properties",
                "Rental Income from Real Estate",
            ],
        ] = IterativeImputer(min_value=0.0).fit_transform(
            self.household_data[
                [
                    "Type",
                    "Value of other Properties",
                    "Rental Income from Real Estate",
                ]
            ].values
        )

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
        self.household_data["Net Wealth"] = self.household_data["Wealth"] - self.household_data["Debt"]

    def set_wealth_distribution_function(self, independents: list[str]) -> None:
        self.household_data["Fraction Deposits / Total Financial Wealth"] = np.divide(
            self.household_data["Wealth in Deposits"].values.astype(float),
            self.household_data["Wealth in Financial Assets"].values.astype(float),
            out=np.ones_like(self.household_data["Wealth in Deposits"].values),
            where=self.household_data["Wealth in Financial Assets"].values.astype(float) != 0.0,
        )
        _, self.wealth_distribution_model = fit_linear(
            household_data=self.household_data,
            independents=independents,
            dependent="Fraction Deposits / Total Financial Wealth",
        )

    def set_household_employee_income(self) -> None:
        self.household_data["Employee Income"] = [
            self.individual_data.loc[self.household_data["Corresponding Individuals ID"][i], "Income"].sum()
            for i in range(len(self.household_data))
        ]

    def set_household_social_transfers(
        self,
        independents: list[str],
        total_social_transfers: float,
    ) -> None:
        # Household regular social transfers and pensions, impute missing values
        self.household_data.loc[
            :,
            ["Type", "Net Wealth", "Regular Social Transfers", "Income from Pensions"],
        ] = IterativeImputer(min_value=0).fit_transform(
            self.household_data.loc[
                :,
                [
                    "Type",
                    "Net Wealth",
                    "Regular Social Transfers",
                    "Income from Pensions",
                ],
            ].values
        )

        # Aggregate
        self.household_data["Regular Social Transfers"] += self.household_data["Income from Pensions"].values

        # Social transfers for each household group
        self.household_data["Regular Social Transfers"] /= self.household_data["Regular Social Transfers"].sum()
        social_transfers, self.social_transfers_model = fit_linear(
            household_data=self.household_data,
            independents=independents,
            dependent="Regular Social Transfers",
        )
        social_transfers[social_transfers < 0] = 0.0
        self.household_data["Regular Social Transfers"] = social_transfers

        # Rescale them
        self.household_data["Regular Social Transfers"] *= (
            total_social_transfers / self.household_data["Regular Social Transfers"].sum()
        )

    def set_household_income_from_financial_assets(self) -> None:
        fa_mask = np.logical_not(np.isnan(self.household_data["Income from Financial Assets"].values.astype(float)))
        self.household_data["Income from Financial Assets"] *= self.scale
        self.coefficient_fa_income = (
            self.household_data["Income from Financial Assets"].values.astype(float)[fa_mask].sum() / 12.0
        ) / (self.household_data["Wealth in Other Financial Assets"].values.astype(float)[fa_mask]).sum()
        self.household_data["Income from Financial Assets"] = (
            self.coefficient_fa_income * self.household_data["Wealth in Other Financial Assets"].values
        )

    def set_household_income(self) -> None:
        self.household_data["Income"] = (
            self.household_data["Employee Income"]
            + self.household_data["Regular Social Transfers"]
            + self.household_data["Rental Income from Real Estate"]
            + self.household_data["Income from Financial Assets"]
        )

    def set_household_saving_rates(self, function_name: str, independents: list[str]) -> None:
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

        # Impute missing values
        temp_imp = IterativeImputer(min_value=0, max_value=1).fit_transform(
            self.household_data[
                [
                    "Type",
                    "Income",
                    "Wealth",
                    "Debt",
                    "Consumption of Consumer Goods/Services as a Share of Income",
                ]
            ].values
        )
        self.household_data.loc[:, "Saving Rate"] = 1.0 - temp_imp[:, 4]

        # Fit a model
        saving_rates, self.saving_rates_model = fit_linear(
            household_data=self.household_data,
            independents=independents,
            dependent="Saving Rate",
        )

        # Saving rates by household characteristics
        if function_name == "AverageSavingRatesSetter":
            self.household_data["Saving Rate"] = saving_rates.mean()
        else:
            self.household_data["Saving Rate"] = saving_rates
