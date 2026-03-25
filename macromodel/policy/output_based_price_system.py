"""Output based price system policy for the macroeconomic model.

This module calculates the tax that firms pay on emissions over a prescribed limit
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OutputBasedPriceSystem:
    """Output based price system policy class.

    This class calculates the tax that firms pay on emissions over a prescribed limit

    Attributes:
        country_name (str): Country with the OBPS policy
        industries (list[str]): Industries in country
        regulated_industries (list[str]): Regulated industries
        regulated_indices (np.ndarray): Indices of the regulated industries
        df_policy (pd.DataFrame): Imported OBPS policy data
        df_rates (pd.DataFrame): Imported OBPS rates data
        reference_emission_intensity (np.ndarray): Emission intensity used to determine emission limit
        reference_emission (np.ndarray): Emissions used to determine emission limit
        reference_production (np.ndarray): Production used to determine emission limit
        emission_limit: np.ndarray  # Temporary - this variable is only for plotting
        price (np.ndarray): Tax rate
        current_t (int): Current index for price
        current_year (int) = Current year
    """

    country_name: str
    industries: list[str]
    regulated_industries: list[str]
    regulated_indices: np.ndarray
    df_policy: pd.DataFrame
    df_policy_elec: pd.DataFrame
    df_rates: pd.DataFrame
    reference_emission_intensity: np.ndarray
    reference_emission: np.ndarray
    reference_production: np.ndarray
    emission_limit: np.ndarray  # Temporary - this variable is only for plotting
    price: np.ndarray
    current_t: int = 0
    current_year: int = 2014

    def __init__(self, country_name, industries, policy_data=None):
        """Initialize a new output based pricing system.

        Args:
            country_name (str): Country with the OBPS policy
            industries (list[str]): Industries in country
            regulated_industries (list[str]): Regulated industries
            regulated_indices (np.ndarray): Indices of the regulated industries
            df_policy (pd.DataFrame): Imported OBPS policy data
            df_policy_elec (pd.DataFrame): Imported OBPS electricity data
            df_rates (pd.DataFrame): Imported OBPS rates data
            reference_emission_intensity (np.ndarray): Emission intensity used to determine emission limit
            reference_emission (np.ndarray): Emissions used to determine emission limit
            reference_production (np.ndarray): Production used to determine emission limit
            emission_limit: np.ndarray  # Temporary - this variable is only for plotting
            price (np.ndarray): Tax rate
            current_t (int): Current index for price
            current_year (int) = Current year
        """
        self.country_name = country_name

        # industries covered by OBPS      https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/page-11.html#h-1185036
        self.industries = industries
        self.regulated_industries = [
            "B05a",
            "B05b",
            "B05c",
            "B07",
            "B09",
            "C10T12",
            "C16",
            "C17",
            "C19",
            "C20",
            "C21",
            "C22",
            "C23",
            "C24a",
            "C24b",
            "C29",
            "C30",
            # "D",
            "D01b",
            "D01c",
        ]
        self.regulated_indices = np.array([list(industries).index(industry) for industry in self.regulated_industries])
        self.reference_emission_intensity = np.zeros(len(self.industries))
        self.reference_emission = np.zeros(len(self.industries))
        self.reference_production = np.zeros(len(self.industries))
        self.emission_limit = np.zeros(len(self.industries))

        ### fetch OBPS values
        # reduction rates https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/section-37.html?txthl=reduction+factor
        # tightening values https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/section-36.2.html?txthl=tightening
        # standards https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/page-11.html#h-1185036
        # OBPS_POLICY_VALUES_PATH = "./macromodel/policy/output_based_price_system_policy_values.csv"
        # OBPS_POLICY_VALUES_PATH = "./macromodel/policy/output_based_price_system_policy_values_disagg.csv"
        self.df_policy = policy_data.obps_policy_data["df_rates"]  # pd.read_csv(OBPS_POLICY_VALUES_PATH)

        ### fetch electricity tightening values  36.1(2) https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/page-5.html#docCont
        # OBPS_POLICY_VALUES_ELEC_PATH = "./macromodel/policy/output_based_price_system_policy_values_elec.csv"
        self.df_policy_elec = policy_data.obps_policy_data[
            "df_policy_elec"
        ]  # pd.read_csv(OBPS_POLICY_VALUES_ELEC_PATH)

        ### Fetch tax rates
        # CAN   https://laws-lois.justice.gc.ca/eng/acts/g-11.55/page-29.html#h-247156
        #       https://440megatonnes.ca/policy-tracker/
        #
        # AB    for 2018-2020: https://www.alberta.ca/carbon-competitiveness-incentive-regulation
        #           https://www.alberta.ca/system/files/custom_downloaded_images/CCEMA-fund-credit-ministerial-order.PDF
        #       for 2021 onward: https://www.alberta.ca/technology-innovation-and-emissions-reduction-regulation
        #           https://kings-printer.alberta.ca/Documents/MinOrders/2022/Environment_and_Protected_Areas/2022_062_Environment_and_Protected_Areas.pdf
        #       alternate references:
        #           https://www.alberta.ca/system/files/custom_downloaded_images/CCI-OBA-presentation-Dec-2017.pdf
        #           https://ieta.b-cdn.net/wp-content/uploads/2024/02/Alberta-business-brief_Dec-23.pdf
        #           https://www.mccarthy.ca/en/insights/blogs/canadian-energy-perspectives/alberta-government-indefinitely-freezes-cost-tier-fund-credits-95-tonne
        # BC    for 2019-2024: https://www2.gov.bc.ca/gov/content/environment/climate-change/industry/cleanbc-industrial-incentive-program/about-ciip
        #           https://www2.gov.bc.ca/assets/gov/environment/climate-change/ind/cleanbc-program-for-industry/guidance/2024__ciip_general_guidance.pdf
        #       for 2025 onward: aligns with federal
        # ON    for 2023 onward: https://ero.ontario.ca/notice/019-5769

        # OBPS_CARBON_PRICE_PATH = "./macromodel/policy/output_based_price_system_rates.csv"
        self.df_rates = policy_data.obps_policy_data["df_policy"]  # pd.read_csv(OBPS_CARBON_PRICE_PATH)

        self.price = np.zeros(len(self.df_rates))
        df_sub = self.df_rates[["Date", self.country_name]]
        for t in range(len(self.df_rates)):
            time = t + 2014
            df_row = df_sub[df_sub["Date"] == time]
            df_item = df_row[self.country_name].values
            self.price[t] = df_item[0]

    def compute_obps(self, use_obps_reg, record_obps_reference, production, input_em, capital_em):
        """Computes the tax owed by firms on input and capital emissions based on a carbon price

        Compare emission limits to the firm emissions and then calculate the cost of the difference

        Returns:
            np.ndarray: Output based pricing system taxation values for each industry
        """

        if use_obps_reg:

            self.emission_limit = np.zeros(len(self.industries))

            # calculate output standards
            # https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/section-37.html?txthl=80
            if record_obps_reference and (
                self.current_year == 2017 or self.current_year == 2018 or self.current_year == 2019
            ):
                # record_obps_reference boolean ensures that the reference isn't double counted between the planning and realized phases
                self.reference_emission += input_em + capital_em
                self.reference_production += production
            if self.current_year == 2019:
                # output standard = (sum_i (A - (B + C + F - G))/ sum_i D) * E = (sum_i A/ sum_i D) * E     when B = C = F = G = 0
                # A = self.reference_emission
                # D = self.reference_production
                output_mask = self.reference_production == 0
                # if np.any(self.reference_production == 0):
                #     print("\tdiv by zero: " + str(np.where(self.reference_production == 0)))
                self.reference_emission_intensity = (
                    self.reference_emission / self.reference_production
                )  # = (sum_i A/ sum_i D)
                self.reference_emission_intensity[output_mask] = 0

            if self.current_year >= 2019:
                obps_cost = np.zeros(len(self.industries))
                for i in range(len(self.industries)):
                    if i in self.regulated_indices and production[i] > 0:
                        limit = self.get_limit(i, production[i])  # [tCO2]
                        self.emission_limit[i] = limit
                        difference_i = (input_em[i] + capital_em[i]) - limit  # [tCO2]
                        obps_cost[i] = difference_i * self.price[self.current_t]  # [tCO2]*[$/tCO2] = [$]
                    else:
                        obps_cost[i] = 0.0

                # if self.current_year >= 2027:
                #     print(self.emission_limit[self.regulated_indices])
            else:
                obps_cost = np.zeros(len(production))
        else:
            obps_cost = None

        return obps_cost

    def get_limit(self, i, production):
        """Calculates the prescribed emission limit for a industry.

        Returns:
            float: Emission limit for a particular industry
        """
        if self.current_year < 2023:
            ### https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/section-36-20190628.html#wb-cont)
            # limit = A * B
            # A = production
            # B = standard
            temp = self.df_policy[self.df_policy["Industry"] == self.industries[i]]
            A = production  # [#]
            B = temp["reduction_factor"].to_list()[0] * self.reference_emission_intensity[i]  # [tCO2/#]
            limit = A * B  # [tCO2]
        else:
            ### https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/page-5.html#h-1184433
            # limit = A * (B - (B * C * (D -2022)))
            # A = production
            # B = standard          https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/page-5.html#docCont
            # C = tighting rate     https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/section-36.2.html?txthl=tightening
            # D = current year

            temp = self.df_policy[self.df_policy["Industry"] == self.industries[i]]
            A = production  # [#]
            B = temp["reduction_factor"].to_list()[0] * self.reference_emission_intensity[i]  # [tCO2/#]

            ### review to determine whether this elec specific rate could work!!!
            # if self.industries[i] == "D01b" or self.industries[i] == "D01c":
            #     # update "tightening_rate" for electricity from data
            #     df_row = self.df_policy_elec[self.df_policy_elec['Date'] == self.current_year]
            #     df_item = df_row["tightening_rate"].values
            #     tightening_rate = df_item[0]
            #     C = tightening_rate
            # else:
            C = temp["tightening_rate"].to_list()[0]
            D = self.current_year

            limit = A * (B - (B * C * (D - 2022)))  # [tCO2]

        return limit

    def get_price(self):
        """Retrieve the current price

        Returns:
            float: Current tax rate
        """
        return self.price[self.current_t]

    def update(self):
        """Advance the timestep by the increment."""
        self.current_t += 1
        self.current_year += 1

    def reset(self):
        """Resets the time variables"""
        self.current_t = 0
        self.current_year = 2014
