"""Canada Output-Based Pricing System (OBPS) policy for the macroeconomic model.

Calculates the tax that firms pay on emissions that exceed a sector-specific
prescribed limit. The tax is computed as:

    obps_cost[i] = max(0, emissions[i] - limit[i]) * carbon_price[t]

Dividing by production gives a per-unit marginal cost passed to firms as
extra_marginal_taxes_firm in the country's target-setting phase.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macro_data.readers.policy_data.obps_can_reader import OBPSCANData


@dataclass
class OutputBasedPriceSystemCAN:
    """Canada Output-Based Pricing System policy class.

    Calculates the tax that regulated firms pay on emissions above a
    prescribed output-weighted limit. Reference emission intensities are
    recorded during 2017–2019 and used to set sector-specific limits from
    2019 onwards.

    Attributes:
        country_name: Jurisdiction column to use from the rates CSV.
        industries: Ordered list of all model industry names.
        regulated_industries: Industries subject to OBPS regulation.
        regulated_indices: Array indices into the industries list for each
            regulated industry.
        df_policy: Per-industry reduction factors and tightening rates.
        df_policy_elec: Electricity-specific tightening rates (optional).
        df_rates: Carbon price schedule by year and jurisdiction.
        reference_emission_intensity: Baseline emission intensity recorded
            over the 2017–2019 reference period.
        reference_emission: Cumulative emissions during the reference period.
        reference_production: Cumulative production during the reference period.
        emission_limit: Current period allowable emissions per industry.
        price: Annual carbon price trajectory (indexed from 2014).
        current_t: Current timestep index (incremented by update()).
        current_year: Current calendar year.
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
    emission_limit: np.ndarray
    price: np.ndarray
    current_t: int = 0
    current_year: int = 2014

    def __init__(self, country_name: str, industries: list[str], obps_data: OBPSCANData):
        """Initialise the OBPS from a loaded OBPSCANData container.

        Args:
            country_name: Jurisdiction code matching a column in obps_data.df_rates.
            industries: Ordered list of all model industry names.
            obps_data: Loaded OBPS CSV data.
        """
        self.country_name = country_name
        self.industries = industries

        # Industries regulated under OBPS (federal schedule)
        self.regulated_industries = [
            "B05a", "B05b", "B05c",
            "B07", "B09",
            "C10T12", "C16", "C17", "C19", "C20", "C21", "C22", "C23",
            "C24a", "C24b", "C29", "C30",
            "D01b", "D01c",
        ]
        self.regulated_indices = np.array(
            [list(industries).index(ind) for ind in self.regulated_industries if ind in industries]
        )

        n = len(industries)
        self.reference_emission_intensity = np.zeros(n)
        self.reference_emission = np.zeros(n)
        self.reference_production = np.zeros(n)
        self.emission_limit = np.zeros(n)

        self.df_policy = obps_data.df_policy
        self.df_policy_elec = obps_data.df_policy_elec if obps_data.df_policy_elec is not None else pd.DataFrame()
        self.df_rates = obps_data.df_rates

        self.price = np.zeros(len(self.df_rates))
        df_sub = self.df_rates[["Date", self.country_name]]
        for t in range(len(self.df_rates)):
            df_row = df_sub[df_sub["Date"] == t + 2014]
            self.price[t] = df_row[self.country_name].values[0]

    def compute_obps(
        self,
        use_obps_reg: bool,
        record_obps_reference: bool,
        production: np.ndarray,
        input_em: np.ndarray,
        capital_em: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sector OBPS tax cost.

        Records reference emission intensities during 2017–2019. From 2019
        onwards, computes the cost of emissions above the prescribed limit
        at the current carbon price.

        Args:
            use_obps_reg: If False, returns a zero array.
            record_obps_reference: If True, accumulate reference period data.
            production: Current-period production per industry.
            input_em: Input-related CO₂e emissions per industry.
            capital_em: Capital-related CO₂e emissions per industry.

        Returns:
            np.ndarray: OBPS tax cost (dollars) per industry; zero for
                unregulated industries or years before 2019.
        """
        if not use_obps_reg:
            return np.zeros(len(self.industries))

        if record_obps_reference and self.current_year in (2017, 2018, 2019):
            self.reference_emission += input_em + capital_em
            self.reference_production += production

        if self.current_year == 2019:
            self.reference_emission_intensity = np.divide(
                self.reference_emission,
                self.reference_production,
                out=np.zeros_like(self.reference_emission),
                where=self.reference_production != 0,
            )

        if self.current_year < 2019:
            return np.zeros(len(self.industries))

        obps_cost = np.zeros(len(self.industries))
        for i in self.regulated_indices:
            if production[i] > 0:
                limit = self.get_limit(i, production[i])
                self.emission_limit[i] = limit
                difference = (input_em[i] + capital_em[i]) - limit
                obps_cost[i] = difference * self.price[min(self.current_t, len(self.price) - 1)]

        return obps_cost

    def get_limit(self, industry_idx: int, production: float) -> float:
        """Calculate the prescribed emission allowance for an industry.

        Uses the pre-2023 formula (reduction factor only) or the post-2023
        formula (reduction factor × tightening adjustment).

        Args:
            industry_idx: Index into self.industries.
            production: Current period output.

        Returns:
            float: Allowable emissions in tCO₂e.
        """
        industry_name = self.industries[industry_idx]
        row = self.df_policy[self.df_policy["Industry"] == industry_name]
        if row.empty:
            return 0.0

        reduction_factor = row["reduction_factor"].values[0]
        B = reduction_factor * self.reference_emission_intensity[industry_idx]

        if self.current_year < 2023:
            return production * B

        tightening_rate = row["tightening_rate"].values[0]
        return production * (B - B * tightening_rate * (self.current_year - 2022))

    def get_price(self) -> float:
        """Return the current period carbon price ($/tCO₂e)."""
        return self.price[self.current_t]

    def update(self) -> None:
        """Advance the timestep by one annual period."""
        self.current_t += 1
        self.current_year += 1

    def reset(self) -> None:
        """Reset time variables to the initial year."""
        self.current_t = 0
        self.current_year = 2014
