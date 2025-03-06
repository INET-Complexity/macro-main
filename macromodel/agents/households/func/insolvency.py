"""Household insolvency management implementation.

This module implements household insolvency handling through:
- Default detection and processing
- Debt restructuring mechanisms
- Asset liquidation procedures
- Bank interaction management

The implementation handles:
- Insolvency identification
- Credit market impacts
- Non-performing loan tracking
- Default rate calculations
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from macromodel.agents.banks.banks import Banks
from macromodel.agents.households.households import Households
from macromodel.markets.credit_market.credit_market import CreditMarket


class HouseholdInsolvencyHandler(ABC):
    """Abstract base class for household insolvency management.

    Defines interface for handling household defaults through:
    - Insolvency detection
    - Default processing
    - Credit market updates
    - Bank interactions
    """

    @abstractmethod
    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
    ) -> Tuple[float, float, float]:
        """Process household insolvency cases.

        Args:
            households (Households): Household sector agent
            banks (Banks): Banking sector agent
            credit_market (CreditMarket): Credit market interface

        Returns:
            Tuple[float, float, float]: Default rate, NPL ratios for consumption
                loans and mortgages
        """
        pass


class DefaultHouseholdInsolvencyHandler(HouseholdInsolvencyHandler):
    """Default implementation of household insolvency management.

    Implements insolvency handling through:
    - Net wealth and deposit checks
    - Loan removal processing
    - NPL ratio calculations
    """

    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
    ) -> Tuple[float, float, float]:
        """Process household defaults using default behavior.

        Handles insolvency through:
        - Identifying insolvent households
        - Processing loan removals
        - Calculating default metrics

        Args:
            households (Households): Household sector agent
            banks (Banks): Banking sector agent
            credit_market (CreditMarket): Credit market interface

        Returns:
            Tuple[float, float, float]: Default rate, NPL ratios for consumption
                loans and mortgages
        """
        insolvent_households = np.where(
            np.logical_and(
                households.ts.current("net_wealth") < 0,
                households.ts.current("wealth_deposits") < 0,
            )
        )[0]
        bad_hh_cons_loans, bad_mortgages = credit_market.remove_loans_to_households(insolvent_households)

        # Calculate NPL ratios
        total_cons_loans = credit_market.ts.current("total_outstanding_loans_granted_households_consumption")[0]
        if total_cons_loans == 0.0:
            npl_hh_cons_loans = 0.0
        else:
            npl_hh_cons_loans = bad_hh_cons_loans / total_cons_loans
        if credit_market.ts.current("total_outstanding_loans_granted_mortgages")[0] == 0.0:
            npl_mortgages = 0.0
        else:
            npl_mortgages = bad_mortgages / credit_market.ts.current("total_outstanding_loans_granted_mortgages")[0]

        return (
            len(insolvent_households) / households.ts.current("n_households"),
            npl_hh_cons_loans,
            npl_mortgages,
        )
