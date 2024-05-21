import numpy as np

from abc import abstractmethod, ABC
from macromodel.banks.banks import Banks
from macromodel.households.households import Households
from macromodel.credit_market.credit_market import CreditMarket

from typing import Tuple


class HouseholdInsolvencyHandler(ABC):
    @abstractmethod
    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
    ) -> Tuple[float, float, float]:
        pass


class DefaultHouseholdInsolvencyHandler(HouseholdInsolvencyHandler):
    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
    ) -> Tuple[float, float, float]:
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
