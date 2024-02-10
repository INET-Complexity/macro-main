import numpy as np
from abc import abstractmethod, ABC

from macromodel.banks.banks import Banks
from macromodel.credit_market.credit_market import CreditMarket
from macromodel.households.households import Households


class HouseholdInsolvencyHandler(ABC):
    @abstractmethod
    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
    ) -> float:
        pass


class DefaultHouseholdInsolvencyHandler(HouseholdInsolvencyHandler):
    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
    ) -> float:
        insolvent_households = np.where(
            np.logical_and(
                households.ts.current("net_wealth") < 0,
                households.ts.current("wealth_deposits") < 0,
            )
        )[0]
        credit_market.remove_loans_to_households(insolvent_households)
        return len(insolvent_households) / households.ts.current("n_households")
