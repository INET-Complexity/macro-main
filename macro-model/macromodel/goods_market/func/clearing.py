from abc import abstractmethod, ABC
from tqdm import trange
from typing import Tuple, Optional

from macromodel.goods_market.func.lib_default import *
from macromodel.goods_market.func.lib_pro_rata import *
from macromodel.goods_market.value_type import ValueType


class GoodsMarketClearer(ABC):
    def __init__(
        self,
        prio_real_countries: bool,
        prio_high_prio_buyers: bool,
        prio_high_prio_sellers: bool,
        prio_domestic_sellers: bool,
        probability_keeping_previous_seller: float,
        price_temperature: float,
        trade_temperature: float,
        remedy_rounding_errors: bool = True,
    ):
        # Parameters
        self.prio_real_countries = prio_real_countries
        self.prio_high_prio_buyers = prio_high_prio_buyers
        self.prio_high_prio_sellers = prio_high_prio_sellers
        self.prio_domestic_sellers = prio_domestic_sellers
        self.probability_keeping_previous_seller = probability_keeping_previous_seller
        self.price_temperature = price_temperature
        self.trade_temperature = trade_temperature
        self.remedy_rounding_errors = remedy_rounding_errors

        # Agents
        self.n_industries = None
        self.goods_market_participants = None

        # Supply chain
        self.current_supply_chain = None
        self.previous_supply_chain = None

    def initiate_agents(
        self,
        n_industries: int,
        goods_market_participants: dict[str, list[Agent]],
    ):
        self.n_industries = n_industries
        self.goods_market_participants = goods_market_participants

    def initiate_the_supply_chain(
        self,
        initial_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]] = None,
    ):
        if initial_supply_chain is None:
            self.current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]] = {
                g: {} for g in range(self.n_industries)
            }
        else:
            self.current_supply_chain = initial_supply_chain
        self.previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]] = {
            g: {} for g in range(self.n_industries)
        }

    def prepare(self) -> None:
        for country_name in self.goods_market_participants.keys():
            for transactor in self.goods_market_participants[country_name]:
                transactor.prepare()
        self.previous_supply_chain = self.current_supply_chain
        self.current_supply_chain = {g: {} for g in range(self.n_industries)}

    def collect_all_supply_and_demand(self) -> tuple[np.ndarray, np.ndarray]:
        (
            total_real_supply,
            aggr_real_supply,
            total_nominal_supply,
            aggr_nominal_supply,
            average_goods_price,
        ) = collect_seller_info(
            goods_market_participants=self.goods_market_participants,
            n_industries=self.n_industries,
        )
        (
            total_real_demand,
            aggr_real_demand,
            total_nominal_demand,
            aggr_nominal_demand,
        ) = collect_buyer_info(
            goods_market_participants=self.goods_market_participants,
            average_price=average_goods_price,
            n_industries=self.n_industries,
        )
        return aggr_nominal_supply, aggr_nominal_demand

    @abstractmethod
    def clear(self, default_trade_proportions: pd.DataFrame) -> None:
        pass

    def record(self) -> None:
        for country_name in self.goods_market_participants.keys():
            for transactor in self.goods_market_participants[country_name]:
                transactor.record()


class NoGoodsMarketClearer(GoodsMarketClearer):
    def clear(self, default_trade_proportions: pd.DataFrame) -> None:
        pass


class DefaultGoodsMarketClearer(GoodsMarketClearer):
    def clear(self, default_trade_proportions: pd.DataFrame) -> None:
        for g in trange(self.n_industries, desc="Clearing the goods market"):
            # Buy and sell
            while True:
                # Get a random buyer
                buyer, buyer_ind = get_random_buyer(
                    industry=g,
                    goods_market_participants=self.goods_market_participants,
                    prio_real_countries=self.prio_real_countries,
                    prio_high_prio=self.prio_high_prio_buyers,
                    field="Remaining Goods",
                )

                # Get a random seller
                seller, seller_ind = get_random_seller(
                    industry=g,
                    goods_market_participants=self.goods_market_participants,
                    chosen_buyer=buyer,
                    chosen_buyer_ind=buyer_ind,
                    previous_supply_chain=self.previous_supply_chain,
                    prio_real_countries=self.prio_real_countries,
                    prio_high_prio_sellers=self.prio_high_prio_sellers,
                    prio_domestic_sellers=self.prio_domestic_sellers,
                    probability_keeping_previous_seller=self.probability_keeping_previous_seller,
                    price_temperature=self.price_temperature,
                    field="Remaining Goods",
                )

                # Handle the transaction
                handle_transaction(
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

                # Update the supply chain
                update_supply_chain(
                    current_supply_chain=self.current_supply_chain,
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

                # Check if we are done
                if (
                    not check_buyers_left(
                        goods_market_participants=self.goods_market_participants,
                        n_industries=self.n_industries,
                        field="Remaining Goods",
                    )[g]
                ) or (
                    not check_sellers_left(
                        goods_market_participants=self.goods_market_participants,
                        n_industries=self.n_industries,
                        field="Remaining Goods",
                    )[g]
                ):
                    break

            # Distribute excess demand
            for country_name in self.goods_market_participants.keys():
                for transactor in self.goods_market_participants[country_name]:
                    if transactor.transactor_buyer_states["Value Type"] != ValueType.NONE:
                        transactor.transactor_buyer_states["Remaining Excess Goods"] = (
                            transactor.transactor_buyer_states["Remaining Goods"].copy()
                        )
            while check_buyers_left(
                goods_market_participants=self.goods_market_participants,
                n_industries=self.n_industries,
                field="Remaining Excess Goods",
            )[g]:
                # Get a random buyer
                buyer, buyer_ind = get_random_buyer(
                    industry=g,
                    goods_market_participants=self.goods_market_participants,
                    prio_real_countries=self.prio_real_countries,
                    prio_high_prio=self.prio_high_prio_buyers,
                    field="Remaining Excess Goods",
                )

                # Get a random seller
                seller, seller_ind = get_random_seller(
                    industry=g,
                    goods_market_participants=self.goods_market_participants,
                    chosen_buyer=buyer,
                    chosen_buyer_ind=buyer_ind,
                    previous_supply_chain=self.previous_supply_chain,
                    prio_real_countries=self.prio_real_countries,
                    prio_high_prio_sellers=self.prio_high_prio_sellers,
                    prio_domestic_sellers=self.prio_domestic_sellers,
                    probability_keeping_previous_seller=self.probability_keeping_previous_seller,
                    price_temperature=self.price_temperature,
                    field="Initial Goods",
                )

                # Handle hypothetical transaction
                handle_hypothetical_transaction(
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

        # Clean up
        if self.remedy_rounding_errors:
            clean_rounding_errors(goods_market_participants=self.goods_market_participants)


class ProRataGoodsMarketClearer(GoodsMarketClearer):
    def clear(self, default_trade_proportions: pd.DataFrame) -> None:
        # Account for trade proportions
        if len(default_trade_proportions) > 0:
            average_prices = {}
            for country_name in self.goods_market_participants.keys():
                for gmp in self.goods_market_participants[country_name]:
                    if gmp.transactor_settings["Seller Value Type"] != ValueType.NONE:
                        average_prices[country_name] = gmp.ts.current("price_offered")
                        break
            trade_proportions = get_trade_proportions(
                country_names=self.goods_market_participants.keys(),
                default_trade_proportions=default_trade_proportions,
                average_prices=average_prices,
                temperature=self.trade_temperature,
                n_industries=self.n_industries,
            ).fillna(0.0)
            self.perform_clearing_in_prio_order(self.goods_market_participants.keys(), trade_proportions)

        # Clear
        self.perform_clearing_in_prio_order()

        # Handle excess demand
        distribute_excess_demand(
            goods_market_participants=self.goods_market_participants,
            n_industries=self.n_industries,
        )

    def perform_clearing_in_prio_order(
        self,
        country_names: Optional[list[str]] = None,
        trade_proportions: Optional[pd.DataFrame] = None,
    ) -> None:
        for prio_pair in [[True, True], [False, True], [True, False], [False, False]]:
            seller_high_prio_only, buyer_high_prio_only = prio_pair[0], prio_pair[1]
            if (not buyer_high_prio_only or self.prio_high_prio_buyers) and (
                not seller_high_prio_only or self.prio_high_prio_sellers
            ):
                if country_names is not None and trade_proportions is not None:
                    for start_country in country_names:
                        if self.prio_real_countries and start_country == "ROW":
                            continue
                        for end_country in country_names:
                            if self.prio_real_countries and end_country == "ROW":
                                continue
                            if start_country == end_country == "ROW":
                                continue
                            current_trade_proportions = (
                                trade_proportions.xs(start_country).xs(end_country).values.flatten()
                            )
                            self.perform_clearing(
                                start_country=start_country,
                                end_country=end_country,
                                seller_high_prio_only=seller_high_prio_only,
                                buyer_high_prio_only=buyer_high_prio_only,
                                current_trade_proportions=current_trade_proportions,
                            )
                else:
                    self.perform_clearing(
                        start_country=None,
                        end_country=None,
                        seller_high_prio_only=seller_high_prio_only,
                        buyer_high_prio_only=buyer_high_prio_only,
                        current_trade_proportions=None,
                    )

    def perform_clearing(
        self,
        start_country: Optional[str],
        end_country: Optional[str],
        seller_high_prio_only: bool,
        buyer_high_prio_only: bool,
        current_trade_proportions: Optional[np.ndarray],
    ) -> None:
        (
            total_real_supply,
            aggr_real_supply,
            _,
            _,
            average_goods_price,
        ) = collect_seller_info(
            goods_market_participants=self.goods_market_participants,
            n_industries=self.n_industries,
            high_prio_only=seller_high_prio_only,
            from_country=start_country,
        )
        (
            total_real_demand,
            aggr_real_demand,
            _,
            _,
        ) = collect_buyer_info(
            goods_market_participants=self.goods_market_participants,
            average_price=average_goods_price,
            n_industries=self.n_industries,
            high_prio_only=buyer_high_prio_only,
            to_country=end_country,
            trade_proportions=current_trade_proportions,
        )
        clear(
            goods_market_participants=self.goods_market_participants,
            n_industries=self.n_industries,
            total_real_supply=total_real_supply,
            aggr_real_supply=aggr_real_supply,
            average_goods_price=average_goods_price,
            total_real_demand=total_real_demand,
            aggr_real_demand=aggr_real_demand,
            sell_high_prio_only=seller_high_prio_only,
            buy_high_prio_only=buyer_high_prio_only,
            from_country=start_country,
            to_country=end_country,
            trade_proportions=current_trade_proportions,
        )
