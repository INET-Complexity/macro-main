from abc import abstractmethod, ABC

from macromodel.goods_market.func.lib_default import *
from macromodel.goods_market.func.lib_water_bucket import *
from macromodel.goods_market.value_type import ValueType
from macromodel.goods_market.func.lib_pro_rata import (
    collect_buyer_info,
    collect_seller_info,
)

from typing import Optional


class GoodsMarketClearer(ABC):
    def __init__(
        self,
        real_country_prioritisation: float,
        prio_high_prio_buyers: bool,
        prio_high_prio_sellers: bool,
        prio_domestic_sellers: bool,
        probability_keeping_previous_seller: float,
        price_temperature: float,
        trade_temperature: float,
        seller_selection_distribution_type: str,
        seller_minimum_fill: float,
        buyer_minimum_fill_macro: float,
        buyer_minimum_fill_micro: float,
        deterministic: bool,
        consider_trade_proportions: bool,
        consider_buyer_priorities: bool,
        additionally_available_factor: float,
        price_markup: float,
        remedy_rounding_errors: bool = True,
        allow_additional_row_exports: bool = True,
    ):
        self.real_country_prioritisation = max(0.0, min(1.0, real_country_prioritisation))
        self.real_country_prioritisation = real_country_prioritisation
        self.prio_high_prio_buyers = prio_high_prio_buyers
        self.prio_high_prio_sellers = prio_high_prio_sellers
        self.prio_domestic_sellers = prio_domestic_sellers
        self.probability_keeping_previous_seller = probability_keeping_previous_seller
        self.price_temperature = price_temperature
        self.trade_temperature = trade_temperature
        self.seller_selection_distribution_type = seller_selection_distribution_type
        self.seller_minimum_fill = seller_minimum_fill
        self.buyer_minimum_fill_macro = buyer_minimum_fill_macro
        self.buyer_minimum_fill_micro = buyer_minimum_fill_micro
        self.deterministic = deterministic
        self.consider_trade_proportions = consider_trade_proportions
        self.consider_buyer_priorities = consider_buyer_priorities
        self.additionally_available_factor = additionally_available_factor
        self.price_markup = price_markup
        self.remedy_rounding_errors = remedy_rounding_errors
        self.allow_additional_row_exports = allow_additional_row_exports

    @staticmethod
    def prepare(goods_market_participants: dict[str, list[Agent]]) -> None:
        for country_name in goods_market_participants.keys():
            for transactor in goods_market_participants[country_name]:
                transactor.prepare()

    @staticmethod
    def collect_all_supply_and_demand(
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        buyer_high_prio_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            total_real_supply,
            aggr_real_supply,
            total_nominal_supply,
            aggr_nominal_supply,
            average_goods_price,
        ) = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
        )
        (
            total_real_demand,
            aggr_real_demand,
            total_nominal_demand,
            aggr_nominal_demand,
        ) = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=average_goods_price,
            n_industries=n_industries,
            high_prio_only=buyer_high_prio_only,
        )
        return aggr_nominal_supply, aggr_nominal_demand

    @abstractmethod
    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        pass

    @staticmethod
    def record(goods_market_participants: dict[str, list[Agent]]) -> None:
        for country_name in goods_market_participants.keys():
            for transactor in goods_market_participants[country_name]:
                transactor.record()


class NoGoodsMarketClearer(GoodsMarketClearer):
    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        pass


class DefaultGoodsMarketClearer(GoodsMarketClearer):
    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        for g in range(n_industries):
            # Check if there are any buyers or sellers left
            if (
                not check_buyers_left(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    field="Remaining Goods",
                )
            ) or (
                not check_sellers_left(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    field="Remaining Goods",
                )
            ):
                continue

            # Buy and sell
            while True:
                # Get a random buyer
                buyer, buyer_ind = get_random_buyer(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio=self.prio_high_prio_buyers,
                    field="Remaining Goods",
                )

                # Get a random seller
                seller, seller_ind = get_random_seller(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    chosen_buyer=buyer,
                    chosen_buyer_ind=buyer_ind,
                    previous_supply_chain=previous_supply_chain,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio_sellers=self.prio_high_prio_sellers,
                    prio_domestic_sellers=self.prio_domestic_sellers,
                    probability_keeping_previous_seller=self.probability_keeping_previous_seller,
                    price_temperature=self.price_temperature,
                    field="Remaining Goods",
                    distribution_type=self.seller_selection_distribution_type,
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
                    current_supply_chain=current_supply_chain,
                    industry=g,
                    buyer=buyer,
                    buyer_ind=buyer_ind,
                    seller=seller,
                    seller_ind=seller_ind,
                )

                # Check if we are done
                if (
                    not check_buyers_left(
                        industry=g,
                        goods_market_participants=goods_market_participants,
                        field="Remaining Goods",
                    )
                ) or (
                    not check_sellers_left(
                        industry=g,
                        goods_market_participants=goods_market_participants,
                        field="Remaining Goods",
                    )
                ):
                    break

            # Distribute excess demand
            for country_name in goods_market_participants.keys():
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_buyer_states["Value Type"] != ValueType.NONE:
                        transactor.transactor_buyer_states["Remaining Excess Goods"] = (
                            transactor.transactor_buyer_states["Remaining Goods"].copy()
                        )
            while check_buyers_left(
                industry=g,
                goods_market_participants=goods_market_participants,
                field="Remaining Excess Goods",
            ):
                if not check_sellers_left(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    field="Remaining Goods",
                ):
                    break

                # Get a random buyer
                buyer, buyer_ind = get_random_buyer(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio=self.prio_high_prio_buyers,
                    field="Remaining Excess Goods",
                )

                # Get a random seller
                seller, seller_ind = get_random_seller(
                    industry=g,
                    goods_market_participants=goods_market_participants,
                    chosen_buyer=buyer,
                    chosen_buyer_ind=buyer_ind,
                    previous_supply_chain=previous_supply_chain,
                    real_country_prioritisation=self.real_country_prioritisation,
                    prio_high_prio_sellers=self.prio_high_prio_sellers,
                    prio_domestic_sellers=self.prio_domestic_sellers,
                    probability_keeping_previous_seller=self.probability_keeping_previous_seller,
                    price_temperature=self.price_temperature,
                    field="Remaining Excess Goods",
                    distribution_type=self.seller_selection_distribution_type,
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
            clean_rounding_errors(goods_market_participants=goods_market_participants)


class WaterBucketGoodsMarketClearer(GoodsMarketClearer):
    def clear(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        default_origin_trade_proportions: np.ndarray,
        default_destin_trade_proportions: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
        row_index: int = -1,
    ) -> None:
        n_countries = len(goods_market_participants.keys())

        # Get average prices
        average_prices_by_country = np.zeros((n_countries + 1, n_industries))
        for ind, country_name in enumerate(goods_market_participants.keys()):
            for gmp in goods_market_participants[country_name]:
                if gmp.transactor_settings["Seller Value Type"] != ValueType.NONE:
                    average_prices_by_country[ind] = gmp.ts.current("price_offered")
                    break
        (
            _,
            _,
            _,
            _,
            average_prices_by_country[-1],
        ) = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
        )
        assert np.all(average_prices_by_country[0:-1] > 0.0)

        # Clear according to trade proportions
        if self.consider_trade_proportions:
            origin_trade_proportions, destin_trade_proportions = get_trade_proportions(
                n_countries=n_countries,
                default_origin_trade_proportions=default_origin_trade_proportions,
                default_destin_trade_proportions=default_destin_trade_proportions,
                average_prices_by_country=average_prices_by_country,
                temperature=self.trade_temperature,
                real_country_prioritisation=self.real_country_prioritisation,
                row_index=row_index,
            )
            for c1 in range(n_countries):
                for c2 in range(n_countries):
                    self.perform_clearing(
                        goods_market_participants=goods_market_participants,
                        n_industries=n_industries,
                        average_prices_by_country=average_prices_by_country,
                        buyer_priorities=buyer_priorities,
                        start_country=c1,
                        end_country=c2,
                        origin_trade_proportions=origin_trade_proportions,
                        destin_trade_proportions=destin_trade_proportions,
                    )

        # Clear evenly
        self.perform_clearing(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
            average_prices_by_country=average_prices_by_country,
            buyer_priorities=buyer_priorities,
        )

        # Handle excess demand
        self.distribute_excess_demand_water_bucket(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
        )

        # Allow additional purchases from the RoW
        if self.allow_additional_row_exports and self.additionally_available_factor > 0.0:
            self.handle_additional_row_exports(
                goods_market_participants=goods_market_participants,
                n_industries=n_industries,
            )

    def perform_clearing(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
        average_prices_by_country: np.ndarray,
        buyer_priorities: dict[str, np.ndarray],
        start_country: Optional[int] = None,
        end_country: Optional[int] = None,
        origin_trade_proportions: Optional[np.ndarray] = None,
        destin_trade_proportions: Optional[np.ndarray] = None,
    ) -> None:
        if origin_trade_proportions is None or destin_trade_proportions is None:
            current_origin_trade_proportions = None
            current_destin_trade_proportions = None
        else:
            current_origin_trade_proportions = origin_trade_proportions[start_country, end_country]
            current_destin_trade_proportions = destin_trade_proportions[start_country, end_country]
        (
            total_real_supply,
            aggr_real_supply,
            _,
            _,
            emp_goods_prices,
        ) = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
            from_country=start_country,
            trade_proportions=current_destin_trade_proportions,
        )
        if start_country is None:
            emp_goods_prices[np.isnan(emp_goods_prices)] = average_prices_by_country[-1][np.isnan(emp_goods_prices)]
            emp_goods_prices[emp_goods_prices == 0.0] = average_prices_by_country[-1][emp_goods_prices == 0.0]
        else:
            emp_goods_prices[np.isnan(emp_goods_prices)] = average_prices_by_country[start_country][
                np.isnan(emp_goods_prices)
            ]
            emp_goods_prices[emp_goods_prices == 0.0] = average_prices_by_country[start_country][
                emp_goods_prices == 0.0
            ]
            emp_goods_prices[emp_goods_prices == 0.0] = average_prices_by_country[-1][emp_goods_prices == 0.0]
        (
            total_real_demand,
            aggr_real_demand,
            _,
            _,
        ) = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=emp_goods_prices,
            n_industries=n_industries,
            to_country=end_country,
            trade_proportions=current_origin_trade_proportions,
        )

        clear_water_bucket(
            goods_market_participants=goods_market_participants,
            buyer_priority=buyer_priorities,
            n_industries=n_industries,
            total_real_supply=total_real_supply,
            aggr_real_supply=aggr_real_supply,
            average_goods_price=emp_goods_prices,
            total_real_demand=total_real_demand,
            aggr_real_demand=aggr_real_demand,
            from_country=start_country,
            to_country=end_country,
            origin_trade_proportions=current_origin_trade_proportions,
            destin_trade_proportions=current_destin_trade_proportions,
            price_temperature=self.price_temperature,
            distribution_type=self.seller_selection_distribution_type,
            seller_minimum_fill=self.seller_minimum_fill,
            buyer_minimum_fill_macro=self.buyer_minimum_fill_macro,
            buyer_minimum_fill_micro=self.buyer_minimum_fill_micro,
            deterministic=self.deterministic,
            consider_buyer_priorities=self.consider_buyer_priorities,
        )

    def distribute_excess_demand_water_bucket(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
    ) -> None:
        # Collect initial values
        _, _, total_nominal_supply, aggr_nominal_supply, average_price = collect_seller_info(
            goods_market_participants=goods_market_participants,
            n_industries=n_industries,
            use_initial=True,
        )
        _, excess_real_demand, _, _ = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=average_price,
            n_industries=n_industries,
        )

        # Distribute excess demand
        for g in range(n_industries):
            if aggr_nominal_supply[g] == 0.0 or excess_real_demand[g] == 0.0:
                continue

            # Adjust the allocation
            current_alloc = np.array(
                [
                    excess_real_demand[g] * total_nominal_supply[country_name][g] / aggr_nominal_supply[g]
                    for country_name in goods_market_participants.keys()
                ]
            )
            current_alloc[-1] *= 1 - max(0.0, min(1.0, self.real_country_prioritisation))
            if current_alloc.sum() == 0.0:
                continue
            current_alloc *= excess_real_demand[g] / current_alloc.sum()

            # Distribute
            for country_ind, country_name in enumerate(goods_market_participants.keys()):
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                        ind = transactor.transactor_seller_states["Industries"] == g
                        if self.deterministic:
                            _, seller_priorities = get_seller_priorities_deterministic(
                                productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                prices=transactor.transactor_seller_states["Prices"][ind],
                                price_temperature=self.price_temperature,
                                distribution_type=self.seller_selection_distribution_type,
                            )
                        else:
                            _, seller_priorities = get_seller_priorities_stochastic(
                                productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                prices=transactor.transactor_seller_states["Prices"][ind],
                                price_temperature=self.price_temperature,
                                distribution_type=self.seller_selection_distribution_type,
                            )
                        transactor.transactor_seller_states["Real Excess Demand"][ind] = fill_buckets(
                            capacities=transactor.transactor_seller_states["Remaining Excess Goods"][ind],
                            fill_amount=current_alloc[country_ind],
                            priorities=seller_priorities,
                            minimum_fill=self.seller_minimum_fill,
                        )

    def handle_additional_row_exports(
        self,
        goods_market_participants: dict[str, list[Agent]],
        n_industries: int,
    ) -> None:
        # Collect initial ROW exports
        _, aggr_real_supply, _, _, average_price = collect_seller_info(
            goods_market_participants={"ROW": goods_market_participants["ROW"]},
            n_industries=n_industries,
            use_initial=True,
        )
        average_price *= 1 + self.price_markup
        additional_real_demand_by_country, additional_real_demand, _, _ = collect_buyer_info(
            goods_market_participants=goods_market_participants,
            average_price=average_price,
            n_industries=n_industries,
            high_prio_only=True,
            exclude_row=True,
        )

        # Distribute excess demand
        for g in range(n_industries):
            if aggr_real_supply[g] == 0.0 or additional_real_demand[g] == 0.0:
                continue

            # Iterate over countries
            for country_name in goods_market_participants.keys():
                if country_name == "ROW":
                    continue
                country_supply = (
                    self.additionally_available_factor
                    * aggr_real_supply.sum()
                    * additional_real_demand_by_country[country_name].sum()
                    / additional_real_demand.sum()
                    * additional_real_demand_by_country[country_name][g]
                    / additional_real_demand_by_country[country_name].sum()
                )
                if country_supply == 0.0:
                    continue
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_buyer_states["Priority"] == 1:
                        buyer_priorities = get_buyer_priorities(
                            n_buyers=transactor.transactor_buyer_states["Remaining Goods"].shape[0]
                        )
                        real_amount_bought = fill_buckets(
                            capacities=transactor.transactor_buyer_states["Remaining Goods"][:, g],
                            fill_amount=country_supply,
                            priorities=buyer_priorities,
                            minimum_fill=self.buyer_minimum_fill_micro,
                        )
                        if np.sum(real_amount_bought) == 0.0:
                            continue

                        # Update the buyer states
                        transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_amount_bought
                        transactor.transactor_buyer_states["Real Amount bought from ROW"][:, g] += real_amount_bought
                        transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                            average_price[g] * real_amount_bought
                        )
                        transactor.transactor_buyer_states["Nominal Amount spent on Goods from ROW"][:, g] += (
                            average_price[g] * real_amount_bought
                        )
                        transactor.transactor_buyer_states["Remaining Goods"][:, g] -= real_amount_bought

                        # Update the seller states
                        ind = goods_market_participants["ROW"][0].transactor_seller_states["Industries"] == g
                        goods_market_participants["ROW"][0].transactor_seller_states["Real Amount sold"][
                            ind
                        ] += real_amount_bought.sum() / np.sum(ind)
                        goods_market_participants["ROW"][0].transactor_seller_states[
                            "Real Amount sold to " + country_name
                        ][ind] += real_amount_bought.sum() / np.sum(ind)
                        goods_market_participants["ROW"][0].transactor_seller_states["Remaining Goods"][
                            ind
                        ] -= real_amount_bought.sum() / np.sum(ind)
