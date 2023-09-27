from data.processing.synthetic_housing_market.synthetic_housing_market import (
    SyntheticHousingMarket,
)


class DefaultSyntheticHousingMarket(SyntheticHousingMarket):
    def create(self) -> None:
        pass

    def set_initial_conditions(self) -> None:
        self.housing_market_data["Newly on the Rental Market"] = False
