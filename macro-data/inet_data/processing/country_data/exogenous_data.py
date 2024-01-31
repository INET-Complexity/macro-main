from dataclasses import dataclass
from pandas import DataFrame


#


@dataclass
class ExogenousCountryData:
    log_inflation: DataFrame
    sectoral_growth: DataFrame
    unemployment_rate: DataFrame
    house_price_index: DataFrame
    vacancy_rate: DataFrame
    total_firm_deposits_and_debt: DataFrame
    iot_industry_data: DataFrame
