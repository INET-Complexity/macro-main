# %%

from pathlib import Path

import numpy as np
import yaml

from macro_data.configuration import DataConfiguration
from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.data_wrapper import DataWrapper

RAW_DATA_PATH = "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data"

DATA_CONFIG_PATH = Path(
    "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/default_data_config.yaml"
)

with open(DATA_CONFIG_PATH, "r") as f:
    config_dict = yaml.safe_load(f)

configuration = DataConfiguration(**config_dict)
configuration.can_disaggregation = False
configuration.aggregate_industries = False
configuration.prune_date = None
configuration.seed = 0

# Get the base configuration (France's config) to copy for all regions
france = Country("FRA")
base_config = configuration.country_configs[france]

base_config.single_firm_per_industry = True
base_config.single_bank = True
base_config.single_government_entity = True

base_config.firms_configuration.constructor = "Default"

base_config.scale = 1000

# Define Canadian provinces
provinces = [
    Region.from_code("CAN_AB", "Alberta"),
    Region.from_code("CAN_BC", "British Columbia"),
    Region.from_code("CAN_MB", "Manitoba"),
    Region.from_code("CAN_NB", "New Brunswick"),
    Region.from_code("CAN_NL", "Newfoundland and Labrador"),
    Region.from_code("CAN_NS", "Nova Scotia"),
    Region.from_code("CAN_ON", "Ontario"),
    Region.from_code("CAN_PE", "Prince Edward Island"),
    Region.from_code("CAN_QC", "Quebec"),
    Region.from_code("CAN_SK", "Saskatchewan"),
]

# Add Canada as the parent country
canada = Country("CAN")
configuration.country_configs[canada] = base_config
configuration.country_configs[canada].eu_proxy_country = france

# Add configurations for all provinces
for province in provinces:
    configuration.country_configs[province] = base_config
    configuration.country_configs[province].eu_proxy_country = france

# Set up the aggregation structure
configuration.aggregation_structure = {canada: provinces}

# Remove France's config since we don't need it for this test
del configuration.country_configs[france]


raw_data_path = RAW_DATA_PATH
data_wrapper = DataWrapper.from_config(
    configuration=configuration,
    raw_data_path=raw_data_path,
    single_hfcs_survey=True,
)

regions = list(data_wrapper.synthetic_countries.keys())


# GDP components are below

# total_sales = (self.firms.firm_data["Production"] * self.firms.firm_data["Price"]).sum()
# used_intermediate_inputs = self.firms.used_intermediate_inputs
# used_intermediate_inputs_costs = np.matmul(self.firms.firm_data["Price"].values, used_intermediate_inputs).sum()

# total_taxes_on_products = self.central_government.central_gov_data["Taxes on Products"].values[0]
# total_taxes_on_production = self.central_government.central_gov_data["Taxes on Production"].values[0]

# rent = self.population.household_data["Rent Paid"].sum()
# imputed_rent = self.population.household_data["Rent Imputed"].sum()

# a function for each component

# %%


def total_sales(region: str) -> float:
    return (
        data_wrapper.synthetic_countries[region].firms.firm_data["Production"]
        * data_wrapper.synthetic_countries[region].firms.firm_data["Price"]
    ).sum()


def used_intermediate_inputs_costs(region: str) -> float:
    return np.matmul(
        data_wrapper.synthetic_countries[region].firms.firm_data["Price"].values,
        data_wrapper.synthetic_countries[region].firms.used_intermediate_inputs,
    ).sum()


def total_taxes_on_products(region: str) -> float:
    return data_wrapper.synthetic_countries[region].central_government.central_gov_data["Taxes on Products"].values[0]


def total_taxes_on_production(region: str) -> float:
    return data_wrapper.synthetic_countries[region].central_government.central_gov_data["Taxes on Production"].values[0]


def rent(region: str) -> float:
    return data_wrapper.synthetic_countries[region].population.household_data["Rent Paid"].sum()


def imputed_rent(region: str) -> float:
    return data_wrapper.synthetic_countries[region].population.household_data["Rent Imputed"].sum()


# %%
import pandas as pd

total_sales_df = pd.DataFrame([(region, total_sales(region)) for region in regions])
total_sales_df.columns = ["region", "total_sales"]
total_sales_df.set_index("region", inplace=True)


used_intermediate_inputs_costs_df = pd.DataFrame(
    [(region, used_intermediate_inputs_costs(region)) for region in regions]
)
used_intermediate_inputs_costs_df.columns = ["region", "used_intermediate_inputs_costs"]
used_intermediate_inputs_costs_df.set_index("region", inplace=True)

total_taxes_on_products_df = pd.DataFrame([(region, total_taxes_on_products(region)) for region in regions])
total_taxes_on_products_df.columns = ["region", "total_taxes_on_products"]
total_taxes_on_products_df.set_index("region", inplace=True)

total_taxes_on_production_df = pd.DataFrame([(region, total_taxes_on_production(region)) for region in regions])
total_taxes_on_production_df.columns = ["region", "total_taxes_on_production"]
total_taxes_on_production_df.set_index("region", inplace=True)

rent_df = pd.DataFrame([(region, rent(region)) for region in regions])
rent_df.columns = ["region", "rent"]
rent_df.set_index("region", inplace=True)

imputed_rent_df = pd.DataFrame([(region, imputed_rent(region)) for region in regions])
imputed_rent_df.columns = ["region", "imputed_rent"]
imputed_rent_df.set_index("region", inplace=True)


# %%
total_sales_df.plot(kind="bar")


# %%
used_intermediate_inputs_costs_df.plot(kind="bar")

# %%
total_taxes_on_products_df.plot(kind="bar")

# %%

rent_df.plot(kind="bar")

# %%

imputed_rent_df.plot(kind="bar")


# %%

(total_sales_df - used_intermediate_inputs_costs_df).plot(kind="bar")

# %%


(total_taxes_on_products_df - total_taxes_on_production_df).plot(kind="bar")

# %%
