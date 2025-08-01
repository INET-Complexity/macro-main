# %%
INPUT_PATH = "/Users/jmoran/Projects/macrocosm/inet/data/raw_data/"
PKL_PATH = "data.pkl"
import os

import numpy as np
import pandas as pd
from generate_configuration import sample_configuration

import macro_data
import macromodel
from macro_data import configuration_utils
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

print(INPUT_PATH)
print(PKL_PATH)

TIMESTEPS = 40
seed = 0

# %%

data = macro_data.DataWrapper.init_from_pickle(PKL_PATH)

SCENARIO = "substitution"

OUTPUT_Directory = "./output/"

n_industries = data.n_industries
bundled_industries = ["B05a", "B05b", "B05c", "C19", "D"]

# %%

industries = data.industries
energy_bundle = [list(industries).index(ind) for ind in bundled_industries]

substitution_bundles = [energy_bundle]


emitting_industries = ["B05a", "B05b", "B05c", "C19"]
electricity = ["D"]
improving_industries = [
    industry for industry in data.industries if industry[0] == "C" and industry not in emitting_industries
]
# %%


def get_production_dataframe(sim: Simulation, country: str = "CAN"):
    production = np.stack(sim.countries[country].firms.ts.historic("production"))
    price = np.stack(sim.countries[country].firms.ts.historic("price"))
    production_price = production * price
    return pd.DataFrame(production_price, columns=data.industries)


def process_simulation(
    use_carbon_price: bool = False,
    asymptotic_carbon_price: float = 10.0,
    carbon_price_t_init=4 * 3,
    fixed_sample: int = 10,
    t_max: int = TIMESTEPS,
):
    if not use_carbon_price:
        asymptotic_carbon_price = 0.0
        carbon_price_time_constant = 0.0
        carbon_price_t_init = 0
        use_carbon_price = True

    configuration = sample_configuration(
        t_max=t_max,
        seed=seed,
        use_carbon_price=use_carbon_price,
        asymptotic_carbon_price=asymptotic_carbon_price,
        carbon_price_time_constant=5.0,
        carbon_price_t_init=carbon_price_t_init,
        bundles=substitution_bundles,
        fixed_sample=fixed_sample,
    )

    simulation = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)

    simulation.run()

    production = get_production_dataframe(simulation, country="CAN")

    gdp = simulation.countries["CAN"].economy.gdp_output()

    gdp = pd.DataFrame(gdp, columns=["GDP"])

    return production, gdp


# %%
import matplotlib.pyplot as plt

production, gdp = process_simulation(fixed_sample=10)

gdp.plot()

# %%

production, gdp = process_simulation(fixed_sample=22)


gdp.plot()

# %%

production2, gdp2 = process_simulation(
    fixed_sample=22, use_carbon_price=True, asymptotic_carbon_price=10.0, carbon_price_t_init=4 * 3
)


gdp2.plot()


# %%

fig, ax = plt.subplots()

production[emitting_industries].plot(ax=ax)
production2[emitting_industries].plot(ax=ax)


# %%
# b05a coal, b05b gas, b05c oil, c19 refined


def rename_production(production: pd.DataFrame):
    production.rename(
        columns={"B05a": "Coal", "B05b": "Gas", "B05c": "Oil", "C19": "Refined oil products", "D": "Electricity"},
        inplace=True,
    )
    # replace index 0,1..,N by a datetime index running quarterly starting in Jan 2014
    production.index = pd.date_range(start="2014-01-01", periods=len(production), freq="Q")

    return production


# %%

fig, ax = plt.subplots(figsize=(10, 5))

production, gdp = process_simulation(fixed_sample=22, t_max=200)

production = rename_production(production)

industries_to_plot = ["Coal", "Gas", "Oil", "Refined oil products"]

colors = ["red", "blue", "green", "orange", "purple"]

for industry, color in zip(industries_to_plot, colors):
    production[industry].plot(ax=ax, color=color, lw=2)

for carbon_price in [100]:
    production_bis, gdp_bis = process_simulation(
        fixed_sample=22,
        use_carbon_price=True,
        asymptotic_carbon_price=carbon_price,
        carbon_price_t_init=4 * 3,
        t_max=200,
        label=f"GDP with {carbon_price}USD/ton carbon price",
    )
    production_bis = rename_production(production_bis)
    for industry, color in zip(industries_to_plot, colors):
        production_bis[industry].plot(ax=ax, color=color, alpha=0.5)

# add legend with proper colors

ax.legend(industries_to_plot)
ax.set_ylabel("Production (output in USD)")


# %%

fig, ax = plt.subplots(figsize=(10, 5))

production, gdp = process_simulation(fixed_sample=22)

production = rename_production(production)

industries_to_plot = ["Coal", "Gas", "Oil", "Refined oil products"]

colors = ["red", "blue", "green", "orange", "purple"]

for industry, color in zip(industries_to_plot, colors):
    (production[industry] / production[industry].iloc[0]).plot(ax=ax, color=color, lw=2)

for carbon_price in [100]:
    production_bis, gdp_bis = process_simulation(
        fixed_sample=22, use_carbon_price=True, asymptotic_carbon_price=carbon_price, carbon_price_t_init=4 * 3
    )
    production_bis = rename_production(production_bis)
    for industry, color in zip(industries_to_plot, colors):
        (production_bis[industry] / production[industry].iloc[0]).plot(ax=ax, color=color, alpha=0.5)

# add legend with proper colors

ax.legend(industries_to_plot)

ax.set_ylabel("Production")


# %%

gdp_growth = []

for i in range(100):
    production, gdp = process_simulation(fixed_sample=i)
    gdp_growth.append(np.log(gdp).diff().mean())


# %%


def add_productivity_improvements(
    industries_to_improve: list[str],
    inputs_to_improve: list[str],
    productivity_improvement: float,
    model: Simulation,
    country: str = "CAN",
):
    for industry in industries_to_improve:
        for input in inputs_to_improve:
            model.countries[country].firms.increase_industry_input_productivity(
                producing_industry=industry,
                input_industry=input,
                increase_pct=productivity_improvement,
            )


def add_global_productivity_improvement(
    productivity_improvement: float, model: Simulation, country: str = "CAN", industries=None
):
    if industries is None:
        industries = improving_industries
    else:
        industries = data.industries
    for industry in industries:
        model.countries[country].firms.global_capital_productivity_increase(industry, productivity_improvement)
        model.countries[country].firms.global_input_productivity_increase(industry, productivity_improvement)


# %%


def run_with_productivity_improvement(
    sim: Simulation,
    global_productivity_improvement: float,
    focused_productivity_improvement: float,
    country: str = "CAN",
):
    sim.reset()
    for _ in range(sim.t_max - 1):
        sim.iterate()
        # add_global_productivity_improvement(
        #     global_productivity_improvement,
        #     sim,
        #     country,
        #     industries=improving_industries,
        # )
        if _ > 10:
            add_productivity_improvements(
                improving_industries,
                emitting_industries,
                focused_productivity_improvement,
                sim,
                country,
            )


# %%
def get_production_dataframe(sim: Simulation, country: str = "CAN"):
    production = np.stack(sim.countries[country].firms.ts.historic("production"))
    return pd.DataFrame(production, columns=data.industries)


# %%
global_productivity_improvement = 0.02
focused_productivity_improvement = 0.10

simulation.reset()
run_with_productivity_improvement(
    simulation,
    global_productivity_improvement,
    focused_productivity_improvement,
    country="CAN",
)

production = get_production_dataframe(simulation, country="CAN")


# %%
fig, ax = plt.subplots()


_, gdp = process_simulation(fixed_sample=22, t_max=200)

_, gdp_bis = process_simulation(
    fixed_sample=22, t_max=200, use_carbon_price=True, asymptotic_carbon_price=100, carbon_price_t_init=4 * 3
)

gdp.index = pd.date_range(start="2014-01-01", periods=len(gdp), freq="Q", name="GDP (no carbon price)")
gdp_bis.index = gdp.index
gdp.plot(ax=ax, color="blue")
gdp_bis.plot(ax=ax, alpha=0.5, color="red", label="GDP (with carbon price)")

# %%
fig, ax = plt.subplots()

gdp.rename(columns={"GDP": "GDP (no carbon price)"}, inplace=True)
gdp_bis.rename(columns={"GDP": "GDP (with carbon price)"}, inplace=True)

gdp.index = pd.date_range(start="2024-01-01", periods=len(gdp), freq="Q")
gdp_bis.index = gdp.index
gdp.loc["2024":].plot(ax=ax, color="blue", label="GDP (no carbon price)")
gdp_bis.loc["2024":].plot(ax=ax, alpha=0.5, color="red", label="GDP (with carbon price)")


# %%

fig, ax = plt.subplots()

gdp.rename(columns={"GDP": "GDP (no carbon price)"}, inplace=True)
gdp_bis.rename(columns={"GDP": "GDP (with carbon price)"}, inplace=True)

gdp.index = pd.date_range(start="2014-01-01", periods=len(gdp), freq="Q")
gdp_bis.index = gdp.index
gdp.loc["2014":].plot(ax=ax, color="blue", label="GDP (no carbon price)")
gdp_bis.loc["2014":].plot(ax=ax, alpha=0.5, color="red", label="GDP (with carbon price)")

# %%
fig, ax = plt.subplots()

gdp.rename(columns={"GDP": "GDP (no carbon price)"}, inplace=True)
gdp_bis.rename(columns={"GDP": "GDP (with carbon price)"}, inplace=True)

gdp.index = pd.date_range(start="2024-01-01", periods=len(gdp), freq="Q")
gdp_bis.index = gdp.index
gdp.loc["2024":"2040"].plot(ax=ax, color="blue", label="GDP (no carbon price)")
gdp_bis.loc["2024":"2040"].plot(ax=ax, alpha=0.5, color="red", label="GDP (with carbon price)")

# %%
