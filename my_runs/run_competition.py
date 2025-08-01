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

# %%

configuration = sample_configuration(
    t_max=TIMESTEPS,
    seed=seed,
    use_carbon_price=True,
    asymptotic_carbon_price=10.0,
    carbon_price_time_constant=5.0,
    carbon_price_t_init=4 * 3,
    bundles=substitution_bundles,
)

simulation = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)

# %%
# RUN SIMULATION
emitting_industries = ["B05a", "B05b", "B05c", "C19"]
electricity = ["D"]
improving_industries = [
    industry for industry in data.industries if industry[0] == "C" and industry not in emitting_industries
]
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


# %%
