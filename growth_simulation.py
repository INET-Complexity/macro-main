from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

RAW_DATA_PATH = "/Users/jmoran/Projects/macrocosm/inet/data/raw_data"

data = DataWrapper.init_from_pickle("./canada_disagg.pkl")

country_config = CountryConfiguration.n_industry_default(n_industries=data.n_industries)

country_config.labour_market.functions.clearing.parameters["random_firing_probability"] = 1e-3


country_config.use_carbon_price = True
country_config.asymptotic_carbon_price = 100
country_config.carbon_price_time_constant = 2

configuration = SimulationConfiguration(
    country_configurations={
        "CAN": country_config,
    },
    t_max=50,
    seed=0,
)

model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)

emitting_industries = ["B05a", "B05b", "B05c", "C19"]

# industries that will improve are industries that start with C and that are not in emitting industries
improving_industries = [
    industry for industry in data.industries if industry[0] == "C" and industry not in emitting_industries
]


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


def run_with_productivity_improvement(
    sim: Simulation,
    productivity_improvement: float,
    country: str = "CAN",
):
    sim.reset()
    for _ in range(sim.t_max):
        sim.iterate()
        add_productivity_improvements(
            improving_industries,
            emitting_industries,
            productivity_improvement,
            sim,
            country,
        )


run_with_productivity_improvement(model, 0.05)
shallow_output = model.get_country_shallow_output("CAN")

shallow_output.to_csv("/Users/jmoran/Projects/macrocosm/macromodel/runs/growth_runs/growth_simulation.csv")

run_with_productivity_improvement(model, -0.05)
shallow_output = model.get_country_shallow_output("CAN")
shallow_output.to_csv("/Users/jmoran/Projects/macrocosm/macromodel/runs/growth_runs/neg_growth_simulation.csv")

model.reset()
model.run()
shallow_output = model.get_country_shallow_output("CAN")
shallow_output.to_csv("/Users/jmoran/Projects/macrocosm/macromodel/runs/growth_runs/no_growth_simulation.csv")
