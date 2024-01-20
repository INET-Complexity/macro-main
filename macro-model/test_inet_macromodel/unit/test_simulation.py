from configurations import SimulationConfiguration, CountryConfiguration
from simulation import Simulation


def test_simulation(datawrapper):
    """Test the simulation."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    simulation.iterate()

    assert True
