import tempfile
from pathlib import Path

from macromodel.configurations import SimulationConfiguration, CountryConfiguration
from macromodel.simulation import Simulation


def test_simulation(datawrapper):
    """Test the simulation."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    for _ in range(5):
        simulation.iterate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        simulation.save(save_dir=tmp, file_name="simulation.pkl")

    assert True
