import os
from pathlib import Path
from typing import Callable, Optional, Protocol, Self

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed

from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation


class PriorSampler(Protocol):
    """
    A protocol for a prior sampler.

    A prior sampler is a function that generates n_samples of parameters (as vectors),
    from a predefined prior distribution.

    The protocol allows to use duck-typing to define such a function, as a function that takes an integer n
    and returns a numpy array.
    """

    def __call__(self, n: int) -> np.ndarray: ...


class Sampler:
    """
    A class to sample from the macromodel.
    This is simply a wrapper that allows you to sample parameters from a prior and run simulations in parallel.
    You can use this to produce a dictionary of thetas (parameters) and simulation observables.
    This is done by instantiating a model in each core of the machine, and doing the sampling in parallel in each core.

    You need to specify the following:
    - A baseline configuration for the simulation
    - A function that updates the configuration based on a vector of parameters
    - A prior, namely a function that generates n_samples of parameters (as vectors)
    - An observer, namely a function that takes a simulation and returns a vector of observables

    """

    def __init__(
        self,
        simulation_configuration: SimulationConfiguration,
        datawrapper: DataWrapper,
        n_cores: int,
        configuration_updater: Callable[[SimulationConfiguration, np.ndarray], SimulationConfiguration],
        observer: Callable[[Simulation], np.ndarray],
    ):
        self.base_configuration = simulation_configuration
        self.n_cores = n_cores
        self.datawrapper = datawrapper
        self.configuration_updater = configuration_updater
        self.observer = observer

    @classmethod
    def default(
        cls,
        configuration_updater: Callable[..., SimulationConfiguration],
        observer: Callable[[Simulation], np.ndarray],
        pickle_path: Path | str,
        country_conf_path: Optional[Path | str] = None,
        n_cores: Optional[int] = None,
        countries: Optional[list[str]] = None,
    ) -> Self:
        """
        A default constructor for the sampler.
        A DataWrapper is instantiated from a pickle file, and a
        baseline SimulationConfiguration is instantiated from a yaml file that contains the configuration
        for a single country. If no country configuration is specified, the default configuration is used.

        The constructor also takes the number of cores that will be used during the sampling as a parameter. The default
        is half the number of cores in the machine.

        Parameters
        ----------
        configuration_updater : Callable[..., SimulationConfiguration]
            A function that updates the baseline configuration based on a vector of parameters.
        observer : Callable[[Simulation], np.ndarray]
            A function that takes a simulation and returns a vector of observables.
        pickle_path : Path | str
            Path to the pickle file that contains the data.
        country_conf_path : Optional[Path | str], optional
            Path to the yaml file that contains the country configuration. Defaults to the default configuration.
        n_cores : Optional[int], optional
            Number of cores to use during the sampling. Defaults to half the number of cores in the machine.
        countries : Optional[list[str]], optional
            List of countries to use in the simulation, by default None.
        """
        if isinstance(pickle_path, str):
            pickle_path = Path(pickle_path)
        if isinstance(country_conf_path, str):
            country_conf_path = Path(country_conf_path)

        data = DataWrapper.init_from_pickle(path=pickle_path)

        if countries is None:
            countries = list(data.synthetic_countries.keys())

        if country_conf_path is not None:
            with open(country_conf_path, "r") as f:
                country_conf_dict = yaml.safe_load(f)

            country_conf = CountryConfiguration(**country_conf_dict)

        else:
            country_conf = CountryConfiguration.n_industry_default(data.n_industries)
        country_configurations = {country: country_conf for country in countries}

        configuration = SimulationConfiguration(country_configurations=country_configurations, t_max=15, seed=0)

        if n_cores is None:
            n_cores = os.cpu_count() // 2

        return cls(
            simulation_configuration=configuration,
            datawrapper=data,
            n_cores=n_cores,
            configuration_updater=configuration_updater,
            observer=observer,
        )

    def instantiate_model(self) -> Simulation:
        """
        Instantiates a model from the datawrapper and the baseline configuration.
        """
        model = Simulation.from_datawrapper(
            datawrapper=self.datawrapper,
            simulation_configuration=self.base_configuration,
        )
        return model

    def simulator_from_model(self, model: Simulation) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a simulator function that takes a vector of new parameters and returns a vector of observables,
        computed using those parameters. The observables are computed from a simulation run using the observer function.

        Parameters
        ----------
        model : Simulation
            The model that will be used to run the simulation.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            A simulator function that takes a vector of new parameters and returns a vector of observables.
        """

        def simulator(theta: np.ndarray) -> np.ndarray:
            new_configuration = self.configuration_updater(self.base_configuration, theta)
            model.reset(configuration=new_configuration)
            model.run()
            return self.observer(model)

        return simulator

    def core_run(self, n_runs: int, prior: PriorSampler) -> dict[str, np.ndarray]:
        """
        Runs the simulation in a single core. This function will be called in parallel in each core.

        Parameters
        ----------
        n_runs : int
            Number of runs to perform in this core.
        prior : PriorSampler
            A function that generates n_samples of parameters (as vectors), from a predefined prior distribution.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing the sampled parameters and the simulation observables, with keys "thetas"
            for the parameters and "simulations" for the observables.
        """
        model = self.instantiate_model()
        thetas = prior(n_runs)
        sim = self.simulator_from_model(model)
        simulations = [sim(theta) for theta in thetas]
        return {"thetas": thetas, "simulations": simulations}

    def parallel_run(self, n_runs: int, prior_sampler: PriorSampler) -> list[np.ndarray]:
        """
        Runs the simulation in parallel in all cores.

        Parameters
        ----------
        n_runs : int
            Number of runs to perform in each core.
        prior_sampler : PriorSampler
            A function that generates n_samples of parameters (as vectors), from a predefined prior distribution.

        Returns
        -------
        list[np.ndarray]
            A list of dictionaries containing the sampled parameters and the simulation observables, with keys "thetas"
            for the parameters and "x" for the observables.
        """
        print(f"Running with {self.n_cores} cores")
        print(f"Running {n_runs} runs per core, total of {n_runs * self.n_cores} runs")
        dic_list = Parallel(
            n_jobs=self.n_cores,
        )(delayed(self.core_run)(n_runs, prior_sampler) for _ in range(self.n_cores))
        # dic_list = [self.core_run(n_runs, prior_sampler) for _ in range(self.n_cores)]
        return dic_list

    def simple_run(self, theta: np.ndarray, model: Optional[Simulation] = None) -> Simulation:
        """
        Runs a single simulation with a given set of parameters.

        Parameters
        ----------
        theta : np.ndarray
            The parameters for the simulation.
        model : Optional[Simulation], optional
            The model to use for the simulation. If None, a new model is instantiated. Defaults to None.

        Returns
        -------
        Simulation
            The model after the simulation has been run.
        """
        if not model:
            model = self.instantiate_model()
        new_configuration = self.configuration_updater(self.base_configuration, theta)
        model.reset(configuration=new_configuration)
        model.run()
        return model


def process_runs(runs_dict: dict) -> (torch.tensor, torch.tensor):
    """
    A function to unpack the list of dictionaries produced by the parallel_run method of the Sampler class.
    The function takes a list of dictionaries, each containing the sampled parameters and the simulation observables,
    It returns a tuple of torch tensors, one containing the simulation observables and the other containing the sampled
    parameters.

    The number of rows in the tensors is the sum of the number of runs in each dictionary in the list, or equivalently
    the total number of samples produced by the parallel_run method.

    Parameters
    ----------
    runs_dict : dict
        A list of dictionaries containing the sampled parameters and the simulation observables.

    Returns
    -------
    (torch.tensor, torch.tensor)
        A tuple of torch tensors, one containing the simulation observables and the other containing the sampled
    """
    x = torch.cat([torch.tensor(np.stack(res["simulations"])) for res in runs_dict])

    theta = torch.cat([torch.tensor(res["thetas"]) for res in runs_dict])

    return x, theta
