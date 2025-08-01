# Simulation-Based Inference (SBI) Calibration Example

This folder demonstrates a complete **Simulation-Based Inference** workflow for calibrating the macroeconomic model using **Sequential Neural Posterior Estimation (SNPE)**. The approach uses neural networks to learn the relationship between model parameters and observables, enabling Bayesian parameter inference for complex simulation models.

## Overview

The calibration process consists of three main stages:

1. **Data Preparation**: Create preprocessed economic data (`create_pkl.py`)
2. **Sample Generation**: Run multiple simulations with varying parameters (`create_samples.py`)
3. **Posterior Training**: Train neural density estimator and build posterior (`train_model.py`)

## Workflow Details

### Stage 1: Data Preparation (`create_pkl.py`)

Creates a preprocessed data file for Canada (2014) using the macro_data package:

```python
def create_pickle(configuration, filename):
    creator = macro_data.DataWrapper.from_config(
        configuration=configuration, raw_data_path=INPUT_PATH, single_hfcs_survey=False
    )
    creator.save(filename)
```

**Key Configuration Parameters:**

```python
representative_year: int = 2014
aggregate_industries = False
single_firm_per_industry = True
use_disagg_can_2014_reader = True
scale = 10000
seed = 1

data_configuration = configuration_utils.default_data_configuration(
    countries=["CAN"],
    proxy_country_dict={"CAN": "FRA"},
    year=representative_year,
    aggregate_industries=aggregate_industries,
    single_firm_per_industry=single_firm_per_industry,
    scale=scale,
    seed=seed,
    use_disagg_can_2014_reader=use_disagg_can_2014_reader,
)
```

### Stage 2: Sample Generation (`create_samples.py`)

Generates training samples by running multiple simulations with different parameter values.

#### Parameter Space

The calibration focuses on **firm price-setting parameters**:

```python
def update_country_conf(country_conf: CountryConfiguration, params: np.ndarray) -> CountryConfiguration:
    country_conf = deepcopy(country_conf)
    
    # Firm price parameters
    country_conf.firms.functions.prices.parameters["price_setting_noise_std"] = params[0]
    country_conf.firms.functions.prices.parameters["price_setting_speed_gf"] = params[1]
    country_conf.firms.functions.prices.parameters["price_setting_speed_dp"] = params[2]
    country_conf.firms.functions.prices.parameters["price_setting_speed_cp"] = params[3]
    return country_conf
```

These parameters control:

- `price_setting_noise_std`: Standard deviation of price-setting noise
- `price_setting_speed_gf`: Speed of adjustment for general factors
- `price_setting_speed_dp`: Speed of adjustment for demand pressure
- `price_setting_speed_cp`: Speed of adjustment for cost pressure

In general, you can set any parameter you want to calibrate. The only thing you need to do is to update the `configuration_updater` function to set the parameters you want to calibrate. Here, I wrote a function that updates the parameters for a single country, and then a function that updates the configuration for all countries (in this case, all countries are updated in the same way, so they will all have the same parameters).

If you wanted to calibrate different parameters for different countries, you would need to write a function that updates the parameters for each country separately.

To understand all the parameters you can calibrate, you can look at the `SimulationConfiguration` class in the `macromodel` package, which has a hierarchical structure with sub-classes for configurations for each country, and for each agent within a country.

#### Observable Functions

The model extracts two key economic indicators:

```python
def observer(simulation: Simulation) -> np.ndarray:
    country = simulation.countries["CAN"]
    
    gdp_growth = np.diff(np.log(country.economy.gdp_output()))
    unemp_growth = np.diff(np.log(country.economy.unemployment_rate()))
    
    return np.array([np.nanmean(gdp_growth), np.nanmean(unemp_growth)])
```

Again, you can set any observable you want to calibrate. Here, I set the mean of the GDP growth and the mean of the unemployment growth.

You can look at the `shallow_output` function in the `macromodel` package to see simple observables that you can use. In practice, each observable should be a scalar (this calibration method does not work well with vector observables, such as time series or the whole distribution of a variable).

#### Prior Sampling

Uses uniform priors for all parameters:

```python
def prior_sampler(n_samples: int) -> np.ndarray:
    return np.random.uniform(0, 1, size=(n_samples, 4))
```

You can set any prior you want. Here, I set uniform priors for all parameters. The idea is that this will be used for sampling: the `Sampler` class will produce pairs `(theta, y)` where `theta` is a parameter vector and `y` is an observable vector. These are the pairs that will be used to train the neural density estimator.

#### Parallel Sampling Execution

The `Sampler` class provides parallel execution capabilities with flexible configuration options:

```python
sampler = Sampler.default(
    configuration_updater=configuration_updater,
    observer=observer,
    pickle_path=PKL_PATH,
)

samples = sampler.parallel_run(n_runs=20, prior_sampler=prior_sampler)
```

**Key Sampler Configuration Parameters:**

- **`configuration_updater`**: Function that updates simulation configuration based on parameter vectors
- **`observer`**: Function that extracts observables from completed simulations  
- **`pickle_path`**: Path to preprocessed data file (required)
- **`country_conf_path`**: Optional path to YAML country configuration file
- **`n_cores`**: Number of CPU cores for parallel execution (optional)
- **`countries`**: List of countries to include in simulation (optional)

**Parallel Execution Details:**

The `n_cores` parameter controls parallelization:

- **Default**: `n_cores = os.cpu_count() // 2` (half of available CPU cores)
- **Custom**: Users can specify any integer value for `n_cores`
- **Total samples**: `n_runs × n_cores` (each core runs `n_runs` simulations independently)

In this example with `n_runs=20`:

- If running on an 8-core machine: `n_cores=4` (default), total samples = 80
- Each core runs 20 simulations independently
- Results are collected and combined from all cores

From the `macrocalib` package, the `Sampler.parallel_run()` method:

```python
def parallel_run(self, n_runs: int, prior_sampler: PriorSampler) -> list[dict]:
    """Run sampling in parallel across multiple cores."""
    print(f"Running with {self.n_cores} cores")
    print(f"Running {n_runs} runs per core, total of {n_runs * self.n_cores} runs")
    
    def run_on_core(core_id: int) -> dict:
        return self.core_run(n_runs, prior_sampler, core_id)
    
    # Parallel execution using joblib
    results = Parallel(n_jobs=self.n_cores)(
        delayed(run_on_core)(core_id) for core_id in range(self.n_cores)
    )
    
    return results
```

**Performance Considerations:**

- Each core instantiates its own model to avoid memory sharing issues
- Cores run completely independently with separate parameter samples
- Design allows easy scaling by adjusting `n_cores` and `n_runs`

### Stage 3: Posterior Training (`train_model.py`)

Trains a neural density estimator to learn the parameter-observable relationship and builds a posterior distribution.

#### Data Preparation

Once everything is set up, you can run the sampler. The output will be a list of dictionaries, each containing the parameter vector and the observable vector for a single simulation.

You need to load them and concatenate them into a single array (I've kept an output per core in case of debugging), and then transform them into PyTorch tensors that are compatible with the `sbi` library.

```python
# Load samples and concatenate results
with open("samples.pkl", "rb") as f:
    samples = pkl.load(f)

simulated_data = np.concatenate([np.stack(sample["simulations"]) for sample in samples])
thetas = np.concatenate([np.stack(sample["thetas"]) for sample in samples])

# Convert to PyTorch tensors
simulated_data = torch.from_numpy(simulated_data)
thetas = torch.from_numpy(thetas)
```

#### Neural Posterior Training

Uses the `macrocalib` training functions, which are wrappers around the `sbi` library.

```python
inference, density_estimator = train_model(simulated_data, thetas)
posterior = build_posterior(inference, density_estimator)
```

From the `macrocalib.training.train` module:

```python
def train_model(simulation_data: torch.Tensor, theta: torch.Tensor) -> tuple[SNPE, nn.Module]:
    """Train a neural posterior estimator using SNPE."""
    
    # Convert to float32 for compatibility
    simulation_data = simulation_data.to(torch.float32)
    theta = theta.to(torch.float32)
    
    # Initialize SNPE with default neural network
    inference = SNPE()
    
    # Train the density estimator
    density_estimator = inference.append_simulations(theta, simulation_data).train()
    
    return inference, density_estimator

def build_posterior(inference: SNPE, density_estimator: nn.Module, **kwargs) -> Posterior:
    """Build posterior distribution from trained components."""
    return inference.build_posterior(density_estimator, **kwargs)
```

#### Posterior Sampling

Generate parameter samples for specific target observations.  Here, I'm sampling 10 parameter vectors for a target of 1% GDP growth and 0% unemployment growth. In practice, you would use the posterior to sample parameter vectors for a target defined from real data (or from some target that you want to reproduce).

```python
# Sample parameters for 1% GDP growth and 0% unemployment growth
samples = posterior.sample(sample_shape=(10,), x=torch.Tensor([0.01, 0]))
```

The samples are then vectors that you can use to generate `SimulationConfiguration` objects. For this, you can take an existing `SimulationConfiguration` object and update it with the parameter vectors using the same `configuration_updater` function you used for the sampler.

```python
# Update the configuration with the parameter vectors to get a single simulation configuration from the first sample
configuration = configuration_updater(configuration, samples[0])
```

You would then be able to run the simulation with the `Simulation` class.

```python
model = Simulation.from_datawrapper(
    datawrapper=self.datawrapper,
    simulation_configuration=configuration,
)

model.run()
```

And then measure whatever observable you want to measure, or do plots with the model output.

#### Posterior Persistence

```python
# Save posterior for later use
with open("posterior.pkl", "wb") as f:
    pkl.dump(posterior, f)
```

## Technical Implementation

### Macrocalib Package Architecture

The calibration relies on the `macrocalib` package with two main components:

**Sampler Class** (`macrocalib.sampler.Sampler`):

- Wraps macromodel simulations for parallel execution
- Handles configuration updates and observable extraction
- Provides parallel sampling infrastructure using joblib

**Training Module** (`macrocalib.training.train`):

- Implements SNPE using the `sbi` library
- Trains neural density estimators to approximate posterior distributions
- Provides utilities for posterior construction and sampling

### Key Dependencies

- **SBI Library**: Sequential Neural Posterior Estimation implementation
- **PyTorch**: Neural network training and tensor operations
- **Joblib**: Parallel processing for sample generation
- **Macromodel**: Core economic simulation framework

## Usage Notes

1. **Data Scaling**: The model uses a scale factor of 10,000 to ensure numerical stability
2. **Single Firm per Industry**: Simplified market structure for faster computation
3. **Parallel Execution**: Samples are generated in parallel across multiple CPU cores
4. **Neural Network Training**: Uses default SNPE configuration with automatic hyperparameter selection

## Expected Outputs

For this example folder, you should get the following files:

- `data.pkl`: Preprocessed economic data for Canada 2014
- `samples.pkl`: Parameter-observable pairs from simulation runs  
- `posterior.pkl`: Trained posterior distribution for parameter inference
- `sbi-logs/`: TensorBoard logs from neural network training

This workflow enables robust parameter calibration for complex macroeconomic models by leveraging modern simulation-based inference techniques.
