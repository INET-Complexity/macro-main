# Macromodel calibration

This repository contains the pipeline we use to calibrate the macromodel. The calibration is done using simulation-based inference, and the pipeline is structured as follows:

1. **Data generation** : We generate synthetic data using the macromodel. This data is used to calibrate the model.

    We sample parameters $` \left(\vec{\theta}_i\right)_{1\leq i \leq N} `$ from a prior distribution, where $N$ is the number of samples.

    For each parameter set $\vec{\theta}$, we simulate the model and generate synthetic data $` \left(\vec{y}_i\right)_{1\leq i \leq N} `$.

3. **Posterior training**: we then train a neural net to predict the posterior distribution of the parameters given the data. The neural net is trained on the synthetic data generated in the previous step. 

    It can be used to generate a posterior distribution for a new dataset, allowing one to sample from $` p(\vec{\theta}|\vec{y}) `$, that is the distribution of parameters given data.


## Sampling process
The sampling is handled by a `Sampler` class. This class is implemented in `sampler/sampler.py`. The sampler has methods that allow it to generate the samples in parallel (refer to the specific module for more details).

The sampler requires an `observer` function to be passed, which defines the observables of the model. This is a function $f`(\mathcal{M}) = \vec{y}`$ that takes a model $\mathcal{M}$ (that has already been run) as input and returns the observables $\vec{y}$. 

It also requires a function called `configuration_updater` that takes the configuration of a model $\mathcal{M}$ and a parameter set $\vec{\theta}$ and updates the configuration of the model with the parameters $\vec{\theta}$.

The sampler then handles the sampling process, generating the samples in parallel. The sampling process requires a prior sampler to be passed, which is a function that takes as an input the number of samples to be generated and returns an array of parameter sets sampled from a prior distribution.


## Example run
An example of how to run the pipeline is given in the `sampling.py` and `training.py` scripts. The number of samples is kept low for illustrative purposes, but in practice, one would use a much larger number of samples.

The `sampling.py` script generates synthetic data using the macromodel and saves it into `data.pkl`. The `training.py` script then trains a neural net to predict the posterior distribution of the parameters given the data. The posterior is saved into `posterior.pkl`. It can then be loaded and the user may sample from it using 

 ```python

  parameter_samples = posterior.sample((n_samples,), x=real_data)
```

where `real_data` **must** correspond to the observables of the model, i.e. it must be the real-world equivalent of the synthetic data generated in the first step.
