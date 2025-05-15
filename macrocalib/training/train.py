from typing import TypeAlias, Union

import torch
from sbi.inference import (
    SNPE,
    DirectPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from torch import nn

# Define the type of posterior
Posterior: TypeAlias = Union[MCMCPosterior, RejectionPosterior, VIPosterior, DirectPosterior]


def train_model(simulation_data: torch.Tensor, theta: torch.Tensor) -> tuple[SNPE, nn.Module]:
    """
    Train the model using the SNPE algorithm.

    Parameters:
    -----------
    simulation_data: torch.Tensor
        The observed data. Number of rows should be equal to the number of observations.
    theta: torch.Tensor
        The parameters that generated the data. Number of rows should be equal to the number of observations.

    Returns:
    --------
    inference: SNPE
        The trained Sequential Neural Prior Estimation (SNPE) model.
    density_estimator: nn.Module
        The trained density estimator.
    """
    inference = SNPE()
    inference = inference.append_simulations(theta.to(dtype=torch.float32), simulation_data.to(dtype=torch.float32))
    density_estimator = inference.train()

    return inference, density_estimator


def build_posterior(inference: SNPE, density_estimator: nn.Module, **kwargs: dict) -> Posterior:
    """
    Build the posterior distribution.

    Parameters:
    -----------
    inference: SNPE
        The trained Sequential Neural Prior Estimation (SNPE) model.
    density_estimator: nn.Module
        The trained density estimator.
    kwargs: dict
        Additional keyword arguments to pass to the posterior builder.

    Returns:
    --------
    posterior: Posterior
        The posterior distribution.
    """
    posterior = inference.build_posterior(density_estimator, **kwargs)
    return posterior


def training_pipeline(simulation_data: torch.Tensor, theta: torch.Tensor, **kwargs: dict) -> Posterior:
    """
    Train the model and build the posterior distribution, in a single function.

    Parameters:
    -----------
    simulation_data: torch.Tensor
        The observed data. Number of rows should be equal to the number of observations.
    theta: torch.Tensor
        The parameters that generated the data. Number of rows should be equal to the number of observations.
    kwargs: dict
        Additional keyword arguments to pass to the posterior builder.

    Returns:
    --------
    posterior: Posterior
        The posterior distribution.
    """
    inference, density_estimator = train_model(simulation_data, theta)
    posterior = build_posterior(inference, density_estimator, **kwargs)
    return posterior
