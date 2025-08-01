# %%
import pickle as pkl

import numpy as np
import torch

from macrocalib.training.train import build_posterior, train_model

# %%
with open("samples.pkl", "rb") as f:
    samples = pkl.load(f)


# %%

simulated_data = np.concatenate([np.stack(sample["simulations"]) for sample in samples])

# %%
thetas = np.concatenate([np.stack(sample["thetas"]) for sample in samples])
# %%

simulated_data = torch.from_numpy(simulated_data)
thetas = torch.from_numpy(thetas)

# %%

inference, density_estimator = train_model(simulated_data, thetas)
# %%

posterior = build_posterior(inference, density_estimator)
# %%
# get posterior samples for 1% GDP growth and 0% unemployment growth

samples = posterior.sample(sample_shape=(10,), x=torch.Tensor([0.01, 0]))
# %%
# save posterior for later use

with open("posterior.pkl", "wb") as f:
    pkl.dump(posterior, f)

# %%
