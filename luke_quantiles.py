# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%


def create_dummy_sample(t_max: int, drift: float = 0.5, volatility: float = 0.5):
    time = np.arange(t_max)
    return np.exp(drift * time + volatility * np.cumsum(np.random.normal(size=t_max)))


def create_sample_df(t_max: int, n_samples: int, drift: float = 0.5, volatility: float = 0.5):
    # samples are columns
    samples = np.array([create_dummy_sample(t_max, drift, volatility) for _ in range(n_samples)]).T
    return pd.DataFrame(samples, columns=[f"sample_{i}" for i in range(n_samples)])


# %%

samples = create_sample_df(t_max=100, n_samples=500, drift=0.02, volatility=0.5)
# %%

median = samples.median(axis=1)

p5 = samples.quantile(0.05, axis=1)
p95 = samples.quantile(0.95, axis=1)
# %%

fig, ax = plt.subplots()

ax.plot(median, label="median")
ax.fill_between(np.arange(len(median)), p5, p95, alpha=0.2, label="5% - 95% range")
ax.legend()
ax.set_yscale("log")

# %%


fig, ax = plt.subplots()

ax.plot(median, label="median")
ax.fill_between(np.arange(len(median)), p5, p95, alpha=0.2, label="5% - 95% range")
ax.legend()

# %%
