import numpy as np

from model.timeseries import TimeSeries
from model.individuals.individual_properties import ActivityStatus


def create_labour_market_timeseries(
    initial_individual_activity: np.ndarray,
    initial_individual_employment_industry: np.ndarray,
    n_industries: int,
) -> TimeSeries:
    num_employed = np.zeros(n_industries)
    for g in range(n_industries):
        num_employed[g] = np.sum(
            np.logical_and(
                initial_individual_employment_industry == g,
                initial_individual_activity == ActivityStatus.EMPLOYED,
            )
        )
    return TimeSeries(
        num_employed_individuals_before_clearing=[np.sum(initial_individual_activity == ActivityStatus.EMPLOYED)],
        num_individuals_newly_joining=[np.nan],
        num_individuals_newly_randomly_fired=[np.nan],
        num_individuals_newly_randomly_quit=[np.nan],
        num_individuals_newly_fired=[np.nan],
        num_individuals_newly_leaving=[np.nan],
        num_employed_individuals_by_sector=num_employed,
        #




    )
