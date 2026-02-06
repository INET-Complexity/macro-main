from macrocalib.sampler import PriorSampler, Sampler


def test_sampler(sampler: Sampler, prior_sampler: PriorSampler):
    sampler.base_configuration.t_max = 5
    sampler.base_configuration.seed = None

    n_runs = 5

    data = sampler.parallel_run(n_runs, prior_sampler)

    # Each element in data is a dict with 'simulations' key, which is a list of length n_runs
    total_runs = sum(len(core_result["simulations"]) for core_result in data)
    assert total_runs == n_runs * sampler.n_cores
