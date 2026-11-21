import logging
import sys

import numpy as np

# =========================================
# ========== Auxiliary Functions ==========
# =========================================


def add_uniform_noise(data_sequence, p=1):
    """
    Function to add uniform random noise to a given dataset.
    Uniform noise in range [-5*10^(-p), 5*10^(-p)] is added to each
    data entry. For integer-valued sequences, p = 1.

    Args:
        data_sequence: numpy array of data to be processed.
        p: integer parameter controlling noise amplitude.

    Returns:
        numpy array with noise-added entries.
    """
    if p < 1:
        # print("Parameter p should be greater or equal to 1.")
        logging.error("Parameter p should be greater or equal to 1.")
        return None
    noise = np.random.uniform(-5.0 * 10 ** (-p), 5 * 10 ** (-p), size=len(data_sequence))
    randomized_data_sequence = data_sequence + noise
    # ensure there are no negative entries after noise is added
    randomized_data_sequence = randomized_data_sequence[np.where(randomized_data_sequence > 0)]
    return randomized_data_sequence


def get_distribution(data_sequence, number_of_bins=30):
    """
    Function to get a log-binned distribution of a given dataset.

    Args:
        data_sequence: numpy array with data to calculate
                       log-binned PDF on.
        number_of_bins: number of logarithmic bins to use.

    Returns:
        x, y: numpy arrays containing midpoints of bins
              and corresponding PDF values.

    """
    # define the support of the distribution
    lower_bound = min(data_sequence)
    upper_bound = max(data_sequence)
    # define bin edges
    log = np.log10
    lower_bound = log(lower_bound) if lower_bound > 0 else -1
    upper_bound = log(upper_bound)
    bins = np.logspace(lower_bound, upper_bound, number_of_bins)

    # compute the histogram using numpy
    y, __ = np.histogram(data_sequence, bins=bins, density=True)
    # for each bin, compute its midpoint
    x = bins[1:] - np.diff(bins) / 2.0
    # if bin is empty, drop it from the resulting list
    drop_indices = [i for i, k in enumerate(y) if k == 0.0]
    x = [k for i, k in enumerate(x) if i not in drop_indices]
    y = [k for i, k in enumerate(y) if i not in drop_indices]
    return x, y


def get_ccdf(degree_sequence):
    """
    Function to get CCDF of the list of degrees.

    Args:
        degree_sequence: numpy array of nodes' degrees.

    Returns:
        uniques: unique degree values met in the sequence.
        1-CDF: CCDF values corresponding to the unique values
               from the 'uniques' array.
    """
    uniques, counts = np.unique(degree_sequence, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / (degree_sequence.size)
    return uniques[::-1], (1.0 - cumprob)[::-1]


# ================================================
# ========== Hill Tail Index Estimation ==========
# ================================================
def get_moments_estimates_1(ordered_data):
    """
    Function to calculate first moments array given an ordered data
    sequence. Decreasing ordering is required.

    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st moment (Hill estimator)
                      is calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.

    """
    logs_1 = np.log(ordered_data)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1.0 / k_vector) * logs_1_cumsum - logs_1[1:]
    return M1


def get_moments_estimates_2(ordered_data):
    """
    Function to calculate first and second moments arrays
    given an ordered data sequence.
    Decreasing ordering is required.

    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator) and 2nd moments
                      are calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
        M2: numpy array of 2nd moments corresponding to all
            possible order statistics of the dataset.

    """
    logs_1 = np.log(ordered_data)
    logs_2 = (np.log(ordered_data)) ** 2
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1.0 / k_vector) * logs_1_cumsum - logs_1[1:]
    M2 = (1.0 / k_vector) * logs_2_cumsum - (2.0 * logs_1[1:] / k_vector) * logs_1_cumsum + logs_2[1:]
    return M1, M2


def get_moments_estimates_3(ordered_data):
    """
    Function to calculate first, second and third moments
    arrays given an ordered data sequence.
    Decreasing ordering is required.

    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator), 2nd and 3rd moments
                      are calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
        M2: numpy array of 2nd moments corresponding to all
            possible order statistics of the dataset.
        M3: numpy array of 3rd moments corresponding to all
            possible order statistics of the dataset.

    """
    logs_1 = np.log(ordered_data)
    logs_2 = (np.log(ordered_data)) ** 2
    logs_3 = (np.log(ordered_data)) ** 3
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    logs_3_cumsum = np.cumsum(logs_3[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1.0 / k_vector) * logs_1_cumsum - logs_1[1:]
    M2 = (1.0 / k_vector) * logs_2_cumsum - (2.0 * logs_1[1:] / k_vector) * logs_1_cumsum + logs_2[1:]
    M3 = (
        (1.0 / k_vector) * logs_3_cumsum
        - (3.0 * logs_1[1:] / k_vector) * logs_2_cumsum
        + (3.0 * logs_2[1:] / k_vector) * logs_1_cumsum
        - logs_3[1:]
    )
    # cleaning exceptional cases
    clean_indices = np.where(
        (M2 <= 0) | (M3 == 0) | (np.abs(1.0 - (M1**2) / M2) < 1e-10) | (np.abs(1.0 - (M1 * M2) / M3) < 1e-10)
    )
    M1[clean_indices] = np.nan
    M2[clean_indices] = np.nan
    M3[clean_indices] = np.nan
    return M1, M2, M3


def hill_dbs(
    ordered_data,
    t_bootstrap=0.5,
    r_bootstrap=500,
    eps_stop=1.0,
    verbose=False,
    diagn_plots=False,
):
    """
    Function to perform double-bootstrap procedure for
    Hill estimator.

    Args:
        ordered_data: numpy array for which double-bootstrap
                      is performed. Decreasing ordering is required.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity.
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        k_star:     number of order statistics optimal for estimation
                    according to the double-bootstrap procedure.
        x1_arr:     array of fractions of order statistics used for the
                    1st bootstrap sample.
        n1_amse:    array of AMSE values produced by the 1st bootstrap
                    sample.
        k1_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 1st bootstrap sample.
        max_index1: index of the 1st bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
        x2_arr:     array of fractions of order statistics used for the
                    2nd bootstrap sample.
        n2_amse:    array of AMSE values produced by the 2nd bootstrap
                    sample.
        k2_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 2nd bootstrap sample.
        max_index2: index of the 2nd bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.

    """
    if verbose:
        # print("Performing Hill double-bootstrap...")
        logging.info("Performing Hill double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5 * (1 + np.log(int(t_bootstrap * n)) / np.log(n))
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1 - 1)
    good_counts1 = np.zeros(n1 - 1)
    k1 = None
    k2 = None
    min_index1 = 1
    min_index2 = 1
    while k2 is None:
        # first bootstrap with n1 sample size
        for i in range(r_bootstrap):
            sample = np.random.choice(ordered_data, n1, replace=True)
            sample[::-1].sort()
            M1, M2 = get_moments_estimates_2(sample)
            current_amse1 = (M2 - 2.0 * (M1) ** 2) ** 2
            samples_n1 += current_amse1
            good_counts1[np.where(current_amse1 != np.nan)] += 1
        averaged_delta = samples_n1 / good_counts1

        max_index1 = (np.abs(np.linspace(1.0 / n1, 1.0, n1) - eps_stop)).argmin()
        k1 = np.nanargmin(averaged_delta[min_index1:max_index1]) + 1 + min_index1  # take care of indexing
        if diagn_plots:
            n1_amse = averaged_delta
            x1_arr = np.linspace(1.0 / n1, 1.0, n1)

        # second bootstrap with n2 sample size
        n2 = int(n1 * n1 / float(n))
        samples_n2 = np.zeros(n2 - 1)
        good_counts2 = np.zeros(n2 - 1)

        for i in range(r_bootstrap):
            sample = np.random.choice(ordered_data, n2, replace=True)
            sample[::-1].sort()
            M1, M2 = get_moments_estimates_2(sample)
            current_amse2 = (M2 - 2.0 * (M1**2)) ** 2
            samples_n2 += current_amse2
            good_counts2[np.where(current_amse2 != np.nan)] += 1
        max_index2 = (np.abs(np.linspace(1.0 / n2, 1.0, n2) - eps_stop)).argmin()
        averaged_delta = samples_n2 / good_counts2

        k2 = np.nanargmin(averaged_delta[min_index2:max_index2]) + 1 + min_index2  # take care of indexing
        if diagn_plots:
            n2_amse = averaged_delta
            x2_arr = np.linspace(1.0 / n2, 1.0, n2)

        if k2 > k1:
            # print("Warning (Hill): k2 > k1, AMSE false minimum suspected, resampling...")
            # move left AMSE boundary to avoid numerical issues
            min_index1 = min_index1 + int(0.005 * n)
            min_index2 = min_index2 + int(0.005 * n)
            k2 = None

    """
    # this constant is provided in the Danielsson's paper
    # use instead of rho below if needed
    rho = (np.log(k1)/(2.*np.log(n1) - np.log(k1)))\
          **(2.*(np.log(n1) - np.log(k1))/(np.log(n1)))
    """

    # this constant is provided in Qi's paper
    rho = (1.0 - (2 * (np.log(k1) - np.log(n1)) / (np.log(k1)))) ** (np.log(k1) / np.log(n1) - 1.0)

    k_star = (k1 * k1 / float(k2)) * rho
    k_star = int(np.round(k_star))

    # enforce k_star to pick 2nd value (rare cases of extreme cutoffs)
    if k_star == 0:
        k_star = 2
    if int(k_star) >= len(ordered_data):
        logging.warning("Estimated threshold k is larger than the size of data")
        # print("WARNING: estimated threshold k is larger than the size of data")
        k_star = len(ordered_data) - 1
    if verbose:
        logging.info("--- Hill double-bootstrap information ---")
        logging.info("Size of the 1st bootstrap sample n1: %s", n1)
        logging.info("Size of the 2nd bootstrap sample n2: %s", n2)
        logging.info("Estimated k1: %s", k1)
        logging.info("Estimated k2: %s", k2)
        logging.info("Estimated constant rho: %s", rho)
        logging.info("Estimated optimal k: %s", k_star)
        logging.info("-----------------------------------------")

    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return (
        k_star,
        x1_arr,
        n1_amse,
        k1 / float(n1),
        max_index1,
        x2_arr,
        n2_amse,
        k2 / float(n2),
        max_index2,
    )


def hill_estimator(
    ordered_data,
    bootstrap=True,
    t_bootstrap=0.5,
    r_bootstrap=500,
    verbose=False,
    diagn_plots=False,
    eps_stop=0.99,
):
    """
    Function to calculate Hill estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity.
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """
    k_arr = np.arange(1, len(ordered_data))
    xi_arr = get_moments_estimates_1(ordered_data)
    if bootstrap:
        results = hill_dbs(
            ordered_data,
            t_bootstrap=t_bootstrap,
            r_bootstrap=r_bootstrap,
            verbose=verbose,
            diagn_plots=diagn_plots,
            eps_stop=eps_stop,
        )
        (
            k_star,
            x1_arr,
            n1_amse,
            k1,
            max_index1,
            x2_arr,
            n2_amse,
            k2,
            max_index2,
        ) = results
        while k_star is None:
            logging.info("Resampling...")
            results = hill_dbs(
                ordered_data,
                t_bootstrap=t_bootstrap,
                r_bootstrap=r_bootstrap,
                verbose=verbose,
                diagn_plots=diagn_plots,
                eps_stop=eps_stop,
            )
            (
                k_star,
                x1_arr,
                n1_amse,
                k1,
                max_index1,
                x2_arr,
                n2_amse,
                k2,
                max_index2,
            ) = results
        xi_star = xi_arr[k_star - 1]
        # print("Adjusted Hill estimated gamma:", 1 + 1.0 / xi_star)
        # print("**********")
        return 1 + 1.0 / xi_star
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4 * [None]
        x2_arr, n2_amse, k2, max_index2 = 4 * [None]
    results = [
        k_arr,
        xi_arr,
        k_star,
        xi_star,
        x1_arr,
        n1_amse,
        k1,
        max_index1,
        x2_arr,
        n2_amse,
        k2,
        max_index2,
    ]
    return results


def smooth_hill_estimator(ordered_data, r_smooth=2):
    """
    Function to calculate smooth Hill estimator for a
    given ordered dataset.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        r_smooth:     integer parameter controlling the width
                      of smoothing window. Typically small
                      value such as 2 or 3.
    Returns:
        k_arr:  numpy array of order statistics based on the data provided.
        xi_arr: numpy array of tail index estimates corresponding to
                the order statistics array k_arr.
    """
    n = len(ordered_data)
    M1 = get_moments_estimates_1(ordered_data)
    xi_arr = np.zeros(int(np.floor(float(n) / r_smooth)))
    k_arr = np.arange(1, int(np.floor(float(n) / r_smooth)) + 1)
    xi_arr[0] = M1[0]
    bin_lengths = np.array([1.0] + [float((r_smooth - 1) * k) for k in k_arr[:-1]])
    cum_sum = 0.0
    for i in range(1, r_smooth * int(np.floor(float(n) / r_smooth)) - 1):
        k = i
        cum_sum += M1[k]
        if (k + 1) % (r_smooth) == 0:
            xi_arr[int(k + 1) // int(r_smooth)] = cum_sum
            cum_sum -= M1[int(k + 1) // int(r_smooth)]
    xi_arr = xi_arr / bin_lengths
    return k_arr, xi_arr


# ===================================================
# ========== Moments Tail Index Estimation ==========
# ===================================================


def moments_dbs_prefactor(xi_n, n1, k1):
    """
    Function to calculate pre-factor used in moments
    double-bootstrap procedure.

    Args:
        xi_n: moments tail index estimate corresponding to
              sqrt(n)-th order statistic.
        n1:   size of the 1st bootstrap in double-bootstrap
              procedure.
        k1:   estimated optimal order statistic based on the 1st
              bootstrap sample.

    Returns:
        prefactor: constant used in estimation of the optimal
                   stopping order statistic for moments estimator.
    """

    def V_sq(xi_n):
        if xi_n >= 0:
            V = 1.0 + (xi_n) ** 2
            return V
        else:
            a = (1.0 - xi_n) ** 2
            b = (1 - 2 * xi_n) * (6 * ((xi_n) ** 2) - xi_n + 1)
            c = (1.0 - 3 * xi_n) * (1 - 4 * xi_n)
            V = a * b / c
            return V

    def V_bar_sq(xi_n):
        if xi_n >= 0:
            V = 0.25 * (1 + (xi_n) ** 2)
            return V
        else:
            a = 0.25 * ((1 - xi_n) ** 2)
            b = 1 - 8 * xi_n + 48 * (xi_n**2) - 154 * (xi_n**3)
            c = 263 * (xi_n**4) - 222 * (xi_n**5) + 72 * (xi_n**6)
            d = (1.0 - 2 * xi_n) * (1 - 3 * xi_n) * (1 - 4 * xi_n)
            e = (1.0 - 5 * xi_n) * (1 - 6 * xi_n)
            V = a * (b + c) / (d * e)
            return V

    def b(xi_n, rho):
        if xi_n < rho:
            a1 = (1.0 - xi_n) * (1 - 2 * xi_n)
            a2 = (1.0 - rho - xi_n) * (1.0 - rho - 2 * xi_n)
            return a1 / a2
        elif xi_n >= rho and xi_n < 0:
            return 1.0 / (1 - xi_n)
        else:
            b = (xi_n / (rho * (1.0 - rho))) + (1.0 / ((1 - rho) ** 2))
            return b

    def b_bar(xi_n, rho):
        if xi_n < rho:
            a1 = 0.5 * (-rho * (1 - xi_n) ** 2)
            a2 = (1.0 - xi_n - rho) * (1 - 2 * xi_n - rho) * (1 - 3 * xi_n - rho)
            return a1 / a2
        elif xi_n >= rho and xi_n < 0:
            a1 = 1 - 2 * xi_n - np.sqrt((1 - xi_n) * (1 - 2.0 * xi_n))
            a2 = (1.0 - xi_n) * (1 - 2 * xi_n)
            return a1 / a2
        else:
            b = (-1.0) * ((rho + xi_n * (1 - rho)) / (2 * (1 - rho) ** 3))
            return b

    rho = np.log(k1) / (2 * np.log(k1) - 2.0 * np.log(n1))
    a = (V_sq(xi_n)) * (b_bar(xi_n, rho) ** 2)
    b = V_bar_sq(xi_n) * (b(xi_n, rho) ** 2)
    prefactor = (a / b) ** (1.0 / (1.0 - 2 * rho))
    return prefactor


def moments_dbs(
    ordered_data,
    xi_n,
    t_bootstrap=0.5,
    r_bootstrap=500,
    eps_stop=1.0,
    verbose=False,
    diagn_plots=False,
):
    """
    Function to perform double-bootstrap procedure for
    moments estimator.

    Args:
        ordered_data: numpy array for which double-bootstrap
                      is performed. Decreasing ordering is required.
        xi_n:         moments tail index estimate corresponding to
                      sqrt(n)-th order statistic.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity.
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.


    Returns:
        k_star:     number of order statistics optimal for estimation
                    according to the double-bootstrap procedure.
        x1_arr:     array of fractions of order statistics used for the
                    1st bootstrap sample.
        n1_amse:    array of AMSE values produced by the 1st bootstrap
                    sample.
        k1_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 1st bootstrap sample.
        max_index1: index of the 1st bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
        x2_arr:     array of fractions of order statistics used for the
                    2nd bootstrap sample.
        n2_amse:    array of AMSE values produced by the 2nd bootstrap
                    sample.
        k2_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 2nd bootstrap sample.
        max_index2: index of the 2nd bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
    """
    if verbose:
        logging.info("Performing moments double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5 * (1 + np.log(int(t_bootstrap * n)) / np.log(n))

    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1 - 1)
    good_counts1 = np.zeros(n1 - 1)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n1, replace=True)
        sample[::-1].sort()
        M1, M2, M3 = get_moments_estimates_3(sample)
        xi_2 = M1 + 1.0 - 0.5 * (1.0 - (M1 * M1) / M2) ** (-1.0)
        xi_3 = np.sqrt(0.5 * M2) + 1.0 - (2.0 / 3.0) * (1.0 / (1.0 - M1 * M2 / M3))
        samples_n1 += (xi_2 - xi_3) ** 2
        good_counts1[np.where((xi_2 - xi_3) ** 2 != np.nan)] += 1
    max_index1 = (np.abs(np.linspace(1.0 / n1, 1.0, n1) - eps_stop)).argmin()
    averaged_delta = samples_n1 / good_counts1
    k1 = np.nanargmin(averaged_delta[:max_index1]) + 1  # take care of indexing
    if diagn_plots:
        n1_amse = averaged_delta
        x1_arr = np.linspace(1.0 / n1, 1.0, n1)

    # r second bootstrap with n2 sample size
    n2 = int(n1 * n1 / float(n))
    samples_n2 = np.zeros(n2 - 1)
    good_counts2 = np.zeros(n2 - 1)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n2, replace=True)
        sample[::-1].sort()
        M1, M2, M3 = get_moments_estimates_3(sample)
        xi_2 = M1 + 1.0 - 0.5 * (1.0 - (M1 * M1) / M2) ** (-1.0)
        xi_3 = np.sqrt(0.5 * M2) + 1.0 - (2.0 / 3.0) * (1.0 / (1.0 - M1 * M2 / M3))
        samples_n2 += (xi_2 - xi_3) ** 2
        good_counts2[np.where((xi_2 - xi_3) ** 2 != np.nan)] += 1
    max_index2 = (np.abs(np.linspace(1.0 / n2, 1.0, n2) - eps_stop)).argmin()
    averaged_delta = samples_n2 / good_counts2
    k2 = np.nanargmin(averaged_delta[:max_index2]) + 1  # take care of indexing
    if diagn_plots:
        n2_amse = averaged_delta
        x2_arr = np.linspace(1.0 / n2, 1.0, n2)

    if k2 > k1:
        logging.warning("Estimated k2 is greater than k1! Re-doing bootstrap...")
        return 9 * [None]

    # calculate estimated optimal stopping k
    prefactor = moments_dbs_prefactor(xi_n, n1, k1)
    k_star = int((k1 * k1 / float(k2)) * prefactor)

    if int(k_star) >= len(ordered_data):
        logging.warning("Estimated threshold k is larger than the size of data")
        k_star = len(ordered_data) - 1
    if verbose:
        logging.info("--- Moments double-bootstrap information ---")
        logging.info("Size of the 1st bootstrap sample n1: %s", n1)
        logging.info("Size of the 2nd bootstrap sample n2: %s", n2)
        logging.info("Estimated k1: %s", k1)
        logging.info("Estimated k2: %s", k2)
        logging.info("Estimated constant: %s", prefactor)
        logging.info("Estimated optimal k: %s", k_star)
        logging.info("--------------------------------------------")

    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return (
        k_star,
        x1_arr,
        n1_amse,
        k1 / float(n1),
        max_index1,
        x2_arr,
        n2_amse,
        k2 / float(n2),
        max_index2,
    )


def moments_estimator(
    ordered_data,
    bootstrap=True,
    t_bootstrap=0.5,
    r_bootstrap=500,
    verbose=False,
    diagn_plots=False,
    eps_stop=0.99,
):
    """
    Function to calculate moments estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity.
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """
    n = len(ordered_data)
    M1, M2 = get_moments_estimates_2(ordered_data)
    xi_arr = M1 + 1.0 - 0.5 * (1.0 - (M1 * M1) / M2) ** (-1)
    k_arr = np.arange(1, len(ordered_data))
    if bootstrap:
        xi_n = xi_arr[int(np.floor(n**0.5)) - 1]
        results = moments_dbs(
            ordered_data,
            xi_n,
            t_bootstrap=t_bootstrap,
            r_bootstrap=r_bootstrap,
            verbose=verbose,
            diagn_plots=diagn_plots,
            eps_stop=eps_stop,
        )
        while results[0] is None:
            logging.info("Resampling...")
            results = moments_dbs(
                ordered_data,
                xi_n,
                t_bootstrap=t_bootstrap,
                r_bootstrap=r_bootstrap,
                verbose=verbose,
                diagn_plots=diagn_plots,
                eps_stop=eps_stop,
            )
        (
            k_star,
            x1_arr,
            n1_amse,
            k1,
            max_index1,
            x2_arr,
            n2_amse,
            k2,
            max_index2,
        ) = results
        xi_star = xi_arr[k_star - 1]
        if xi_star <= 0:
            logging.info("Moments estimated gamma: infinity (xi <= 0).")
        else:
            logging.info("Moments estimated gamma:", 1 + 1.0 / xi_star)
        logging.info("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4 * [None]
        x2_arr, n2_amse, k2, max_index2 = 4 * [None]
    results = [
        k_arr,
        xi_arr,
        k_star,
        xi_star,
        x1_arr,
        n1_amse,
        k1,
        max_index1,
        x2_arr,
        n2_amse,
        k2,
        max_index2,
    ]
    return results


# =======================================================
# ========== Kernel-type Tail Index Estimation ==========
# =======================================================


def get_biweight_kernel_estimates(ordered_data, hsteps, alpha):
    """
    Function to calculate biweight kernel-type estimates for tail index.
    Biweight kernel is defined as:
    phi(u) = (15/8) * (1 - u^2)^2

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.

    Returns:
        h_arr:  numpy array of fractions of order statistics included
                in kernel-type tail index estimation.
        xi_arr: numpy array with tail index estimated corresponding
                to different fractions of order statistics included
                listed in h_arr array.
    """
    n = len(ordered_data)
    logs = np.log(ordered_data)
    differences = logs[:-1] - logs[1:]
    i_arr = np.arange(1, n) / float(n)
    i3_arr = i_arr**3
    i5_arr = i_arr**5
    i_alpha_arr = i_arr**alpha
    i_alpha2_arr = i_arr ** (2.0 + alpha)
    i_alpha4_arr = i_arr ** (4.0 + alpha)
    t1 = np.cumsum(i_arr * differences)
    t2 = np.cumsum(i3_arr * differences)
    t3 = np.cumsum(i5_arr * differences)
    t4 = np.cumsum(i_alpha_arr * differences)
    t5 = np.cumsum(i_alpha2_arr * differences)
    t6 = np.cumsum(i_alpha4_arr * differences)
    h_arr = np.logspace(np.log10(1.0 / n), np.log10(1.0), hsteps)
    max_i_vector = (np.floor((n * h_arr)) - 2.0).astype(int)
    gamma_pos = (
        (15.0 / (8 * h_arr)) * t1[max_i_vector]
        - (15.0 / (4 * (h_arr**3))) * t2[max_i_vector]
        + (15.0 / (8 * (h_arr**5))) * t3[max_i_vector]
    )

    q1 = (
        (15.0 / (8 * h_arr)) * t4[max_i_vector]
        + (15.0 / (8 * (h_arr**5))) * t6[max_i_vector]
        - (15.0 / (4 * (h_arr**3))) * t5[max_i_vector]
    )

    q2 = (
        (15.0 * (1 + alpha) / (8 * h_arr)) * t4[max_i_vector]
        + (15.0 * (5 + alpha) / (8 * (h_arr**5))) * t6[max_i_vector]
        - (15.0 * (3 + alpha) / (4 * (h_arr**3))) * t5[max_i_vector]
    )

    xi_arr = gamma_pos - 1.0 + q2 / q1
    return h_arr, xi_arr


def get_triweight_kernel_estimates(ordered_data, hsteps, alpha):
    """
    Function to calculate triweight kernel-type estimates for tail index.
    Triweight kernel is defined as:
    phi(u) = (35/16) * (1 - u^2)^3

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.

    Returns:
        h_arr:  numpy array of fractions of order statistics included
                in kernel-type tail index estimation.
        xi_arr: numpy array with tail index estimated corresponding
                to different fractions of order statistics included
                listed in h_arr array.
    """
    n = len(ordered_data)
    logs = np.log(ordered_data)
    differences = logs[:-1] - logs[1:]
    i_arr = np.arange(1, n) / float(n)
    i3_arr = i_arr**3
    i5_arr = i_arr**5
    i7_arr = i_arr**7
    i_alpha_arr = i_arr**alpha
    i_alpha2_arr = i_arr ** (2.0 + alpha)
    i_alpha4_arr = i_arr ** (4.0 + alpha)
    i_alpha6_arr = i_arr ** (6.0 + alpha)
    t1 = np.cumsum(i_arr * differences)
    t2 = np.cumsum(i3_arr * differences)
    t3 = np.cumsum(i5_arr * differences)
    t4 = np.cumsum(i7_arr * differences)
    t5 = np.cumsum(i_alpha_arr * differences)
    t6 = np.cumsum(i_alpha2_arr * differences)
    t7 = np.cumsum(i_alpha4_arr * differences)
    t8 = np.cumsum(i_alpha6_arr * differences)
    h_arr = np.logspace(np.log10(1.0 / n), np.log10(1.0), hsteps)
    max_i_vector = (np.floor((n * h_arr)) - 2.0).astype(int)

    gamma_pos = (
        (35.0 / (16 * h_arr)) * t1[max_i_vector]
        - (105.0 / (16 * (h_arr**3))) * t2[max_i_vector]
        + (105.0 / (16 * (h_arr**5))) * t3[max_i_vector]
        - (35.0 / (16 * (h_arr**7))) * t4[max_i_vector]
    )

    q1 = (
        (35.0 / (16 * h_arr)) * t5[max_i_vector]
        + (105.0 / (16 * (h_arr**5))) * t7[max_i_vector]
        - (105.0 / (16 * (h_arr**3))) * t6[max_i_vector]
        - (35.0 / (16 * (h_arr**7))) * t8[max_i_vector]
    )

    q2 = (
        (35.0 * (1 + alpha) / (16 * h_arr)) * t5[max_i_vector]
        + (105.0 * (5 + alpha) / (16 * (h_arr**5))) * t7[max_i_vector]
        - (105.0 * (3 + alpha) / (16 * (h_arr**3))) * t6[max_i_vector]
        - (35.0 * (7 + alpha) / (16 * (h_arr**7))) * t8[max_i_vector]
    )

    xi_arr = gamma_pos - 1.0 + q2 / q1
    return h_arr, xi_arr


def kernel_type_dbs(
    ordered_data,
    hsteps,
    t_bootstrap=0.5,
    r_bootstrap=500,
    alpha=0.6,
    eps_stop=1.0,
    verbose=False,
    diagn_plots=False,
):
    """
    Function to perform double-bootstrap procedure for
    moments estimator.

    Args:
        ordered_data: numpy array for which double-bootstrap
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity.
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.


    Returns:
        h_star:       fraction of order statistics optimal for estimation
                      according to the double-bootstrap procedure.
        x1_arr:       array of fractions of order statistics used for the
                      1st bootstrap sample.
        n1_amse:      array of AMSE values produced by the 1st bootstrap
                      sample.
        h1:           value of fraction of order statistics corresponding
                      to the minimum of AMSE for the 1st bootstrap sample.
        max_k_index1: index of the 1st bootstrap sample's order statistics
                      array corresponding to the minimization boundary set
                      by eps_stop parameter.
        x2_arr:       array of fractions of order statistics used for the
                      2nd bootstrap sample.
        n2_amse:      array of AMSE values produced by the 2nd bootstrap
                      sample.
        h2:           value of fraction of order statistics corresponding
                      to the minimum of AMSE for the 2nd bootstrap sample.
        max_k_index2: index of the 2nd bootstrap sample's order statistics
                      array corresponding to the minimization boundary set
                      by eps_stop parameter.
    """
    if verbose:
        logging.info("Performing kernel double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5 * (1 + np.log(int(t_bootstrap * n)) / np.log(n))

    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(hsteps)
    good_counts1 = np.zeros(hsteps)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n1, replace=True)
        sample[::-1].sort()
        _, xi2_arr = get_biweight_kernel_estimates(sample, hsteps, alpha)
        _, xi3_arr = get_triweight_kernel_estimates(sample, hsteps, alpha)
        samples_n1 += (xi2_arr - xi3_arr) ** 2
        good_counts1[np.where((xi2_arr - xi3_arr) ** 2 != np.nan)] += 1
    max_index1 = (np.abs(np.logspace(np.log10(1.0 / n1), np.log10(1.0), hsteps) - eps_stop)).argmin()
    x1_arr = np.logspace(np.log10(1.0 / n1), np.log10(1.0), hsteps)
    averaged_delta = samples_n1 / good_counts1
    h1 = x1_arr[np.nanargmin(averaged_delta[:max_index1])]
    if diagn_plots:
        n1_amse = averaged_delta

    # second bootstrap with n2 sample size
    n2 = int(n1 * n1 / float(n))
    if n2 < hsteps:
        sys.exit(
            "Number of h points is larger than number "
            + "of order statistics! Please either increase "
            + "the size of 2nd bootstrap or decrease number "
            + "of h grid points."
        )
    samples_n2 = np.zeros(hsteps)
    good_counts2 = np.zeros(hsteps)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n2, replace=True)
        sample[::-1].sort()
        _, xi2_arr = get_biweight_kernel_estimates(sample, hsteps, alpha)
        _, xi3_arr = get_triweight_kernel_estimates(sample, hsteps, alpha)
        samples_n2 += (xi2_arr - xi3_arr) ** 2
        good_counts2[np.where((xi2_arr - xi3_arr) ** 2 != np.nan)] += 1
    max_index2 = (np.abs(np.logspace(np.log10(1.0 / n2), np.log10(1.0), hsteps) - eps_stop)).argmin()
    x2_arr = np.logspace(np.log10(1.0 / n2), np.log10(1.0), hsteps)
    averaged_delta = samples_n2 / good_counts2
    h2 = x2_arr[np.nanargmin(averaged_delta[:max_index2])]
    if diagn_plots:
        n2_amse = averaged_delta

    A = (143.0 * ((np.log(n1) + np.log(h1)) ** 2) / (3 * (np.log(n1) - 13.0 * np.log(h1)) ** 2)) ** (
        -np.log(h1) / np.log(n1)
    )

    h_star = (h1 * h1 / float(h2)) * A

    if h_star > 1:
        logging.warning("Estimated threshold is larger than the size of data!")
        logging.info("Optimal h is set to 1...")
        h_star = 1.0

    if verbose:
        logging.info("--- Kernel-type double-bootstrap information ---")
        logging.info("Size of the 1st bootstrap sample n1: %s", n1)
        logging.info("Size of the 2nd bootstrap sample n2: %s", n2)
        logging.info("Estimated h1: %s", h1)
        logging.info("Estimated h2: %s", h2)
        logging.info("Estimated constant A: %s", A)
        logging.info("Estimated optimal h: %s", h_star)
        logging.info("------------------------------------------------")

    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    if x1_arr is not None:
        max_k_index1 = x1_arr[max_index1]
    else:
        max_k_index1 = None
    if x2_arr is not None:
        max_k_index2 = x2_arr[max_index2]
    else:
        max_k_index2 = None
    return h_star, x1_arr, n1_amse, h1, max_k_index1, x2_arr, n2_amse, h2, max_k_index2


def kernel_type_estimator(
    ordered_data,
    hsteps,
    alpha=0.6,
    bootstrap=True,
    t_bootstrap=0.5,
    r_bootstrap=500,
    verbose=False,
    diagn_plots=False,
    eps_stop=0.99,
):
    """
    Function to calculate kernel-type estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity.
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of fractions of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """

    n = len(ordered_data)
    h_arr, xi_arr = get_biweight_kernel_estimates(ordered_data, hsteps, alpha=alpha)
    if bootstrap:
        results = kernel_type_dbs(
            ordered_data,
            hsteps,
            t_bootstrap=t_bootstrap,
            alpha=alpha,
            r_bootstrap=r_bootstrap,
            verbose=verbose,
            diagn_plots=diagn_plots,
            eps_stop=eps_stop,
        )
        (
            h_star,
            x1_arr,
            n1_amse,
            h1,
            max_index1,
            x2_arr,
            n2_amse,
            h2,
            max_index2,
        ) = results
        while h_star is None:
            print("Resampling...")
            results = kernel_type_dbs(
                ordered_data,
                hsteps,
                t_bootstrap=t_bootstrap,
                alpha=alpha,
                r_bootstrap=r_bootstrap,
                verbose=verbose,
                diagn_plots=diagn_plots,
                eps_stop=eps_stop,
            )
            (
                h_star,
                x1_arr,
                n1_amse,
                h1,
                max_index1,
                x2_arr,
                n2_amse,
                h2,
                max_index2,
            ) = results

        # get k index which corresponds to h_star
        k_star = np.argmin(np.abs(h_arr - h_star))
        xi_star = xi_arr[k_star]
        k_arr = []
        k_star = int(np.floor(h_arr[k_star] * n)) - 1
        k_arr = np.floor((h_arr * n))
        if xi_star <= 0:
            logging.info("Kernel-type estimated gamma: infinity (xi <= 0).")
        else:
            # print("Kernel-type estimated gamma:", 1 + 1.0 / xi_star)
            logging.info("Kernel-type estimated gamma: %s", 1 + 1.0 / xi_star)
        # print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, h1, max_index1 = 4 * [None]
        x2_arr, n2_amse, h2, max_index2 = 4 * [None]
        k_arr = np.floor(h_arr * n)
    results = [
        np.array(k_arr),
        xi_arr,
        k_star,
        xi_star,
        x1_arr,
        n1_amse,
        h1,
        max_index1,
        x2_arr,
        n2_amse,
        h2,
        max_index2,
    ]
    return results


# ====================================================
# ========== Pickands Tail Index Estimation ==========
# ====================================================


def pickands_estimator(ordered_data):
    """
    Function to calculate Pickands estimator for the tail index.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.

    Returns:
        k_arr:  array containing order statistics used for
                Pickands estimator calculation. Note that only estimates
                up to floor(n/4)-th order statistic can be calculated.
        xi_arr: array containing tail index estimates corresponding
                to k-order statistics provided in k_arr.
    """
    n = len(ordered_data)
    indices_k = np.arange(1, int(np.floor(n / 4.0)) + 1)
    indices_2k = 2 * indices_k
    indices_4k = 4 * indices_k
    Z_k = ordered_data[indices_k - 1]
    Z_2k = ordered_data[indices_2k - 1]
    Z_4k = ordered_data[indices_4k - 1]
    xi_arr = (1.0 / np.log(2)) * np.log((Z_k - Z_2k) / (Z_2k - Z_4k))
    k_arr = np.array([float(i) for i in range(1, int(np.floor(n / 4.0)) + 1)])
    return k_arr, xi_arr


# ==================================================
# ========== Plotting and Data Processing ==========
# ==================================================


def make_plots(
    ordered_data,
    output_file_path,
    number_of_bins,
    r_smooth,
    alpha,
    hsteps,
    bootstrap_flag,
    t_bootstrap,
    r_bootstrap,
    diagn_plots,
    eps_stop,
    theta1,
    theta2,
    verbose,
    noise_flag,
    p_noise,
    savedata,
) -> float:
    """
    Function to create plots and save tail index estimation data.

    Args:
        ordered_data:     numpy array for which tail index estimation
                          is performed. Decreasing ordering is required.
        output_file_path: file path to which plots should be saved.
        number_of_bins:   number of log-bins for degree distribution.
        r_smooth:         integer parameter controlling the width
                          of smoothing window. Typically small
                          value such as 2 or 3.
        alpha:            parameter controlling the amount of "smoothing"
                          for the kernel-type estimator. Should be greater
                          than 0.5.
        hsteps:           parameter controlling number of bandwidth steps
                          of the kernel-type estimator.
        bootstrap_flag:   flag to switch on/off double-bootstrap procedure.
        t_bootstrap:      parameter controlling the size of the 2nd
                          bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:      number of bootstrap resamplings for the 1st and 2nd
                          bootstraps.
        diagn_plots:      flag to switch on/off generation of AMSE diagnostic
                          plots.
        eps_stop:         parameter controlling range of AMSE minimization.
                          Defined as the fraction of order statistics to
                          consider during the AMSE minimization step.
        theta1:           Lower bound of plotting range, defined as
                          k_min = ceil(n^theta1).
                          Overwritten if plots behave badly within the range.
        theta2:           Upper bound of plotting range, defined as
                          k_max = floor(n^theta2).
                          Overwritten if plots behave badly within the range.
        verbose:          flag controlling bootstrap verbosity.
        noise_flag:       Switch on/off uniform noise in range
                          [-5*10^(-p), 5*10^(-p)] that is added to each
                          data point. Used for integer-valued sequences
                          with p = 1 (default = 1).
        p_noise:          integer parameter controlling noise amplitude.
        savedata:         Flag to save data files in the directory with plots.
    """

    # add noise if needed
    if noise_flag:
        discrete_ordered_data = ordered_data
        discrete_ordered_data[::-1].sort()
        ordered_data = add_uniform_noise(ordered_data, p=p_noise)
    ordered_data[::-1].sort()

    # perform smooth Hill estimation
    k_sh_arr, xi_sh_arr = smooth_hill_estimator(ordered_data, r_smooth=r_smooth)

    # perform adjusted Hill estimation
    if verbose:
        logging.info("Calculating adjusted Hill...")
    hill_results = hill_estimator(
        ordered_data,
        bootstrap=bootstrap_flag,
        t_bootstrap=t_bootstrap,
        r_bootstrap=r_bootstrap,
        diagn_plots=diagn_plots,
        eps_stop=eps_stop,
        verbose=verbose,
    )
    return hill_results


# ==========================
# ========== Main ==========
# ==========================


def calculate_tail_exponent(data: np.ndarray) -> float:
    number_of_bins = 30
    r_smooth = 3
    alpha = 0.6
    hsteps = 200
    noise_flag = True
    p_noise = 1
    bootstrap_flag = True
    t_bootstrap = 0.5
    r_bootstrap = 500
    amse_border = 1.0
    diagnostic_plots_flag = False
    verbose = False

    data = data.copy()

    # enforce minimization boundary to the order statistics larger than border value
    eps_stop = 1 - float(len(data[np.where(data <= amse_border)])) / len(data)
    return make_plots(
        data,
        "",
        number_of_bins,
        r_smooth,
        alpha,
        hsteps,
        bootstrap_flag,
        t_bootstrap,
        r_bootstrap,
        diagnostic_plots_flag,
        eps_stop,
        0.01,
        0.99,
        verbose,
        noise_flag,
        p_noise,
        0,
    )
