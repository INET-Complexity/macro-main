import numpy as np
import pandas as pd
import random


def ensure_minimum_workers_in_industries(individual_data: pd.DataFrame, n_industries: int) -> pd.DataFrame:
    """
    Ensures that each industry has at least one employed individual (Activity Status == 1).
    If any industry has zero workers:
      1) Attempt to reassign surplus employed individuals from other industries.
      2) If not enough employed, convert some unemployed individuals (Activity Status == 2).
      3) If not enough total individuals, raise ValueError.

    Args:
        individual_data (pd.DataFrame): DataFrame of individuals.
          Must contain columns "Activity Status" (1=employed, 2=unemployed)
          and "Employment Industry" (integer-coded from 0..n_industries-1).
        n_industries (int): Number of distinct industries.

    Returns:
        pd.DataFrame: Updated individual_data, with possible changes to
                      "Activity Status" and "Employment Industry".
    """

    # Helper to count how many workers are employed in each industry
    def count_employees(data: pd.DataFrame) -> np.ndarray:
        counts = np.zeros(n_industries, dtype=int)
        for ind in range(n_industries):
            counts[ind] = np.sum((data["Employment Industry"] == ind) & (data["Activity Status"] == 1))
        return counts

    employees_per_industry = count_employees(individual_data)
    zero_employee_industries = np.where(employees_per_industry == 0)[0]

    # If every industry has at least one employee, do nothing
    if len(zero_employee_industries) == 0:
        return individual_data

    # Otherwise, we need to fill those empty industries
    total_employed = employees_per_industry.sum()

    # Check if total employed >= number of industries
    if total_employed >= len(zero_employee_industries):
        # Reassign surplus employees from industries with many workers
        # until no industry is empty
        for industry_idx in zero_employee_industries:
            # find an industry with surplus = employees_per_industry[i] > 1
            surplus_inds = np.where(employees_per_industry > 1)[0]
            if len(surplus_inds) == 0:
                break  # no more surplus to take from

            # pick the industry with the greatest surplus
            donor_ind = surplus_inds[np.argmax(employees_per_industry[surplus_inds])]
            # find one person in that donor industry
            donor_candidates = individual_data[
                (individual_data["Employment Industry"] == donor_ind) & (individual_data["Activity Status"] == 1)
            ].index.tolist()

            if len(donor_candidates) == 0:
                # This shouldn't happen if employees_per_industry was correct, but just in case
                continue

            # pick any one of them (could do random or other logic)
            chosen_one = random.choice(donor_candidates)
            # reassign this person
            individual_data.at[chosen_one, "Employment Industry"] = industry_idx
            # employees remain employed
            # update counters
            employees_per_industry[donor_ind] -= 1
            employees_per_industry[industry_idx] += 1

        # re-check if we still have empty industries
        employees_per_industry = count_employees(individual_data)
        zero_employee_industries = np.where(employees_per_industry == 0)[0]

    # If some industries are still zero, try using unemployed
    if len(zero_employee_industries) > 0:
        total_unemployed = np.sum(individual_data["Activity Status"] == 2)
        if total_employed + total_unemployed >= len(zero_employee_industries):
            # We'll convert some unemployed
            # gather unemployed indices
            unemployed_indices = individual_data[(individual_data["Activity Status"] == 2)].index.tolist()

            # fill empty industries from unemployed pool
            for industry_idx in zero_employee_industries:
                if len(unemployed_indices) == 0:
                    break
                # pick any unemployed person
                chosen_one = random.choice(unemployed_indices)
                unemployed_indices.remove(chosen_one)
                # make them employed in the empty industry
                individual_data.at[chosen_one, "Activity Status"] = 1
                individual_data.at[chosen_one, "Employment Industry"] = industry_idx

            # recalc employees again
            employees_per_industry = count_employees(individual_data)
            zero_employee_industries = np.where(employees_per_industry == 0)[0]

    # Final check — if we still have empty industries, raise an error
    if len(zero_employee_industries) > 0:
        raise ValueError(
            "Not enough individuals to fill each industry with at least one worker. "
            "Consider increasing the scale of the simulation."
        )

    return individual_data
