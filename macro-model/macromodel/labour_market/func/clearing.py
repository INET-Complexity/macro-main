import numpy as np

from numba import njit, float64, types, int64, boolean, types
from numba.typed import List


from macromodel.firms.firms import Firms
from macromodel.households.households import Households
from macromodel.individuals.individuals import Individuals
from macromodel.individuals.individual_properties import ActivityStatus

from abc import abstractmethod, ABC

from typing import Callable, Tuple


class LabourMarketClearer(ABC):
    def __init__(
        self,
        hiring_speed: float,
        firing_speed: float,
        random_firing_probability: float,
        sorted_firing: bool,
        optimised_hiring: bool,
        allow_switching_industries: bool,
        consider_reservation_wages: bool,
        firing_cost_fraction: float,
        hiring_cost_fraction: float,
        individuals_quitting: bool,
        individuals_quitting_temperature: float,
        compare_with_normalised_inputs: float,
        round_target_employment: bool,
    ):
        self.hiring_speed = hiring_speed
        self.firing_speed = firing_speed
        self.random_firing_probability = random_firing_probability
        self.sorted_firing = sorted_firing
        self.optimised_hiring = optimised_hiring
        self.allow_switching_industries = allow_switching_industries
        self.consider_reservation_wages = consider_reservation_wages
        self.firing_cost_fraction = firing_cost_fraction
        self.hiring_cost_fraction = hiring_cost_fraction
        self.individuals_quitting = individuals_quitting
        self.individuals_quitting_temperature = individuals_quitting_temperature
        self.compare_with_normalised_inputs = compare_with_normalised_inputs
        self.round_target_employment = round_target_employment

    @abstractmethod
    def clear(
        self,
        firms: Firms,
        households: Households,
        individuals: Individuals,
    ) -> tuple[np.ndarray, int, int, int, int]:
        pass


class NoLabourMarketClearer(LabourMarketClearer):
    def clear(
        self,
        firms: Firms,
        households: Households,
        individuals: Individuals,
    ) -> tuple[np.ndarray, int, int, int, int]:
        return np.zeros(firms.ts.current("n_firms")), 0, 0, 0, 0


class DefaultLabourMarketClearer(LabourMarketClearer):
    def clear(
        self,
        firms: Firms,
        households: Households,
        individuals: Individuals,
    ) -> tuple[np.ndarray, int, int, int, int]:
        if self.compare_with_normalised_inputs:
            prev_labour_inputs = firms.ts.current("normalised_labour_inputs")
            desired_labour_inputs = firms.ts.current("desired_labour_inputs")
        else:
            prev_labour_inputs = firms.ts.current("labour_inputs")
            desired_labour_inputs = firms.ts.current("desired_labour_inputs")
        current_individuals_activity = individuals.states["Activity Status"]
        current_individuals_industry = individuals.states["Employment Industry"]
        prev_individuals_productivity = individuals.ts.current("labour_inputs")
        individuals_corresponding_firm = individuals.states["Corresponding Firm ID"]
        firm_employments = firms.states["Employments"]
        current_individual_wages = individuals.ts.current("employee_income")
        current_household_wealth = households.ts.current("wealth")
        individuals_corresponding_household = individuals.states["Corresponding Household ID"]
        firm_industries = firms.states["Industry"]
        offered_wage_function = firms.states["offered_wage_function"]
        individual_reservation_wages = individuals.ts.current("reservation_wages")

        # Individuals are fired at random
        firing_costs_random_firing, num_newly_randomly_fired = random_firing(
            number_of_firms=prev_labour_inputs.shape[0],
            current_individuals_activity=current_individuals_activity,
            individuals_corresponding_firm=individuals_corresponding_firm,
            firm_employments=firm_employments,
            current_individual_wages=current_individual_wages,
            random_firing_probability=self.random_firing_probability,
            firing_cost_fraction=self.firing_cost_fraction,
        )

        # Individuals quit at random
        if self.individuals_quitting:
            num_newly_randomly_quit = random_quitting(
                current_individuals_activity=current_individuals_activity,
                individuals_corresponding_firm=individuals_corresponding_firm,
                firm_employments=firm_employments,
                current_individual_wages=current_individual_wages,
                current_household_wealth=current_household_wealth,
                individuals_corresponding_household=individuals_corresponding_household,
                individuals_quitting_temperature=self.individuals_quitting_temperature,
            )
        else:
            num_newly_randomly_quit = 0

        # Firing
        firing_costs_regular, num_newly_fired = self.firing(
            firm_employments=firm_employments,
            current_individuals_activity=current_individuals_activity,
            individuals_corresponding_firm=individuals_corresponding_firm,
            prev_individuals_productivity=prev_individuals_productivity,
            desired_labour_inputs=desired_labour_inputs,
            prev_labour_inputs=prev_labour_inputs,
            current_individual_wages=current_individual_wages,
            firm_industries=firm_industries,
            average_industry_productivity=firms.states["Labour Productivity by Industry"],
        )

        # Hiring
        individuals.states["Offered Wage of Accepted Job"] = np.zeros(len(current_individuals_activity))
        hiring_costs_regular, num_newly_joining = self.hiring(
            firm_employments=firm_employments,
            firm_industries=firm_industries,
            current_individuals_activity=current_individuals_activity,
            current_individuals_industry=current_individuals_industry,
            individuals_corresponding_firm=individuals_corresponding_firm,
            prev_individuals_productivity=prev_individuals_productivity,
            desired_labour_inputs=desired_labour_inputs,
            prev_labour_inputs=prev_labour_inputs,
            offered_wage_function=offered_wage_function,
            offered_wage=individuals.states["Offered Wage of Accepted Job"],
            individual_reservation_wages=individual_reservation_wages,
            current_individual_wages=current_individual_wages,
            average_industry_productivity=firms.states["Labour Productivity by Industry"],
        )

        return (
            firing_costs_random_firing + firing_costs_regular + hiring_costs_regular,
            num_newly_joining,
            num_newly_randomly_fired,
            num_newly_randomly_quit,
            num_newly_fired,
        )

    def firing(
        self,
        firm_employments: list[np.ndarray],
        current_individuals_activity: np.ndarray,
        individuals_corresponding_firm: np.ndarray,
        prev_individuals_productivity: np.ndarray,
        desired_labour_inputs: np.ndarray,
        prev_labour_inputs: np.ndarray,
        current_individual_wages: np.ndarray,
        firm_industries: np.ndarray,
        average_industry_productivity: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        firing_costs = np.zeros_like(desired_labour_inputs)
        num_newly_fired = 0
        excess_productivity = prev_labour_inputs - desired_labour_inputs
        initial_excess_productivity = excess_productivity.copy()
        for firm_id in np.where(excess_productivity > 0)[0]:
            if len(firm_employments[firm_id]) == 1:
                continue

            # The order in which individuals are fired
            ind_firing_queue = self.get_firing_queue(firm_employments, firm_id, prev_individuals_productivity)

            # Firing them in that order
            labour_supply_lost = 0
            firing_costs[firm_id], curr_num_newly_fired = self.fire_in_order(
                current_individuals_activity,
                excess_productivity,
                firm_employments,
                firm_id,
                ind_firing_queue,
                individuals_corresponding_firm,
                initial_excess_productivity,
                labour_supply_lost,
                prev_individuals_productivity,
                current_individual_wages,
                firm_productivity=float(average_industry_productivity[firm_industries[firm_id]]),
            )

            # Count
            num_newly_fired += curr_num_newly_fired

        return firing_costs, num_newly_fired

    def fire_in_order(
        self,
        current_individuals_activity: np.ndarray,
        excess_productivity: np.ndarray,
        firm_employments: list[np.ndarray],
        firm_id: int,
        ind_firing_queue: np.ndarray,
        individuals_corresponding_firm: np.ndarray,
        initial_excess_productivity: np.ndarray,
        labour_supply_lost: float,
        prev_individuals_productivity: np.ndarray,
        current_individual_wages: np.ndarray,
        firm_productivity: float,
    ) -> tuple[float, int]:
        firing_costs = 0.0
        num_newly_fired = 0
        for i_to_fire in range(len(firm_employments[firm_id]) - 1):
            ind_to_fire = ind_firing_queue[i_to_fire]
            if self.round_target_employment:
                firing_reference = firm_productivity * prev_individuals_productivity[ind_to_fire]  # / 2.0
            else:
                firing_reference = 0.0

            if excess_productivity[firm_id] >= firing_reference:
                # Fire them
                fire_individual(
                    individual_id=int(ind_to_fire),
                    current_individuals_activity=current_individuals_activity,
                    individuals_corresponding_firm=individuals_corresponding_firm,
                    firm_employments=firm_employments,
                )

                # Update the remaining excess productivity
                excess_productivity[firm_id] -= firm_productivity * prev_individuals_productivity[ind_to_fire]

                # Calculate firing costs
                firing_costs += self.firing_cost_fraction * current_individual_wages[ind_to_fire]

                # Count
                num_newly_fired += 1

                # Frictions
                labour_supply_lost += firm_productivity * prev_individuals_productivity[ind_to_fire]
                if labour_supply_lost > self.firing_speed * initial_excess_productivity[firm_id]:
                    break
            else:
                break

        return firing_costs, num_newly_fired

    def get_firing_queue(
        self,
        firm_employments: list[np.ndarray],
        firm_id: int,
        prev_individuals_productivity: np.ndarray,
    ) -> np.ndarray:
        if self.sorted_firing:
            return sort_employees_by_productivity(
                current_firm_employments=firm_employments[firm_id],
                prev_individuals_productivity=prev_individuals_productivity,
            )
        else:
            return np.random.choice(
                firm_employments[firm_id],
                len(firm_employments[firm_id]),
                replace=False,
            )

    def hiring(
        self,
        firm_employments: list[list],
        firm_industries: np.ndarray,
        current_individuals_activity: np.ndarray,
        current_individuals_industry: np.ndarray,
        individuals_corresponding_firm: np.ndarray,
        prev_individuals_productivity: np.ndarray,
        desired_labour_inputs: np.ndarray,
        prev_labour_inputs: np.ndarray,
        offered_wage_function: Callable[[int, float | np.ndarray], float | np.ndarray],
        offered_wage: np.ndarray,
        individual_reservation_wages: np.ndarray,
        current_individual_wages: np.ndarray,  # noqa
        average_industry_productivity: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        if not self.allow_switching_industries:
            raise NotImplementedError("haven't done this yet")
        hiring_costs = np.zeros_like(desired_labour_inputs)
        num_newly_joining = 0
        missing_productivity = desired_labour_inputs - prev_labour_inputs
        initial_missing_productivity = missing_productivity.copy()

        # Collect potential employees
        unemployed_ind = np.array(current_individuals_activity == ActivityStatus.UNEMPLOYED)

        # Iterate over firms in random order
        firm_id_rnd = np.nonzero(missing_productivity > 0)[0]
        np.random.shuffle(firm_id_rnd)
        for firm_id in firm_id_rnd:
            labour_supply_gained = 0

            # Iterate until we're happy
            while True:
                # Find an appropriate employee
                ind_chosen = self.scout_for_employee(
                    unemployed_ind=unemployed_ind,
                    prev_individuals_productivity=prev_individuals_productivity,
                    current_individuals_industry=current_individuals_industry,
                    firm_industry=firm_industries[firm_id],
                    firm_missing_productivity=missing_productivity[firm_id],
                    firm_id=firm_id,
                    offered_wage_function=offered_wage_function,
                    individual_reservation_wages=individual_reservation_wages,
                    offered_wage=offered_wage,
                    average_industry_productivity=average_industry_productivity,
                )
                if ind_chosen is None:
                    break

                # Employ them
                hire_individual(
                    firm_employments=firm_employments,
                    current_individuals_activity=current_individuals_activity,
                    individuals_corresponding_firm=individuals_corresponding_firm,
                    current_individuals_industry=current_individuals_industry,
                    firm_id=firm_id,
                    firm_industry=firm_industries[firm_id],
                    ind_chosen=ind_chosen,  # noqa
                )

                # Update missing productivity
                missing_productivity[firm_id] -= (
                    average_industry_productivity[firm_industries[firm_id]] * prev_individuals_productivity[ind_chosen]
                )

                # Calculate hiring costs
                hiring_costs[firm_id] += self.hiring_cost_fraction * offered_wage[ind_chosen]

                # Count
                num_newly_joining += 1

                # Update
                unemployed_ind[ind_chosen] = False

                # Frictions
                labour_supply_gained += (
                    average_industry_productivity[firm_industries[firm_id]] * prev_individuals_productivity[ind_chosen]
                )
                if labour_supply_gained > self.hiring_speed * initial_missing_productivity[firm_id]:
                    break

        return hiring_costs, num_newly_joining

    def scout_for_employee(
        self,
        unemployed_ind: np.ndarray,
        prev_individuals_productivity: np.ndarray,
        current_individuals_industry: np.ndarray,  # noqa
        firm_industry: int,
        firm_missing_productivity: float,
        firm_id: int,
        offered_wage_function: Callable[[int, float | np.ndarray], float | np.ndarray],
        individual_reservation_wages: np.ndarray,
        offered_wage: np.ndarray,
        average_industry_productivity: np.ndarray,
    ) -> int | None:
        # If reservation wages are taken into account
        current_offered_wage = offered_wage_function(firm_id, prev_individuals_productivity)
        if self.consider_reservation_wages:
            would_accept_offer = current_offered_wage >= individual_reservation_wages
            unemployed_ind = np.logical_and(unemployed_ind, would_accept_offer)

        # Record offered wages
        offered_wage[unemployed_ind] = current_offered_wage[unemployed_ind]

        # If we're rounding
        if self.round_target_employment:
            unemployed_ind = np.logical_and(
                unemployed_ind,
                firm_missing_productivity
                > average_industry_productivity[firm_industry] * prev_individuals_productivity / 2.0,
            )

        # If individuals can not switch industries
        """
        if not self.allow_switching_industries:
            unemployed_ind = np.logical_and(
                unemployed_ind,
                current_individuals_industry == firm_industry,
            )
        """

        # Check if anyone would accept the offer
        if len(prev_individuals_productivity[unemployed_ind]) == 0:
            return None

        # Find the most suited individual
        if self.optimised_hiring:
            dist = np.abs(
                average_industry_productivity[firm_industry] * prev_individuals_productivity[unemployed_ind]
                - firm_missing_productivity
            )
            ind = unemployed_ind[np.argmin(dist)]
        else:
            ind = np.random.choice(np.where(unemployed_ind)[0])

        return ind


def sort_employees_by_productivity(
    current_firm_employments: np.ndarray,
    prev_individuals_productivity: np.ndarray,
) -> np.ndarray:
    return np.array(current_firm_employments)[np.argsort(prev_individuals_productivity[current_firm_employments])]


def random_firing(
    number_of_firms: int,
    current_individuals_activity: np.ndarray,
    individuals_corresponding_firm: np.ndarray,
    firm_employments: list,
    current_individual_wages: np.ndarray,
    random_firing_probability: float,
    firing_cost_fraction: float,
) -> tuple[np.ndarray, int]:
    firing_costs = np.zeros(number_of_firms)
    num_newly_randomly_fired = 0
    if random_firing_probability == 0.0:
        return firing_costs, num_newly_randomly_fired

    employed: np.ndarray = current_individuals_activity == ActivityStatus.EMPLOYED  # noqa

    is_fired = np.random.random(employed.sum()) <= random_firing_probability

    individual_indices = np.arange(current_individuals_activity.shape[0])

    for ind_id in individual_indices[employed][is_fired]:
        # Account for costs
        firing_costs[individuals_corresponding_firm[ind_id]] += firing_cost_fraction * current_individual_wages[ind_id]

        # Fire the individual
        fire_individual(
            individual_id=ind_id,
            current_individuals_activity=current_individuals_activity,
            individuals_corresponding_firm=individuals_corresponding_firm,
            firm_employments=firm_employments,
        )

        # Count
        num_newly_randomly_fired += 1

    return firing_costs, num_newly_randomly_fired


def random_quitting(
    current_individuals_activity: np.ndarray,
    individuals_corresponding_firm: np.ndarray,
    firm_employments: list,
    current_individual_wages: np.ndarray,
    current_household_wealth: np.ndarray,
    individuals_corresponding_household: np.ndarray,
    individuals_quitting_temperature: float,
) -> int:
    num_newly_randomly_quit = 0
    employed_individuals: np.ndarray = current_individuals_activity == ActivityStatus.EMPLOYED  # noqa
    individual_indices = np.arange(employed_individuals.shape[0])

    household_wealth = current_household_wealth[individuals_corresponding_household]

    exponentials = np.exp(-individuals_quitting_temperature * current_individual_wages / household_wealth)

    random_quit = np.random.random(employed_individuals.sum()) <= 1 - exponentials[employed_individuals]

    num_newly_randomly_quit = random_quit.sum()

    for ind_id in individual_indices[employed_individuals][random_quit]:
        # Fire the individual
        fire_individual(
            individual_id=ind_id,
            current_individuals_activity=current_individuals_activity,
            individuals_corresponding_firm=individuals_corresponding_firm,
            firm_employments=firm_employments,
        )

    return num_newly_randomly_quit


def fire_individual(
    individual_id: int,
    current_individuals_activity: np.ndarray,
    individuals_corresponding_firm: np.ndarray,
    firm_employments: list,
) -> None:
    current_individuals_activity[individual_id] = ActivityStatus.UNEMPLOYED
    corresponding_firm = individuals_corresponding_firm[individual_id]
    try:
        firm_employments[corresponding_firm].remove(individual_id)
    except ValueError:
        pass
    individuals_corresponding_firm[individual_id] = -1


def hire_individual(
    firm_employments: list[list],
    current_individuals_activity: np.ndarray,
    individuals_corresponding_firm: np.ndarray,
    current_individuals_industry: np.ndarray,
    firm_id: int,
    firm_industry: int,
    ind_chosen: int,
) -> None:
    assert current_individuals_activity[ind_chosen] == ActivityStatus.UNEMPLOYED
    current_individuals_activity[ind_chosen] = ActivityStatus.EMPLOYED
    individuals_corresponding_firm[ind_chosen] = firm_id
    current_individuals_industry[ind_chosen] = firm_industry
    firm_employments[firm_id].append(ind_chosen)


def check_employed_correspondence(activity_array: np.ndarray, firm_employments: list):
    all_employments = np.concatenate(firm_employments)
    all_employments = np.sort(all_employments)

    employed = activity_array == ActivityStatus.EMPLOYED
    ind_indices = np.arange(activity_array.shape[0])
    emp_indices = ind_indices[employed]

    size_matches = len(all_employments) == len(emp_indices)

    return size_matches and np.all(all_employments == emp_indices)


def check_employed_in_list(activity_array: np.ndarray, corresponding_firm: np.ndarray, firm_employments: list):
    employed = activity_array == ActivityStatus.EMPLOYED
    ind_indices = np.arange(activity_array.shape[0])
    emp_indices = ind_indices[employed]

    def try_index(employed_index: int) -> bool:
        firm_idx = corresponding_firm[employed_index]
        return employed_index in firm_employments[firm_idx]

    employees_match = np.all([try_index(i) for i in emp_indices])

    firms_match = True
    for i, employments in enumerate(firm_employments):
        # Check if all employed individuals are in the list
        for employee in employments:
            if corresponding_firm[employee] != i:
                firms_match = False
                break
            if employee not in emp_indices:
                firms_match = False
                break

    return employees_match and firms_match


class PolednaLabourMarketClearer(LabourMarketClearer):
    def clear(
        self,
        firms: Firms,
        households: Households,
        individuals: Individuals,
    ) -> tuple[np.ndarray, int, int, int, int]:
        if self.compare_with_normalised_inputs:
            prev_labour_inputs = firms.ts.current("normalised_labour_inputs")
            desired_labour_inputs = firms.ts.current("desired_labour_inputs")
        else:
            prev_labour_inputs = firms.ts.current("labour_inputs")
            desired_labour_inputs = firms.ts.current("desired_labour_inputs")
        current_individuals_activity = individuals.states["Activity Status"]
        current_individuals_industry = individuals.states["Employment Industry"]
        prev_individuals_productivity = individuals.ts.current("labour_inputs")
        individuals_corresponding_firm = individuals.states["Corresponding Firm ID"]
        firm_employments = firms.states["Employments"]
        current_individual_wages = individuals.ts.current("employee_income")
        current_household_wealth = households.ts.current("wealth")
        individuals_corresponding_household = individuals.states["Corresponding Household ID"]
        firm_industries = firms.states["Industry"]
        individual_reservation_wages = individuals.ts.current("reservation_wages")

        # Individuals are fired at random
        firing_costs_random_firing, num_newly_randomly_fired = random_firing(
            number_of_firms=prev_labour_inputs.shape[0],
            current_individuals_activity=current_individuals_activity,
            individuals_corresponding_firm=individuals_corresponding_firm,
            firm_employments=firm_employments,
            current_individual_wages=current_individual_wages,
            random_firing_probability=self.random_firing_probability,
            firing_cost_fraction=self.firing_cost_fraction,
        )

        # Individuals quit at random
        if self.individuals_quitting:
            num_newly_randomly_quit = random_quitting(
                current_individuals_activity=current_individuals_activity,
                individuals_corresponding_firm=individuals_corresponding_firm,
                firm_employments=firm_employments,
                current_individual_wages=current_individual_wages,
                current_household_wealth=current_household_wealth,
                individuals_corresponding_household=individuals_corresponding_household,
                individuals_quitting_temperature=self.individuals_quitting_temperature,
            )

        else:
            num_newly_randomly_quit = 0

        # Firing
        firing_costs_regular, num_newly_fired = firing(
            individuals_corresponding_firm=individuals_corresponding_firm,
            prev_individuals_productivity=prev_individuals_productivity,
            desired_labour_inputs=desired_labour_inputs,
            prev_labour_inputs=prev_labour_inputs,
            current_individual_wages=current_individual_wages,
            firm_industries=firm_industries,
            average_industry_productivity=firms.states["Labour Productivity by Industry"],
            firing_speed=self.firing_speed,
            firing_cost_fraction=self.firing_cost_fraction,
        )
        # Hiring
        individuals.states["Offered Wage of Accepted Job"] = np.zeros(len(current_individuals_activity))
        hiring_costs_regular, num_newly_joining, new_hires = hiring(
            firm_industries=firm_industries,
            current_individuals_industry=current_individuals_industry,
            individuals_corresponding_firm=individuals_corresponding_firm,
            prev_individuals_productivity=prev_individuals_productivity,
            current_ind_ea=np.logical_not(current_individuals_activity == ActivityStatus.NOT_ECONOMICALLY_ACTIVE),
            desired_labour_inputs=desired_labour_inputs,
            prev_labour_inputs=prev_labour_inputs,
            offered_wage=individuals.states["Offered Wage of Accepted Job"],
            individual_reservation_wages=individual_reservation_wages,
            current_individual_wages=current_individual_wages,
            average_industry_productivity=firms.states["Labour Productivity by Industry"],
            hiring_speed=self.hiring_speed,
            hiring_cost_fraction=self.hiring_cost_fraction,
        )

        for employment, hires in zip(firm_employments, new_hires):
            employment.extend(hires)
            current_individuals_activity[hires] = ActivityStatus.EMPLOYED

        # Sanity check
        assert np.all(
            np.bincount(
                individuals_corresponding_firm[individuals_corresponding_firm >= 0],
                minlength=firms.ts.current("n_firms"),
            )
            > 0
        )

        # Update individuals activity status
        current_individuals_activity[
            np.logical_and(
                current_individuals_activity != ActivityStatus.NOT_ECONOMICALLY_ACTIVE,
                np.isnan(individuals_corresponding_firm),
            )
        ] = ActivityStatus.UNEMPLOYED
        current_individuals_activity[individuals_corresponding_firm >= 0] = ActivityStatus.EMPLOYED

        return (
            firing_costs_random_firing + firing_costs_regular + hiring_costs_regular,
            num_newly_joining,
            num_newly_randomly_fired,
            num_newly_randomly_quit,
            num_newly_fired,
        )


@njit(
    types.Tuple((float64[:], int64))(
        int64[:],  # individuals_corresponding_firm
        float64[:],  # prev_individuals_productivity
        float64[:],  # desired_labour_inputs
        float64[:],  # prev_labour_inputs
        float64[:],  # current_individual_wages
        int64[:],  # firm_industries
        float64[:],  # average_industry_productivity
        float64,  # firing_speed
        float64,  # firing_cost_fraction
    ),
    cache=True,
)
def firing(
    individuals_corresponding_firm: np.ndarray,
    prev_individuals_productivity: np.ndarray,
    desired_labour_inputs: np.ndarray,
    prev_labour_inputs: np.ndarray,
    current_individual_wages: np.ndarray,
    firm_industries: np.ndarray,
    average_industry_productivity: np.ndarray,
    firing_speed: float,
    firing_cost_fraction: float,
) -> Tuple[np.ndarray, int]:
    firing_costs = np.zeros(desired_labour_inputs.shape)
    excess_employees = np.round(
        firing_speed
        * (
            prev_labour_inputs / average_industry_productivity[firm_industries]
            - np.maximum(
                1.0,
                desired_labour_inputs / average_industry_productivity[firm_industries],
            )
        )
    )
    for firm_id in np.where(excess_employees > 0)[0]:
        emp_ind = np.where(individuals_corresponding_firm == firm_id)[0]
        ind_firing = np.random.choice(
            emp_ind,
            int(min(emp_ind.shape[0] - 1, excess_employees[firm_id])),
            replace=False,
        )
        individuals_corresponding_firm[ind_firing] = -1
        firing_costs[firm_id] += firing_cost_fraction * current_individual_wages[ind_firing].sum()
    return firing_costs, int(excess_employees.sum())  # noqa


# @njit(
#     types.Tuple((float64[:], int64, List(List(int64))))(
#         int64[:],  # firm industries
#         float64[:],  # current individuals industry
#         int64[:],  # individuals corresponding firm
#         float64[:],  # prev individuals productivity
#         boolean[:],  # current individuals activity
#         float64[:],  # desired labour inputs
#         float64[:],  # prev labour inputs
#         float64[:],  # offered wage
#         float64[:],  # individual reservation wages
#         float64[:],  # current individual wages
#         float64[:],  # average industry productivity
#         float64,  # hiring speed
#         float64,  # hiring cost fraction
#     ),
#     cache=True,
# )
@njit(cache=True)
def hiring(
    firm_industries: np.ndarray,
    current_individuals_industry: np.ndarray,
    individuals_corresponding_firm: np.ndarray,
    prev_individuals_productivity: np.ndarray,
    current_ind_ea: np.ndarray,
    desired_labour_inputs: np.ndarray,
    prev_labour_inputs: np.ndarray,
    offered_wage: np.ndarray,
    individual_reservation_wages: np.ndarray,
    current_individual_wages: np.ndarray,  # noqa
    average_industry_productivity: np.ndarray,
    hiring_speed: float,
    hiring_cost_fraction: float,
) -> Tuple[np.ndarray, int, list]:
    hiring_costs, num_newly_joining = (
        np.zeros_like(desired_labour_inputs, np.float64),
        0,
    )
    extra_employees = np.floor(
        hiring_speed * (desired_labour_inputs - prev_labour_inputs) / average_industry_productivity[firm_industries]
    )

    new_hires = List()
    for _ in range(len(extra_employees)):
        new_hires.append(List.empty_list(int64))

    for firm_id in range(len(extra_employees)):
        if extra_employees[firm_id] > 0:
            ind_unemployed = np.where(np.logical_and(individuals_corresponding_firm == -1, current_ind_ea))[0]
            n_hiring = int(min(extra_employees[firm_id], len(ind_unemployed)))
            ind_hiring = np.random.choice(ind_unemployed, n_hiring, replace=False)
            individuals_corresponding_firm[ind_hiring] = firm_id
            for ind in ind_hiring:
                new_hires[firm_id].append(ind)
            hiring_costs[firm_id] += hiring_cost_fraction * offered_wage[ind_hiring].sum()
            num_newly_joining += n_hiring
    return hiring_costs, num_newly_joining, new_hires
