import numpy as np
from abc import abstractmethod, ABC
from typing import Callable

from inet_macromodel.firms.firms import Firms
from inet_macromodel.households.households import Households
from inet_macromodel.individuals.individual_properties import ActivityStatus
from inet_macromodel.individuals.individuals import Individuals


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
        hiring_threshold: float = 10.0,
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
        self.hiring_threshold = hiring_threshold

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
        prev_labour_productivity = firms.ts.current("labour_inputs")
        desired_labour_productivity = firms.ts.current("desired_labour_inputs")
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
            number_of_firms=prev_labour_productivity.shape[0],
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
            desired_labour_productivity=desired_labour_productivity,
            prev_labour_productivity=prev_labour_productivity,
            current_individual_wages=current_individual_wages,
        )

        # Hiring
        hiring_costs_regular, num_newly_joining = self.hiring(
            firm_employments=firm_employments,
            firm_industries=firm_industries,
            current_individuals_activity=current_individuals_activity,
            current_individuals_industry=current_individuals_industry,
            individuals_corresponding_firm=individuals_corresponding_firm,
            prev_individuals_productivity=prev_individuals_productivity,
            desired_labour_productivity=desired_labour_productivity,
            prev_labour_productivity=prev_labour_productivity,
            offered_wage_function=offered_wage_function,
            individual_reservation_wages=individual_reservation_wages,
            current_individual_wages=current_individual_wages,
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
        desired_labour_productivity: np.ndarray,
        prev_labour_productivity: np.ndarray,
        current_individual_wages: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        firing_costs = np.zeros_like(desired_labour_productivity)
        num_newly_fired = 0
        excess_productivity = prev_labour_productivity - desired_labour_productivity
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
    ) -> tuple[float, int]:
        firing_costs = 0.0
        num_newly_fired = 0
        for i_to_fire in range(len(firm_employments[firm_id]) - 1):
            ind_to_fire = ind_firing_queue[i_to_fire]
            if excess_productivity[firm_id] >= prev_individuals_productivity[ind_to_fire]:
                # Fire them
                fire_individual(
                    individual_id=ind_to_fire,
                    current_individuals_activity=current_individuals_activity,
                    individuals_corresponding_firm=individuals_corresponding_firm,
                    firm_employments=firm_employments,
                )

                # Update the remaining excess productivity
                excess_productivity[firm_id] -= prev_individuals_productivity[ind_to_fire]

                # Calculate firing costs
                firing_costs += self.firing_cost_fraction * current_individual_wages[ind_to_fire]

                # Count
                num_newly_fired += 1

                # Frictions
                labour_supply_lost += prev_individuals_productivity[ind_to_fire]
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
            ind_firing_queue = sort_employees_by_productivity(
                current_firm_employments=firm_employments[firm_id],
                prev_individuals_productivity=prev_individuals_productivity,
            )
        else:
            ind_firing_queue = np.random.choice(
                firm_employments[firm_id],
                len(firm_employments[firm_id]),
                replace=False,
            )
        return ind_firing_queue

    def hiring(
        self,
        firm_employments: list[list],
        firm_industries: np.ndarray,
        current_individuals_activity: np.ndarray,
        current_individuals_industry: np.ndarray,
        individuals_corresponding_firm: np.ndarray,
        prev_individuals_productivity: np.ndarray,
        desired_labour_productivity: np.ndarray,
        prev_labour_productivity: np.ndarray,
        offered_wage_function: Callable[[int, float | np.ndarray], float | np.ndarray],
        individual_reservation_wages: np.ndarray,
        current_individual_wages: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        hiring_costs = np.zeros_like(desired_labour_productivity)
        num_newly_joining = 0
        missing_productivity = desired_labour_productivity - prev_labour_productivity
        initial_missing_productivity = missing_productivity.copy()

        # Iterate over firms in random order
        firm_id_rnd = np.nonzero(missing_productivity > 0)[0]
        np.random.shuffle(firm_id_rnd)
        for firm_id in firm_id_rnd:
            labour_supply_gained = 0
            while self.should_continue_hiring(missing_productivity, firm_id, current_individuals_activity):
                # Find an appropriate employee
                ind_chosen = self.scout_for_employee(
                    current_individuals_activity=current_individuals_activity,
                    prev_individuals_productivity=prev_individuals_productivity,
                    current_individuals_industry=current_individuals_industry,
                    firm_industry=firm_industries[firm_id],
                    firm_missing_productivity=missing_productivity[firm_id],
                    firm_id=firm_id,
                    offered_wage_function=offered_wage_function,
                    individual_reservation_wages=individual_reservation_wages,
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
                missing_productivity[firm_id] -= prev_individuals_productivity[ind_chosen]

                # Calculate hiring costs
                hiring_costs[firm_id] += self.hiring_cost_fraction * current_individual_wages[ind_chosen]

                # Count
                num_newly_joining += 1

                # Frictions
                labour_supply_gained += prev_individuals_productivity[ind_chosen]
                if labour_supply_gained > self.hiring_speed * initial_missing_productivity[firm_id]:
                    break

        return hiring_costs, num_newly_joining

    def should_continue_hiring(
        self,
        missing_productivity: np.ndarray,
        firm_id: int,
        current_individuals_activity: np.ndarray,
    ) -> bool:
        return missing_productivity[firm_id] > self.hiring_threshold and np.any(
            current_individuals_activity == ActivityStatus.UNEMPLOYED
        )

    def scout_for_employee(
        self,
        current_individuals_activity: np.ndarray,
        prev_individuals_productivity: np.ndarray,
        current_individuals_industry: np.ndarray,
        firm_industry: int,
        firm_missing_productivity: float,
        firm_id: int,
        offered_wage_function: Callable[[int, float | np.ndarray], float | np.ndarray],
        individual_reservation_wages: np.ndarray,
    ) -> int | None:
        # Find all unemployed individuals
        unemployed_ind = current_individuals_activity == ActivityStatus.UNEMPLOYED

        # If reservation wages are taken into account
        if self.consider_reservation_wages:
            would_accept_offer = (
                offered_wage_function(firm_id, prev_individuals_productivity) >= individual_reservation_wages
            )
            unemployed_ind = np.logical_and(unemployed_ind, would_accept_offer)

        # If individuals can not switch industries
        if not self.allow_switching_industries:
            unemployed_ind = np.logical_and(
                unemployed_ind,
                current_individuals_industry == firm_industry,
            )

        # Check if anyone would accept the offer
        if len(prev_individuals_productivity[unemployed_ind]) == 0:
            return None

        # Find the most suited individual
        if self.optimised_hiring:
            dist = np.abs(prev_individuals_productivity[unemployed_ind] - firm_missing_productivity)
            return np.where(unemployed_ind)[0][np.argmin(dist)]
        else:
            return np.random.choice(np.where(unemployed_ind)[0])


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
    for ind_id in np.where(current_individuals_activity == ActivityStatus.EMPLOYED)[0]:
        if len(firm_employments[individuals_corresponding_firm[ind_id]]) == 1:
            continue
        if np.random.random() <= random_firing_probability:
            # Account for costs
            firing_costs[individuals_corresponding_firm[ind_id]] += (
                firing_cost_fraction * current_individual_wages[ind_id]
            )

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
    for ind_id in np.where(current_individuals_activity == ActivityStatus.EMPLOYED)[0]:
        if len(firm_employments[individuals_corresponding_firm[ind_id]]) == 1:
            continue
        quitting_probability = 1 - np.exp(
            -individuals_quitting_temperature
            * current_individual_wages[ind_id]
            / current_household_wealth[individuals_corresponding_household[ind_id]]
        )
        if np.random.random() <= quitting_probability:
            # Fire the individual
            fire_individual(
                individual_id=ind_id,
                current_individuals_activity=current_individuals_activity,
                individuals_corresponding_firm=individuals_corresponding_firm,
                firm_employments=firm_employments,
            )

            # Count
            num_newly_randomly_quit += 1

    return num_newly_randomly_quit


def fire_individual(
    individual_id: int,
    current_individuals_activity: np.ndarray,
    individuals_corresponding_firm: np.ndarray,
    firm_employments: list,
) -> None:
    current_individuals_activity[individual_id] = ActivityStatus.UNEMPLOYED
    corresponding_firm = individuals_corresponding_firm[individual_id]
    firm_employments[corresponding_firm].remove(individual_id)
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
    current_individuals_activity[ind_chosen] = ActivityStatus.EMPLOYED
    individuals_corresponding_firm[ind_chosen] = firm_id
    current_individuals_industry[ind_chosen] = firm_industry
    firm_employments[firm_id].append(ind_chosen)
