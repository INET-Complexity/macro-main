from abc import abstractmethod, ABC


class IndividualDemography(ABC):
    @abstractmethod
    def update(
        self,
        prev_n_individuals: float,
    ) -> float:
        pass

    @abstractmethod
    def check_for_death(
        self,
    ) -> None:
        pass

    @abstractmethod
    def check_for_birth(
        self,
    ) -> None:
        pass

    @abstractmethod
    def individuals_joining_the_workforce(
        self,
    ) -> None:
        pass

    @abstractmethod
    def individuals_leaving_the_workforce(
        self,
    ) -> None:
        pass


class NoAging(IndividualDemography):
    def update(
        self,
        prev_n_individuals: float,
    ) -> float:
        return prev_n_individuals

    def check_for_death(
        self,
    ) -> None:
        pass

    def check_for_birth(
        self,
    ) -> None:
        pass

    def individuals_joining_the_workforce(
        self,
    ) -> None:
        pass

    def individuals_leaving_the_workforce(
        self,
    ) -> None:
        pass
