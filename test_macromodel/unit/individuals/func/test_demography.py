from macromodel.individuals.func.demography import NoAging


class TestIndividualDemography:
    def test__update(self):
        assert NoAging().update(prev_n_individuals=100) == 100

    def test__check_for_birth(self):
        NoAging().check_for_birth()

    def test__check_for_death(self):
        NoAging().check_for_death()

    def test__individuals_joining_the_workforce(self):
        NoAging().individuals_joining_the_workforce()

    def test__individuals_leaving_the_workforce(self):
        NoAging().individuals_leaving_the_workforce()
