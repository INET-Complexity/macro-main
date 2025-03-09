import pathlib

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCreditMarket:
    def test__create(
        self,
        readers,
    ):
        # TODO
        # this is just a dataclass, so we don't need to test it ftm
        # we need to test the create_loans functions though
        ...
