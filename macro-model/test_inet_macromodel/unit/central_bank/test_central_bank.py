from central_bank import CentralBank
from configurations import CentralBankConfiguration


class TestCentralBank:
    def test__init(self, datawrapper):
        synthetic_central_bank = datawrapper.synthetic_countries["FRA"].central_bank

        test_central_bank = CentralBank.from_pickled_agent(
            synthetic_central_bank=synthetic_central_bank,
            configuration=CentralBankConfiguration(),
            country_name="FRA",
            all_country_names=["FRA"],
            n_industries=18,
        )

        # check that set is empty
        assert not test_central_bank.states

        ts_keys = [
            "policy_rate",
        ]

        assert set(ts_keys).issubset(set(test_central_bank.ts.get_keys()))

        assert test_central_bank.compute_rate() == synthetic_central_bank.central_bank_data["Policy Rate"].iloc[0]
