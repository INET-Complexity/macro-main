import pytest

from macro_data.readers.policy_data.obps_can_reader import OBPSCANReader

RATES_CSV = "Date,CAN,CAN_BC\n2014,15.0,12.0\n2019,40.0,30.0\n2030,170.0,150.0\n"
POLICY_CSV = "Industry,reduction_factor,tightening_rate\nC24a,0.80,0.02\nC19,0.75,0.02\n"
ELEC_CSV = "Industry,reduction_factor,tightening_rate\nD01b,0.70,0.01\n"


class TestOBPSCANReader:
    def test_missing_rates_file_returns_none(self, tmp_path):
        policy = tmp_path / "policy.csv"
        policy.write_text(POLICY_CSV)
        reader = OBPSCANReader.read_from_raw_data(tmp_path / "missing.csv", policy)
        assert reader.obps_data is None

    def test_missing_policy_file_returns_none(self, tmp_path):
        rates = tmp_path / "rates.csv"
        rates.write_text(RATES_CSV)
        reader = OBPSCANReader.read_from_raw_data(rates, tmp_path / "missing.csv")
        assert reader.obps_data is None

    def test_loads_rates_and_policy(self, tmp_path):
        rates = tmp_path / "rates.csv"
        rates.write_text(RATES_CSV)
        policy = tmp_path / "policy.csv"
        policy.write_text(POLICY_CSV)

        reader = OBPSCANReader.read_from_raw_data(rates, policy)

        assert reader.obps_data is not None
        assert list(reader.obps_data.df_rates.columns) == ["Date", "CAN", "CAN_BC"]
        assert reader.obps_data.df_rates.loc[0, "CAN"] == pytest.approx(15.0)
        assert list(reader.obps_data.df_policy["Industry"]) == ["C24a", "C19"]
        assert reader.obps_data.df_policy.loc[0, "reduction_factor"] == pytest.approx(0.80)

    def test_optional_elec_file_loaded_when_present(self, tmp_path):
        rates = tmp_path / "rates.csv"
        rates.write_text(RATES_CSV)
        policy = tmp_path / "policy.csv"
        policy.write_text(POLICY_CSV)
        elec = tmp_path / "elec.csv"
        elec.write_text(ELEC_CSV)

        reader = OBPSCANReader.read_from_raw_data(rates, policy, policy_elec_path=elec)

        assert reader.obps_data.df_policy_elec is not None
        assert list(reader.obps_data.df_policy_elec["Industry"]) == ["D01b"]

    def test_missing_optional_elec_file_gives_none(self, tmp_path):
        rates = tmp_path / "rates.csv"
        rates.write_text(RATES_CSV)
        policy = tmp_path / "policy.csv"
        policy.write_text(POLICY_CSV)

        reader = OBPSCANReader.read_from_raw_data(rates, policy, policy_elec_path=tmp_path / "missing.csv")

        assert reader.obps_data is not None
        assert reader.obps_data.df_policy_elec is None
