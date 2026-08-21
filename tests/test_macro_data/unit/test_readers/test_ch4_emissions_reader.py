import pathlib

import numpy as np
import pytest

from macro_data.readers.emissions.emissions_reader import CH4EmissionsDataCAN, CH4EmissionsReaderCAN

DATA_PATH = pathlib.Path(__file__).parent.parent / "sample_raw_data" / "emission_factors"
SAMPLE_CSV = DATA_PATH / "EN-GHG_EconSectByGas-CA_sample.csv"


class TestCH4EmissionsReaderCAN:
    def test__read_data_returns_reader(self):
        reader = CH4EmissionsReaderCAN.read_data(SAMPLE_CSV)
        assert isinstance(reader, CH4EmissionsReaderCAN)

    def test__string_path_accepted(self):
        reader = CH4EmissionsReaderCAN.read_data(str(SAMPLE_CSV))
        assert isinstance(reader, CH4EmissionsReaderCAN)

    def test__ignores_rows_without_id(self):
        reader = CH4EmissionsReaderCAN.read_data(SAMPLE_CSV)
        result = reader.get_ch4_by_industry_code(2014)
        assert "Oil and Gas" not in result
        assert None not in result

    def test__sums_multiple_rows_with_same_id(self):
        # B05c appears twice (42.0 + 8.0 = 50.0 ktCO2e → 50000 tCO2e)
        reader = CH4EmissionsReaderCAN.read_data(SAMPLE_CSV)
        result = reader.get_ch4_by_industry_code(2014)
        assert result["B05c"] == pytest.approx(50_000.0)

    def test__converts_ktco2e_to_tco2e(self):
        reader = CH4EmissionsReaderCAN.read_data(SAMPLE_CSV)
        result = reader.get_ch4_by_industry_code(2014)
        assert result["B05b"] == pytest.approx(35_000.0)

    def test__single_row_industry_correct(self):
        reader = CH4EmissionsReaderCAN.read_data(SAMPLE_CSV)
        result = reader.get_ch4_by_industry_code(2014)
        assert result["A01"] == pytest.approx(30_000.0)


class TestCH4EmissionsDataCAN:
    @pytest.fixture
    def reader(self):
        return CH4EmissionsReaderCAN.read_data(SAMPLE_CSV)

    @pytest.fixture
    def industries(self):
        return ["A01", "B05b", "B05c", "C19"]

    @pytest.fixture
    def production(self):
        # A01=1000, B05b=2000, B05c=5000, C19=0 (tests zero-production guard)
        return np.array([1000.0, 2000.0, 5000.0, 0.0])

    def test__from_reader_returns_data(self, reader, industries, production):
        data = CH4EmissionsDataCAN.from_reader(reader, industries, production)
        assert isinstance(data, CH4EmissionsDataCAN)

    def test__emitting_indices_subset_of_industries(self, reader, industries, production):
        data = CH4EmissionsDataCAN.from_reader(reader, industries, production)
        assert all(idx < len(industries) for idx in data.emitting_indices)

    def test__emission_factors_computed_from_production(self, reader, industries, production):
        # A01: 30000 tCO2e / 1000 LCU = 30.0
        data = CH4EmissionsDataCAN.from_reader(reader, industries, production)
        a01_pos = np.where(data.emitting_indices == industries.index("A01"))[0][0]
        assert data.emission_factors[a01_pos] == pytest.approx(30.0)

    def test__emission_factor_zero_when_no_csv_entry(self, reader, industries, production):
        # C19 is in the CH4 emitting list but not in the fixture CSV
        data = CH4EmissionsDataCAN.from_reader(reader, industries, production)
        c19_pos = np.where(data.emitting_indices == industries.index("C19"))[0][0]
        assert data.emission_factors[c19_pos] == pytest.approx(0.0)

    def test__emission_factor_zero_when_production_zero(self, reader, industries, production):
        # C19 has production=0 — should not produce NaN or division error
        data = CH4EmissionsDataCAN.from_reader(reader, industries, production)
        assert not np.any(np.isnan(data.emission_factors))

    def test__industries_not_in_list_excluded(self, reader, production):
        industries = ["A01", "B05b"]  # only 2 of the CH4 emitting industries
        data = CH4EmissionsDataCAN.from_reader(reader, industries, production[:2])
        assert len(data.emitting_indices) == 2
