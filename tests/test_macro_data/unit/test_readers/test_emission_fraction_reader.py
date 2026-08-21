import pathlib

import numpy as np
import pytest

from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions, EmissionsFractionReader

DATA_PATH = pathlib.Path(__file__).parent.parent / "sample_raw_data" / "emission_factors"


class TestEmissionsFractionReader:
    def test__read_fraction_data_returns_reader(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        assert isinstance(reader, EmissionsFractionReader)

    def test__string_path_accepted(self):
        reader = EmissionsFractionReader.read_fraction_data(str(DATA_PATH))
        assert isinstance(reader, EmissionsFractionReader)

    def test__co2_shape(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        # fixture has 4 emitting rows (B05a, B05b, B05c, C19), 50 industry columns
        assert reader.emitting_fraction_co2.shape == (4, 50)

    def test__ch4_shape(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        # fixture has 1 gas-type row, 50 industry columns
        assert reader.emitting_fraction_ch4.shape == (1, 50)

    def test__consumption_shape(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        assert reader.emitting_fraction_consumption.shape == (1, 50)

    def test__investment_shape(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        assert reader.emitting_fraction_investment.shape == (1, 50)

    def test__co2_values_non_negative(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        assert (reader.emitting_fraction_co2.values >= 0).all()

    def test__consumption_values_between_zero_and_one(self):
        reader = EmissionsFractionReader.read_fraction_data(DATA_PATH)
        vals = reader.emitting_fraction_consumption.values
        assert (vals >= 0).all() and (vals <= 1).all()


class TestEmissionFractions:
    @pytest.fixture
    def reader(self):
        return EmissionsFractionReader.read_fraction_data(DATA_PATH)

    def test__from_reader_populates_all_fields(self, reader):
        fractions = EmissionFractions.from_reader(reader)
        assert fractions.co2 is not None
        assert fractions.ch4 is not None
        assert fractions.consumption is not None
        assert fractions.investment is not None

    def test__from_reader_produces_numpy_arrays(self, reader):
        fractions = EmissionFractions.from_reader(reader)
        assert isinstance(fractions.co2, np.ndarray)
        assert isinstance(fractions.ch4, np.ndarray)
        assert isinstance(fractions.consumption, np.ndarray)
        assert isinstance(fractions.investment, np.ndarray)

    def test__co2_shape_preserved(self, reader):
        fractions = EmissionFractions.from_reader(reader)
        assert fractions.co2.shape == (4, 50)
        assert fractions.consumption.shape == (1, 50)

    def test__default_instance_all_none(self):
        fractions = EmissionFractions()
        assert fractions.co2 is None
        assert fractions.ch4 is None
        assert fractions.consumption is None
        assert fractions.investment is None

    def test__partial_instance_missing_fields_are_none(self):
        fractions = EmissionFractions(co2=np.ones((4, 50)))
        assert fractions.co2 is not None
        assert fractions.ch4 is None
        assert fractions.consumption is None
        assert fractions.investment is None
