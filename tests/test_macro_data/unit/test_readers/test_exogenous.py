import numpy as np

from macro_data.configuration.countries import Country
from macro_data.readers.exogenous_data import ExogenousCountryData


class TestExogenous:
    def test__national_accounts_growth_uses_oecd_for_components_missing_from_imf(self, readers):
        country = Country("FRA")

        merged_growth = readers.get_national_accounts_growth(country)
        oecd_growth = readers.oecd_econ.get_na_growth_rates(country)
        imf_growth = readers.imf_reader.get_na_growth_rates(country)

        oecd_only_columns = {
            "Compensation of Employees",
            "Exports",
            "Gross Operating Surplus and Mixed Income",
            "Gross Value Added",
            "Gross Value Added - A",
            "Gross Value Added - B, C, D, E",
            "Gross Value Added - C",
            "Gross Value Added - F",
            "Gross Value Added - G, H, I",
            "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U",
            "Gross Value Added - J",
            "Gross Value Added - K",
            "Gross Value Added - L",
            "Gross Value Added - M, N",
            "Gross Value Added - O, P, Q",
            "Gross Value Added - R, S, T, U",
            "HH Cons",
            "Imports",
            "Taxes less Subsidies on Production",
        }

        assert oecd_only_columns.issubset(oecd_growth.columns)
        assert oecd_only_columns.isdisjoint(imf_growth.columns)

        for column in oecd_only_columns:
            assert merged_growth[column].equals(oecd_growth.loc[merged_growth.index, column])

    def test__nominal_and_real_hh_to_gdp_ratios_are_consistent(self, readers, industry_data):
        country = Country("FRA")
        data = ExogenousCountryData.from_data_readers(
            country_name=country,
            readers=readers,
            year=2014,
            quarter=1,
            industry_vectors=industry_data[country]["industry_vectors"],
        )

        nominal_ratio = data.national_accounts["Household Consumption (Value)"] / data.national_accounts["GDP (Value)"]
        real_ratio = (
            data.national_accounts["Real Household Consumption (Value)"] / data.national_accounts["Real GDP (Value)"]
        )

        assert np.allclose(nominal_ratio, real_ratio, equal_nan=True)

    def test__exogenous(self, readers, industry_data):
        country = Country("FRA")
        data = ExogenousCountryData.from_data_readers(
            country_name=country,
            readers=readers,
            year=2014,
            quarter=1,
            industry_vectors=industry_data[country]["industry_vectors"],
        )

        assert data.inflation.shape[0] > 0

        calibration_data = data.get_calibration_data(2014, 1)

        assert (calibration_data[("FRA", "HPI (Value)")].dropna() > 0).all()
