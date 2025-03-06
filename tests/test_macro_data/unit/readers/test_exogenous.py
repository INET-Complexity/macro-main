from macro_data.configuration.countries import Country
from macro_data.readers.exogenous_data import ExogenousCountryData


class TestExogenous:

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
