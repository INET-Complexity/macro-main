import numpy as np
import pytest

from macromodel.forecaster.forecaster import check_len, OLSForecaster, ManualAutoregForecaster


def test__check_len():
    data = np.ones(2)
    with pytest.raises(ValueError, match="Array is too small"):
        check_len(data)


class TestOLS:
    def test__forecast(self):
        forecaster = OLSForecaster()
        data = np.ones(10)
        assert forecaster.forecast(data=data) == pytest.approx(1)

    def test__forecast_increasing(self):
        forecaster = OLSForecaster()
        data = np.arange(10)
        assert forecaster.forecast(data=data) == pytest.approx(10)


class TestAutoreg:
    def test__forecast(self):
        forecaster = ManualAutoregForecaster()
        data = np.ones(10)
        assert forecaster.forecast(data=data) == pytest.approx(1)
