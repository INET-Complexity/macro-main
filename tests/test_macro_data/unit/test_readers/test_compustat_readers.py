import pandas as pd
import pytest

from macro_data.configuration.countries import Country
from macro_data.readers.population_data.compustat_banks_reader import CompustatBanksReader
from macro_data.readers.population_data.compustat_firms_reader import CompustatFirmsReader


def _write_firm_files(tmp_path):
    annual_path = tmp_path / "firms_annual.csv"
    quarterly_path = tmp_path / "firms_quarterly.csv"

    pd.DataFrame(
        [
            {
                "fyear": 2014,
                "datadate": "2014-12-31",
                "emp": 10.0,
                "conm": "FIRM A",
                "loc": "USA",
            }
        ]
    ).to_csv(annual_path, index=False)

    pd.DataFrame(
        [
            {
                "curcdq": "USD",
                "fqtr": 1,
                "fyearq": 2014,
                "datadate": "2014-03-31",
                "atq": 100.0,
                "ceqq": 50.0,
                "dlttq": 20.0,
                "dptbq": 15.0,
                "gpq": 30.0,
                "invtq": 5.0,
                "ltq": 60.0,
                "revtq": 90.0,
                "conm": "FIRM A",
                "gsector": 10.0,
                "loc": "USA",
            },
            {
                "curcdq": "USD",
                "fqtr": 2,
                "fyearq": 2014,
                "datadate": "2014-06-30",
                "atq": 200.0,
                "ceqq": 70.0,
                "dlttq": 40.0,
                "dptbq": 25.0,
                "gpq": 60.0,
                "invtq": 10.0,
                "ltq": 80.0,
                "revtq": 120.0,
                "conm": "FIRM A",
                "gsector": 10.0,
                "loc": "USA",
            },
        ]
    ).to_csv(quarterly_path, index=False)

    return annual_path, quarterly_path


def _write_bank_file(tmp_path):
    path = tmp_path / "banks.csv"
    pd.DataFrame(
        [
            {
                "curcdq": "USD",
                "fqtr": 1,
                "fyearq": 2014,
                "datadate": "2014-03-31",
                "conm": "BANK A",
                "atq": 100.0,
                "ciq": 9.0,
                "dlttq": 20.0,
                "dptcq": 15.0,
                "ltq": 60.0,
                "teqq": 50.0,
                "dltisy": 3.0,
                "dltry": 2.0,
                "loc": "USA",
            },
            {
                "curcdq": "USD",
                "fqtr": 2,
                "fyearq": 2014,
                "datadate": "2014-06-30",
                "conm": "BANK A",
                "atq": 200.0,
                "ciq": 12.0,
                "dlttq": 40.0,
                "dptcq": 25.0,
                "ltq": 80.0,
                "teqq": 70.0,
                "dltisy": 6.0,
                "dltry": 4.0,
                "loc": "USA",
            },
        ]
    ).to_csv(path, index=False)
    return path


def test__compustat_firms_uses_configured_quarter(tmp_path):
    annual_path, quarterly_path = _write_firm_files(tmp_path)

    reader = CompustatFirmsReader.from_raw_data(
        year=2014,
        quarter=2,
        raw_annual_path=annual_path,
        raw_quarterly_path=quarterly_path,
        countries=[Country("USA")],
    )

    firm = reader.data.iloc[0]
    assert firm["Assets"] == pytest.approx(200.0)
    assert firm["Revenue"] == pytest.approx(120.0)
    assert firm["Profits"] == pytest.approx(60.0)


def test__compustat_firms_converts_active_quarterly_flows_to_monthly(tmp_path):
    annual_path, quarterly_path = _write_firm_files(tmp_path)

    reader = CompustatFirmsReader.from_raw_data(
        year=2014,
        quarter=2,
        raw_annual_path=annual_path,
        raw_quarterly_path=quarterly_path,
        countries=[Country("USA")],
        time_unit=1,
    )

    firm = reader.data.iloc[0]
    assert firm["Assets"] == pytest.approx(200.0)
    assert firm["Revenue"] == pytest.approx(40.0)
    assert firm["Profits"] == pytest.approx(20.0)


def test__compustat_firms_converts_active_quarterly_flows_to_bimonthly(tmp_path):
    annual_path, quarterly_path = _write_firm_files(tmp_path)

    reader = CompustatFirmsReader.from_raw_data(
        year=2014,
        quarter=2,
        raw_annual_path=annual_path,
        raw_quarterly_path=quarterly_path,
        countries=[Country("USA")],
        time_unit=2,
    )

    firm = reader.data.iloc[0]
    assert firm["Revenue"] == pytest.approx(80.0)
    assert firm["Profits"] == pytest.approx(40.0)


def test__compustat_banks_uses_configured_quarter_without_converting_inactive_flows(tmp_path):
    bank_path = _write_bank_file(tmp_path)

    reader = CompustatBanksReader.from_raw_data(
        year=2014,
        quarter=2,
        raw_quarterly_path=bank_path,
        countries=[Country("USA")],
        proxy_with_us=False,
        time_unit=1,
    )

    bank = reader.data.iloc[0]
    assert "Income" not in reader.data.columns
    assert "Long-term Debt Issuance" not in reader.data.columns
    assert "Long-term Debt Reduction" not in reader.data.columns
    assert bank["Assets"] == pytest.approx(200.0)
    assert bank["Debt"] == pytest.approx(40.0)
