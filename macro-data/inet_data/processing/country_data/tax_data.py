from dataclasses import dataclass

from inet_data.readers import DataReaders


@dataclass
class TaxData:
    value_added_tax: float
    export_tax: float
    employer_social_insurance_tax: float
    employee_social_insurance_tax: float
    profit_tax: float
    income_tax: float
    capital_formation_tax: float
    risk_premium: float

    @classmethod
    def from_readers(cls, readers: DataReaders, country: str, year: int):
        return cls(
            value_added_tax=readers.world_bank.get_tau_vat(country, year),
            export_tax=readers.world_bank.get_tau_exp(country, year),
            employer_social_insurance_tax=readers.oecd_econ.read_tau_sif(country, year),
            employee_social_insurance_tax=readers.oecd_econ.read_tau_siw(country, year),
            profit_tax=readers.oecd_econ.read_tau_firm(country, year),
            income_tax=readers.oecd_econ.read_tau_income(country, year),
            risk_premium=readers.eurostat.firm_risk_premium(country, year),
            capital_formation_tax=readers.eurostat.taxrate_on_capital_formation(country, year),
        )
