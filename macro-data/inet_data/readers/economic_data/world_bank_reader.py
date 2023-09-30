import numpy as np
import pandas as pd

from pathlib import Path


class WorldBankReader:
    def __init__(self, path: Path):
        self.files_with_codes = self.get_files_with_codes()

        self.data = {}
        for key in self.files_with_codes.keys():
            if key in [
                "long_term_interest_rates",
                "short_term_interest_rates",
                "government_debt_perc_gdp",
                "ppi",
                "cpi",
            ]:
                skiprows = []
            else:
                skiprows = [0, 1, 2, 3]
            self.data[key] = pd.read_csv(
                path / (self.files_with_codes[key] + ".csv"),
                skiprows=skiprows,
            )

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        return {
            "unemployment": "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_4325868",
            "participation": "API_SL.TLF.CACT.NE.ZS_DS2_en_csv_v2_4354787",
            "tau_vat": "API_GC.TAX.GSRV.VA.ZS_DS2_en_csv_v2_4028900",
            "tau_exp": "API_GC.TAX.EXPT.CN_DS2_en_csv_v2_4157140",
            "gini_coefs": "API_SI.POV.GINI_DS2_en_csv_v2_5358360",
            "fertility_rates": "API_SP.DYN.TFRT.IN_DS2_en_csv_v2_4151057",
            "interest_rates_on_govt_debt": "API_FR.INR.RINR_DS2_en_csv_v2_4150781",
            "long_term_interest_rates": "LONG_TERM_IR",
            "short_term_interest_rates": "SHORT_TERM_IR",
            "ppi": "ppi",
            "cpi": "cpi",
            "historic_gdp": "API_NY.GDP.MKTP.CN_DS2_en_csv_v2_5358562",
        }

    def get_unemployment_rate(self, country: str, year: int) -> float:
        df = self.data["unemployment"]
        df = df.loc[df["Country Code"] == country, str(year)]
        return df.values[0] / 100.0

    def get_participation_rate(self, country: str, year: int) -> float:
        df = self.data["participation"]
        df = df.loc[df["Country Code"] == country, str(year)]
        return df.values[0] / 100.0

    def get_tau_vat(self, country: str, year: int) -> float:
        df = self.data["tau_vat"]
        df = df.loc[df["Country Code"] == country][str(year)]
        return df.values[0] / 100.0

    def get_tau_exp(self, country: str, year: int) -> float:
        df = self.data["tau_exp"]
        df = df.loc[df["Country Code"] == country][str(year)]
        return df.values[0] / 100.0

    def get_gini_coef(self, country: str, year: int) -> float:
        df = self.data["gini_coefs"]
        return df.loc[df["Country Code"] == country][str(year)].values[0] / 100

    # current domestic
    def get_historic_gdp(self, country: str, year: int) -> float:
        df = self.data["historic_gdp"]
        df = df.loc[df["Country Code"] == country].iloc[:, 4:]
        return df.loc[:, str(year)].values[0]

    # current domestic
    def get_current_monthly_gdp(self, country: str, year: int) -> float:
        return self.get_historic_gdp(country, year) / 12.0

    def get_log_inflation(self, country: str) -> pd.DataFrame:
        # Get CPI and PPI inet_data
        data_cpi = self.data["cpi"].loc[self.data["cpi"]["Country Code"] == country]
        data_ppi = self.data["ppi"].loc[self.data["cpi"]["Country Code"] == country]
        dates, vals_cpi, vals_ppi = [], [], []
        for year in range(1970, 2024):
            for month in range(1, 13):
                s_month = str(month) if month > 9 else "0" + str(month)
                dates.append(str(year) + "-" + str(month))

                # CPI
                if str(year) + s_month in data_cpi.columns:
                    val_cpi = data_cpi.loc[:, str(year) + s_month].values
                    if len(val_cpi) == 0:
                        vals_cpi.append(np.nan)
                    else:
                        vals_cpi.append(val_cpi[0])
                else:
                    vals_cpi.append(np.nan)

                # PPI
                if str(year) + s_month in data_ppi.columns:
                    val_ppi = data_ppi.loc[:, str(year) + s_month].values
                    if len(val_ppi) == 0:
                        vals_ppi.append(np.nan)
                    else:
                        vals_ppi.append(val_ppi[0])
                else:
                    vals_ppi.append(np.nan)

        # Compute inflation
        data_df = pd.DataFrame(
            index=dates,
            data={
                "Real CPI Inflation": vals_cpi,
                "Real PPI Inflation": vals_ppi,
            },
        )
        data_df["Real CPI Inflation"] = np.log(data_df["Real CPI Inflation"] / data_df["Real CPI Inflation"].shift(1))
        data_df["Real PPI Inflation"] = np.log(data_df["Real PPI Inflation"] / data_df["Real PPI Inflation"].shift(1))

        return data_df
