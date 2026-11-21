# WorldBankReader

Expected files: 
`LONG_TERM_IR.csv` - Long Term Interest Rate

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.A+M+Q.IRLT.PA._Z._Z._Z._Z.N?dimensionAtObservation=AllDimensions&format=csvfile

`ppi.csv` - Producer Price Index

API: https://thedocs.worldbank.org/en/doc/1ad246272dbbc437c74323719506aa0c-0350012021/original/Inflation-data.xlsx

`SHORT_TERM_IR.csv` - Short Term Interest Rate

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.Q+M+A.IR3TIB.PA._Z._Z._Z._Z.N?dimensionAtObservation=AllDimensions&format=csvfile

`central_gov_debt.csv` - Central Government Debt

API: https://api.worldbank.org/v2

SERIES_CODE = "GC.DOD.TOTL.CN"

SOURCE_ID = "2"

`cpi.csv` - Consumer Price Index

API: https://www.worldbank.org/en/research/brief/inflation-database


`npl_ratios.csv` - Bank non-performing loans to gross loans (%)

API: (
f"https://api.worldbank.org/v2/sources/{SOURCE_ID}/country/all/"
f"series/{SERIES_CODE}/time/all/data"
)

SOURCE_ID = "32"

SERIES_CODE = "GFDD.SI.02"

________________________________________
API for the following files:  https://api.worldbank.org/v2

`API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_4325868.csv` - Unemployment rate (% of labor force; modeled ILO estimate)

`API_SL.TLF.CACT.NE.ZS_DS2_en_csv_v2_4354787.csv` - Labor force participation rate, ages 15+ (%; national estimate

`API_GC.TAX.GSRV.VA.ZS_DS2_en_csv_v2_4028900.csv` - Taxes on goods and services (% of industry and services value added)

`API_GC.TAX.EXPT.CN_DS2_en_csv_v2_4157140.csv` - Taxes on exports (current LCU)

`API_SI.POV.GINI_DS2_en_csv_v2_5358360.csv` - Gini Index

`API_SP.DYN.TFRT.IN_DS2_en_csv_v2_4151057.csv` - Total Fertility Rate

`API_FR.INR.RINR_DS2_en_csv_v2_4150781.csv` - Real Interest Rate

`API_NY.GDP.MKTP.CN_DS2_en_csv_v2_5358562.csv` - GDP (current local currency units)

`API_SP.POP.TOTL_DS2_en_csv_v2_79.csv` - Total population (persons; midyear estimate)


::: macro_data.readers.economic_data.world_bank_reader
    options:
        members:
            - WorldBankReader
            - get_files_with_codes
            - get_value
            - get_log_inflation
            - get_current_scaled_gdp
            - get_lcu_exports
            - prune
            - get_lcu_imports
            - get_gdp_growth_rate
            - get_population_growth_rate
            - get_unemployment_rate
            - get_inflation_rate
            - get_interest_rate
            - get_exchange_rate
            - get_trade_balance
            - get_current_account_balance
            - get_government_debt_to_gdp
            - get_government_deficit_to_gdp
            - get_household_savings_rate
            - get_business_investment_rate
            - get_export_market_share
            - get_import_market_share
            - get_foreign_direct_investment
            - get_foreign_portfolio_investment
