# INET Oxford Macroeconomic Model Data Sources Documentation

This document provides an overview of all data sources used in the macro model and their associated variables.

## Economic Data Sources

### 1. OECD ICIO (Inter-Country Input-Output) Data

Aggregate time series data for each sector in each country. 

- **Source**: OECD ICIO Database
- **Script**: `io_tables/icio_reader.py`
- **Files**:
  - `icio/2010_SML.csv` - Main ICIO data for 2010
  - `icio/2011_SML.csv` - Main ICIO data for 2011
  - `icio/2012_SML.csv` - Main ICIO data for 2012
  - `icio/2013_SML.csv` - Main ICIO data for 2013
  - `icio/2014_SML.csv` - Main ICIO data for 2014
  - `icio/2015_SML.csv` - Main ICIO data for 2015
  - `icio/2016_SML.csv` - Main ICIO data for 2016
  - `icio/2017_SML.csv` - Main ICIO data for 2017
  - `icio/2018_SML.csv` - Main ICIO data for 2018
  - `icio/2019_SML.csv` - Main ICIO data for 2019
  - `icio/2020_SML.csv` - Main ICIO data for 2020
  - `icio/2010_SML_P.csv` - Pivoted ICIO data for 2010
  - `icio/2011_SML_P.csv` - Pivoted ICIO data for 2011
  - `icio/2012_SML_P.csv` - Pivoted ICIO data for 2012
  - `icio/2013_SML_P.csv` - Pivoted ICIO data for 2013
  - `icio/2014_SML_P.csv` - Pivoted ICIO data for 2014
  - `icio/2015_SML_P.csv` - Pivoted ICIO data for 2015
  - `icio/2016_SML_P.csv` - Pivoted ICIO data for 2016
  - `icio/2017_SML_P.csv` - Pivoted ICIO data for 2017
  - `icio/mappings.json` - Industry and country mappings
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Input-Output Tables | Track aggregate inter-industry relationships and economic flows between sectors | float | `{year}_SML.csv` | Industry classification (SIC codes) from ASHE survey |
| Industry-level data | Analyze aggregate sector-specific economic activities and resource utilization | float | `{year}_SML.csv` | Employment by industry (SIC codes) from ASHE survey |
| Trade flows | Monitor aggregate international trade patterns between countries | float | `{year}_SML.csv` | Not directly available in ONS surveys |
| Value added | Measure aggregate economic contribution at industry level | float | `{year}_SML.csv` | Gross annual income by industry from ASHE survey |
| Investment fractions | Calculate aggregate sector-specific investment allocations | float | `{year}_SML_P.csv` | Not directly available in ONS surveys |

### 2. WIOD-SEA (World Input-Output Database - Socio Economic Accounts)

Aggregate time series data for each sector in each country.

- **Source**: World Input-Output Database
- **Script**: `socioeconomic_data/wiod_sea_data.py`
- **Files**:
  - `wiod_sea/wiod_sea.csv` - Main WIOD-SEA data
  - `wiod_sea/mappings.json` - Industry and country mappings
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Socio-economic accounts | Track aggregate economic and social indicators across sectors | float | `wiod_sea.csv` | Economic activity and employment status from ASHE survey |
| Industry-level data | Analyze aggregate sector-specific economic activities | float | `wiod_sea.csv` | Industry classification and employment from ASHE survey |
| Value added reconciliation | Ensure consistency in aggregate value added calculations | float | `wiod_sea.csv` | Gross annual income by sector from ASHE survey |
| Investment matrix data | Monitor aggregate investment patterns across sectors | float | `wiod_sea.csv` | Not directly available in ONS surveys |

### 3. Eurostat

Aggregate time-series data for each country.

- **Script**: `economic_data/eurostat_reader.py`
- **Files**:
  - `eurostat/nasa_10_f_bs.csv` - Financial balance sheets
  - `eurostat/nasa_10_nf_tr.csv` - Non-financial transactions
  - `eurostat/eurostat_cpi.csv` - Consumer price indices
  - `eurostat/tec00132_linear.csv` - Employment data
  - `eurostat/perc_growth_services.csv` - Services growth rates
  - `eurostat/perc_growth_sector_F.csv` - Sector F growth rates
  - `eurostat/sector_l_iot.csv` - Sector labor data
  - `eurostat/perc_growth_sector_D.csv` - Sector D growth rates
  - `eurostat/perc_growth_sector_C.csv` - Sector C growth rates
  - `eurostat/perc_growth_sector_B.csv` - Sector B growth rates
  - `eurostat/eurostat_gdp.csv` - GDP data
  - `eurostat/namq_10_gdp.csv` - Quarterly GDP data
  - `eurostat/nama_10_nfa_st.csv` - Non-financial assets
  - `eurostat/naio_10_cp1700.csv` - Input-Output tables
  - `eurostat/lfst_hhnhtych.csv` - Labor force statistics
  - `eurostat/nama_10_nfa_fl.csv` - Non-financial assets by industry
  - `eurostat/irt_st_a.csv` - Interest rates
  - `eurostat/eurostat_hh_property_income.csv` - Household property income
  - `eurostat/eurostat_longterm_govbond_rates.csv` - Government bond rates
  - `eurostat/eurostat_hh_surplus_income.csv` - Household surplus income
  - `eurostat/eurostat_general_govdebt_ratios.csv` - Government debt ratios
  - `eurostat/eurostat_firmdeposit_ratios.csv` - Firm deposit ratios
  - `eurostat/eurostat_firm_surplus.csv` - Firm surplus data
  - `eurostat/eurostat_firmdebt_ratios.csv` - Firm debt ratios
  - `eurostat/eurostat_cbdebt_ratios.csv` - Central bank debt ratios
  - `eurostat/eurostat_cbequity_ratios.csv` - Central bank equity ratios
  - `eurostat/eurostat_central_govdebt_ratios.csv` - Central government debt ratios
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Financial balance sheets and ratios | Track aggregate financial positions across sectors | float | `nasa_10_f_bs.csv` | Total value of financial assets from WAS survey |
| Non-financial transactions | Monitor aggregate economic activities between sectors | float | `nasa_10_nf_tr.csv` | Not directly available in ONS surveys |
| Consumer price indices | Measure aggregate inflation and price changes | float | `eurostat_cpi.csv` | Not directly available in ONS surveys |
| Employment data | Track aggregate labor market conditions | float | `tec00132_linear.csv` | Employment status from ASHE survey |
| Sectoral growth rates | Monitor aggregate economic growth by sector | float | `perc_growth_*.csv` | Not directly available in ONS surveys |
| GDP data | Measure aggregate economic performance | float | `eurostat_gdp.csv`, `namq_10_gdp.csv` | Not directly available in ONS surveys |
| Non-financial assets | Track aggregate real asset values | float | `nama_10_nfa_st.csv`, `nama_10_nfa_fl.csv` | Property values and physical assets from WAS survey |
| Input-Output tables | Analyze aggregate economic relationships | float | `naio_10_cp1700.csv` | Industry classification from ASHE survey |
| Labor force statistics | Monitor aggregate employment trends | float | `lfst_hhnhtych.csv` | Employment status and economic activity from ASHE survey |
| Interest rates | Track aggregate monetary policy indicators | float | `irt_st_a.csv` | Not directly available in ONS surveys |
| Household data | Analyze aggregate consumer behavior | float | `eurostat_hh_*.csv` | Household income, wealth, and consumption from WAS survey |
| Government debt ratios | Monitor aggregate fiscal health | float | `eurostat_*_govdebt_ratios.csv` | Not directly available in ONS surveys |
| Firm data | Track aggregate business performance | float | `eurostat_firm*.csv` | Business ownership and income from WAS survey |
| Central bank data | Monitor aggregate monetary policy indicators | float | `eurostat_cb*.csv` | Not directly available in ONS surveys |

### 4. OECD Economic Data

Aggregate time-series data for each country.

- **Script**: `economic_data/oecd_economic_data.py`
- **Files**:
  - `oecd_econ/QNA.csv` - Quarterly National Accounts
  - `oecd_econ/SSIS_BSC_ISIC4.csv` - Business Statistics
  - `oecd_econ/EGDNA_PUBLIC.csv` - Economic Growth and Development
  - `oecd_econ/HISTPOP.csv` - Historical Population
  - `oecd_econ/SDBS_BDI_ISIC4.csv` - Structural Business Statistics
  - `oecd_econ/DP_LIVE_PPI_INFL.csv` - Producer Price Inflation
  - `oecd_econ/unemployment_rates.csv` - Unemployment Statistics
  - `oecd_econ/KEI.csv` - Key Economic Indicators
  - `oecd_econ/HOUSE_PRICES.csv` - House Price Indices
  - `oecd_econ/ALFS_EMP.csv` - Employment Statistics
  - `oecd_econ/oecd_govt_debt_usd_ppp.csv` - Government Debt
  - `oecd_econ/DP_LIVE_21042023152700149.csv` - Economic Indicators
  - `oecd_econ/DP_LIVE_21042023152525378.csv` - Economic Indicators
  - `oecd_econ/oecd_bank_data.csv` - Banking Statistics
  - `oecd_econ/LAB_REG_VAC.csv` - Labor Market Statistics
  - `oecd_econ/STLABOUR.csv` - Short-term Labor Statistics
  - `oecd_econ/MEI_FIN.csv` - Financial Indicators
  - `oecd_econ/CTS_CIT.csv` - Corporate Tax Statistics
  - `oecd_econ/TABLE_III2.csv` - Tax Statistics
  - `oecd_econ/TABLE_I6.csv` - Economic Statistics
  - `oecd_econ/TABLE_I7.csv` - Economic Statistics
  - `oecd_econ/TABLE_III1.csv` - Tax Statistics
  - `oecd_econ/SOCX_AGG.csv` - Social Expenditure
  - `oecd_econ/BPF1.csv` - Business Statistics
  - `oecd_econ/consumption_by_income_quintiles.csv` - Consumption Data
  - `oecd_econ/mappings.json` - Industry Mappings
  - `oecd_econ/oecd_bank_capital_reqs.csv` - Bank Capital Requirements
  - `oecd_econ/SDBS_BDI_ISIC4_BIRTH.csv` - Business Demography
  - `oecd_econ/SNA_TABLE750_24042023172303744.csv` - National Accounts
  - `oecd_econ/SDBS_BDI_ISIC4_DEATH.csv` - Business Demography
  - `oecd_econ/DP_LIVE_UNEMP.csv` - Unemployment Data
  - `oecd_econ/GINI_COEF.csv` - Income Inequality
  - `oecd_econ/DP_LIVE_HH_SAV_RATE.csv` - Household Savings
  - `oecd_econ/DP_LIVE_HH_DISP_INC.csv` - Household Income
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Employment by industry | Track aggregate sector-specific employment | float | `ALFS_EMP.csv` | Employment by industry (SIC codes) from ASHE survey |
| Business demographics | Monitor aggregate firm characteristics | float | `SDBS_BDI_ISIC4*.csv` | Business ownership and self-employment from WAS survey |
| Tax rates | Track aggregate fiscal policy indicators | float | `CTS_CIT.csv`, `TABLE_III*.csv` | Not directly available in ONS surveys |
| Banking sector statistics | Monitor aggregate financial sector health | float | `oecd_bank_data.csv`, `oecd_bank_capital_reqs.csv` | Not directly available in ONS surveys |
| Government debt and benefits | Track aggregate fiscal position | float | `oecd_govt_debt_usd_ppp.csv`, `SOCX_AGG.csv` | Not directly available in ONS surveys |
| Interest rates | Monitor aggregate monetary conditions | float | `MEI_FIN.csv` | Not directly available in ONS surveys |
| Housing market indicators | Track aggregate real estate market | float | `HOUSE_PRICES.csv` | Property values and mortgage data from WAS survey |
| Unemployment and vacancy rates | Monitor aggregate labor market | float | `unemployment_rates.csv`, `LAB_REG_VAC.csv` | Employment status from ASHE survey |
| Consumption statistics | Analyze aggregate consumer behavior | float | `consumption_by_income_quintiles.csv` | Household income and spending patterns from WAS survey |
| National accounts growth rates | Track aggregate economic performance | float | `QNA.csv`, `SNA_TABLE750_*.csv` | Not directly available in ONS surveys |
| Bank demographics | Monitor aggregate financial sector structure | float | `oecd_bank_data.csv` | Not directly available in ONS surveys |
| Social expenditure | Track aggregate welfare spending | float | `SOCX_AGG.csv` | Benefit income from WAS survey |
| Income inequality | Measure aggregate economic distribution | float | `GINI_COEF.csv` | Income distribution from ASHE and WAS surveys |
| Household savings | Track aggregate consumer saving behavior | float | `DP_LIVE_HH_SAV_RATE.csv` | Savings account values from WAS survey |
| Business birth/death rates | Monitor aggregate economic dynamism | float | `SDBS_BDI_ISIC4_BIRTH.csv`, `SDBS_BDI_ISIC4_DEATH.csv` | Not directly available in ONS surveys |
| Corporate tax statistics | Track aggregate business taxation | float | `CTS_CIT.csv` | Not directly available in ONS surveys |
| Producer price inflation | Monitor aggregate price changes | float | `DP_LIVE_PPI_INFL.csv` | Not directly available in ONS surveys |
| Key economic indicators | Track aggregate economic health | float | `KEI.csv` | Not directly available in ONS surveys |

### 5. World Bank

Aggregate time-series data for each country.

- **Script**: `economic_data/world_bank_reader.py`
- **Files**:
  - `world_bank/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_4325868.csv` - Unemployment rates
  - `world_bank/API_SL.TLF.CACT.NE.ZS_DS2_en_csv_v2_4354787.csv` - Labor force participation
  - `world_bank/API_GC.TAX.GSRV.VA.ZS_DS2_en_csv_v2_4028900.csv` - VAT rates
  - `world_bank/API_GC.TAX.EXPT.CN_DS2_en_csv_v2_4157140.csv` - Export tax rates
  - `world_bank/API_SI.POV.GINI_DS2_en_csv_v2_5358360.csv` - Gini coefficients
  - `world_bank/API_SP.DYN.TFRT.IN_DS2_en_csv_v2_4151057.csv` - Fertility rates
  - `world_bank/API_FR.INR.RINR_DS2_en_csv_v2_4150781.csv` - Interest rates on government debt
  - `world_bank/LONG_TERM_IR.csv` - Long-term interest rates
  - `world_bank/SHORT_TERM_IR.csv` - Short-term interest rates
  - `world_bank/ppi.csv` - Producer price indices
  - `world_bank/cpi.csv` - Consumer price indices
  - `world_bank/API_NY.GDP.MKTP.CN_DS2_en_csv_v2_5358562.csv` - GDP (current LCU)
  - `world_bank/API_SP.POP.TOTL_DS2_en_csv_v2_79.csv` - Total population
  - `world_bank/central_gov_debt.csv` - Central government debt
  - `world_bank/npl_ratios.csv` - Non-performing loan ratios
  - `world_bank/inflation_arg.csv` - Argentina inflation data
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| GDP data | Track aggregate economic output | float | `API_NY.GDP.MKTP.CN_DS2_en_csv_v2_5358562.csv` | Not directly available in ONS surveys |
| Population statistics | Monitor aggregate demographic trends | integer | `API_SP.POP.TOTL_DS2_en_csv_v2_79.csv` | Household size and composition from ASHE survey |
| Labor market indicators | Track aggregate employment conditions | float | `API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_4325868.csv`, `API_SL.TLF.CACT.NE.ZS_DS2_en_csv_v2_4354787.csv` | Employment status and economic activity from ASHE survey |
| Tax rates | Monitor aggregate fiscal policy | float | `API_GC.TAX.GSRV.VA.ZS_DS2_en_csv_v2_4028900.csv`, `API_GC.TAX.EXPT.CN_DS2_en_csv_v2_4157140.csv` | Not directly available in ONS surveys |
| Income inequality measures | Track aggregate economic distribution | float | `API_SI.POV.GINI_DS2_en_csv_v2_5358360.csv` | Income distribution from ASHE and WAS surveys |
| Interest rates | Monitor aggregate monetary conditions | float | `API_FR.INR.RINR_DS2_en_csv_v2_4150781.csv`, `LONG_TERM_IR.csv`, `SHORT_TERM_IR.csv` | Not directly available in ONS surveys |
| Price indices | Track aggregate inflation | float | `ppi.csv`, `cpi.csv` | Not directly available in ONS surveys |
| Government debt data | Monitor aggregate fiscal health | float | `central_gov_debt.csv` | Not directly available in ONS surveys |
| Financial sector health | Track aggregate banking stability | float | `npl_ratios.csv` | Not directly available in ONS surveys |
| Fertility rates | Monitor aggregate demographic trends | float | `API_SP.DYN.TFRT.IN_DS2_en_csv_v2_4151057.csv` | Household composition from ASHE survey |
| Inflation data | Track aggregate price stability | float | `inflation_arg.csv` | Not directly available in ONS surveys |

### 6. IMF

Aggregate time-series data for each country.

- **Script**: `economic_data/imf_reader.py`
- **Files**:
  - `imf/IFS.csv` - International Financial Statistics
  - `imf/imf_fas_bank_demographics.csv` - Bank demographics data
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Banking sector statistics | Monitor aggregate financial sector structure | float | `imf_fas_bank_demographics.csv` | Not directly available in ONS surveys |
| National accounts growth rates | Track aggregate economic performance | float | `IFS.csv` | Not directly available in ONS surveys |
| Inflation rates | Monitor aggregate price stability | float | `IFS.csv` | Not directly available in ONS surveys |
| Labor market statistics | Track aggregate employment conditions | float | `IFS.csv` | Employment status from ASHE survey |
| Financial access indicators | Monitor aggregate financial inclusion | float | `IFS.csv` | Financial asset ownership from WAS survey |
| Banking sector demographics | Track aggregate financial sector composition | float | `imf_fas_bank_demographics.csv` | Not directly available in ONS surveys |

### 7. Exchange Rates

Aggregate time-series data for each currency pair.

- **Script**: `economic_data/exchange_rates.py`
- **Files**:
  - `exchange_rates.csv` - World Bank exchange rate data
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| USD exchange rates | Convert aggregate values to USD | float | `exchange_rates.csv` | Not directly available in ONS surveys |
| EUR exchange rates | Convert aggregate values to EUR | float | `exchange_rates.csv` | Not directly available in ONS surveys |
| Local currency units | Track aggregate currency values | float | `exchange_rates.csv` | Not directly available in ONS surveys |
| Special handling rates | Handle aggregate regional arrangements | float | `exchange_rates.csv` | Not directly available in ONS surveys |

### 8. Policy Rates

Aggregate time-series data for each country.

- **Script**: `economic_data/policy_rates.py`
- **Files**:
  - `policy_rates.csv` - Main policy rates data file
  - `policy_rate_sgp.csv` - Singapore policy rates (SORA)
  - `policy_rate_cri.csv` - Costa Rica policy rates
  - `country_codes.csv` - Country code mappings
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Central bank policy rates | Track aggregate monetary policy | float | `policy_rates.csv` | Not directly available in ONS surveys |
| Special handling rates | Handle aggregate regional arrangements | float | `policy_rate_sgp.csv`, `policy_rate_cri.csv` | Not directly available in ONS surveys |
| Default rates | Provide aggregate fallback values | float | `policy_rates.csv` | Not directly available in ONS surveys |

### 9. European Central Bank (ECB)

Aggregate time-series data for each country.

- **Script**: `economic_data/ecb_reader.py`
- **Files**:
  - `firm_loans.csv` - Firm loan interest rates
  - `household_loans_for_consumption.csv` - Household consumption loan rates
  - `household_loans_for_mortgages.csv` - Household mortgage rates
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Firm loan rates | Track aggregate business borrowing costs | float | `firm_loans.csv` | Not directly available in ONS surveys |
| Household consumption loan rates | Monitor aggregate consumer credit costs | float | `household_loans_for_consumption.csv` | Consumer debt levels from WAS survey |
| Household mortgage rates | Track aggregate housing finance costs | float | `household_loans_for_mortgages.csv` | Mortgage debt and property values from WAS survey |

### 10. Office for National Statistics (ONS)

UK distributional data on industries.

- **Script**: `economic_data/ons_reader.py`
- **Files**:
  - `UKCompSizes.csv` - UK firm size distribution data
  - `UKSec_map.csv` - UK sector mapping (SIC07 to ISIC)
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Firm size distributions | Track distribution of firm sizes | float | `UKCompSizes.csv` | Employment size from ASHE survey |
| Zeta distribution parameters | Model firm size distribution | float | `UKCompSizes.csv` | Not directly available in ONS surveys |
| Industry classification mappings | Standardize sector classifications | string | `UKSec_map.csv` | Industry classification (SIC codes) from ASHE survey |

### 11. Household Finance and Consumption Survey (HFCS)

Micro-level household survey data for European countries.

This is the data we are seeking to replace with UK data, so import to SRS is desirable but not essential. If we are able to import to SRS we would be able to run simulations with multiple countries in parallel, for example to understand coevolution and interdependency between the Republic of Ireland and the United Kingdom.

- **Source**: European Central Bank (ECB)
- **Script**: `population_data/hfcs_reader.py`
- **Files**:
  - `hfcs/{country}_{year}_P.csv` - Individual-level data
  - `hfcs/{country}_{year}_H.csv` - Household-level data
  - `hfcs/{country}_{year}_D.csv` - Derived variables
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| **Individual Characteristics** | | | | |
| ID/HID/iid | Unique identifiers and household links | string | `{country}_{year}_P.csv` | `pidno` (Personal identifier) from WAS survey |
| HW0010 | Survey weight | float | `{country}_{year}_P.csv` | `R7xsperswgt` (Person weight) from WAS survey |
| RA0200 | Gender | string | `{country}_{year}_P.csv` | `sexr7` (Sex) from WAS survey |
| RA0300 | Age | integer | `{country}_{year}_P.csv` | `dvager7` (Age) from WAS survey, `age_ashe` from ASHE survey |
| PA0200 | Education level | string | `{country}_{year}_P.csv` | `edlevelr7` (Education level) from WAS survey |
| PE0100a | Labour status | string | `{country}_{year}_P.csv` | `wrkingr7` (Working status) from WAS survey, `ecopuk11_cen` (economic activity) from ASHE survey |
| PE0400 | Employment industry | string | `{country}_{year}_P.csv` | `sic07_ashe` (Industry SIC codes) from ASHE survey |
| DHHTYPE | Household type | string | `{country}_{year}_H.csv` | `ahthuk11_cen` (Household type) from ASHE survey |
| **Income Sources** | | | | |
| PG0110 | Employee income | float | `{country}_{year}_P.csv` | `dvgrspayannualr7` (Gross annual income), `dvnetpayannualr7` (Net annual income) from WAS survey, `agp_ashe` (Annual gross pay) from ASHE survey |
| PG0210 | Self-employment income | float | `{country}_{year}_P.csv` | `dvsegrspayr7` (Gross annual self-employed income), `dvsenetpayr7` (Net annual self-employed income), `DVGISER7` (Total Annual Gross self employed income) from WAS survey |
| DI1300 | Rental income from real estate | float | `{country}_{year}_P.csv` | `dvrentincamannualr7` (Annual amount received from rental income), `DVGrsRentAmtAnnualR7_aggr` (Gross rental income), `DVNetRentAmtAnnualR7_aggr` (Net rental income) from WAS survey |
| DI1400 | Income from financial assets | float | `{country}_{year}_P.csv` | `DVGIINVR7_aggr` (Gross investment income), `DVNIINVR7_aggr` (Net investment income) from WAS survey |
| DI1500 | Income from pensions | float | `{country}_{year}_P.csv` | `DVPenInc1R7` (Annual pension income first pension), `DVPenInc2R7` (Annual pension income second pension), `popensionmval1r7` (Net monthly income from occupational pension) from WAS survey |
| DI1620 | Regular social transfers | float | `{country}_{year}_P.csv` | `DVBenefitAnnualR7_aggr` (Total benefits received), `wageben1r7` (Working Age Benefits), `disben1r7` (Disability Benefits), `penben1r7` (Pensioner Benefits) from WAS survey |
| DI2000 | Total income | float | `{country}_{year}_P.csv` | `totpartgrspayr7` (Total partners income) from WAS survey |
| PG0510 | Income from unemployment benefits | float | `{country}_{year}_P.csv` | `unidkr7` (Unemployment benefit indicators), `unifdkr7` (Unemployment benefit indicators) from WAS survey - No direct unemployment benefit amount variable available |
| **Household Assets** | | | | |
| DA1110 | Value of main residence | float | `{country}_{year}_H.csv` | `DVPropertyR7` (Sum of all property values) from WAS survey |
| DA1120 | Value of other properties | float | `{country}_{year}_H.csv` | `DVPropertyR7` (Sum of all property values) from WAS survey - No separate variable for other properties |
| DA1130 | Value of household vehicles | float | `{country}_{year}_H.csv` | No direct vehicle asset value variable available in WAS survey |
| DA1131 | Value of household valuables | float | `{country}_{year}_H.csv` | `DVGcollVR7` (Value of collectables and valuables), `gcollvr7` (Estimated value of valuables) from WAS survey |
| DA1140 | Value of self-employment businesses | float | `{country}_{year}_H.csv` | No direct business asset value variable available in WAS survey |
| DA2101 | Wealth in deposits | float | `{country}_{year}_H.csv` | `DVSaValR7_SUM` (Total value of savings accounts) from WAS survey |
| DA2102 | Mutual funds | float | `{country}_{year}_H.csv` | `DVFFAssetsR7_SUM` (Total value of formal financial assets) from WAS survey |
| DA2103 | Bonds | float | `{country}_{year}_H.csv` | `DVFFAssetsR7_SUM` (Total value of formal financial assets) from WAS survey |
| DA2104 | Value of private businesses | float | `{country}_{year}_H.csv` | No direct private business value variable available in WAS survey |
| DA2105 | Shares | float | `{country}_{year}_H.csv` | `DVFFAssetsR7_SUM` (Total value of formal financial assets) from WAS survey |
| DA2106 | Managed accounts | float | `{country}_{year}_H.csv` | `DVFFAssetsR7_SUM` (Total value of formal financial assets) from WAS survey |
| DA2107 | Money owed to households | float | `{country}_{year}_H.csv` | No direct receivables variable available in WAS survey |
| DA2108 | Other assets | float | `{country}_{year}_H.csv` | `DVFInvOtVR7` (Value of other investments) from WAS survey |
| DA2109 | Voluntary pension | float | `{country}_{year}_H.csv` | `TOTPENR7` (Total value of individual pension wealth) from WAS survey |
| **Household Liabilities** | | | | |
| DL1110 | Outstanding balance of HMR mortgages | float | `{country}_{year}_H.csv` | `dburdhr7` (Burden of mortgage and other debt on household) from WAS survey |
| DL1120 | Outstanding balance of mortgages on other properties | float | `{country}_{year}_H.csv` | `dburdhr7` (Burden of mortgage and other debt on household) from WAS survey |
| DL1210 | Outstanding balance of credit line | float | `{country}_{year}_H.csv` | `dburdhr7` (Burden of mortgage and other debt on household) from WAS survey |
| DL1220 | Outstanding balance of credit card debt | float | `{country}_{year}_H.csv` | `dburdhr7` (Burden of mortgage and other debt on household) from WAS survey |
| DL1230 | Outstanding balance of other non-mortgage loans | float | `{country}_{year}_H.csv` | `dburdhr7` (Burden of mortgage and other debt on household) from WAS survey |
| **Housing Characteristics** | | | | |
| HB0300 | Tenure status of main residence | string | `{country}_{year}_H.csv` | `hhldrr7` (Whether owns or rents accommodation) from WAS survey |
| HB2300 | Rent paid | float | `{country}_{year}_H.csv` | No direct rent payment variable available in WAS survey |
| HB2410 | Number of properties other than main residence | integer | `{country}_{year}_H.csv` | No direct property count variable available in WAS survey |
| **Consumption Patterns** | | | | |
| DOCOGOODP | Consumption as share of income | float | `{country}_{year}_D.csv` | No directly available in ONS surveys |
| HI0220 | Amount spent on consumption | float | `{country}_{year}_D.csv` | No directly available in ONS surveys |

### 12. Emissions Data

Aggregate time-series data for each fuel type.

- **Source**: Federal Reserve Bank of St. Louis (FRED)
- **Script**: `emissions/emissions_reader.py`
- **Files**:
  - `emissions/PCOALAUUSDM.csv` - Coal prices
  - `emissions/POILBREUSDM.csv` - Oil prices
  - `emissions/PNGASEUUSDM.csv` - Natural gas prices
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Fuel prices | Track aggregate energy costs | float | `PCOALAUUSDM.csv`, `POILBREUSDM.csv`, `PNGASEUUSDM.csv` | Not directly available in ONS surveys |
| Emissions factors | Calculate aggregate CO2 emissions | float | Derived from price data | Not directly available in ONS surveys |
| Energy conversion rates | Convert between aggregate energy units | float | Constants in code | Not directly available in ONS surveys |
| Refining emissions | Track aggregate emissions from refining | float | Derived from ICIO data | Not directly available in ONS surveys |

### 13. Goods Criticality Data

Aggregate data on critical goods and supply chain dependencies.

- **Script**: `criticality_data/goods_criticality_reader.py`
- **Files**:
  - `criticality/goods_criticality.csv` - Criticality matrix
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Criticality matrix | Track aggregate supply chain dependencies | float | `goods_criticality.csv` | Not directly available in ONS surveys |
| NACE sector mappings | Map aggregate sectors to NACE 1-digit level | string | Derived from matrix data | Industry classification from ASHE survey |
| Supply-demand relationships | Monitor aggregate critical goods flows | float | `goods_criticality.csv` | Not directly available in ONS surveys |

### 14. Compustat Banks Data

Micro-level bank financial data from Compustat.

- **Source**: Compustat Database
- **Script**: `population_data/compustat_banks_reader.py`
- **Files**:
  - `compustat/quarterly.csv` - Quarterly bank financial statements
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Balance sheet items | Track individual bank assets and liabilities | float | `quarterly.csv` | Not directly available in ONS surveys |
| Funding sources | Monitor individual bank deposits and debt | float | `quarterly.csv` | Not directly available in ONS surveys |
| Debt dynamics | Track individual bank debt issuance | float | `quarterly.csv` | Not directly available in ONS surveys |
| Income data | Monitor individual bank profitability | float | `quarterly.csv` | Not directly available in ONS surveys |
| Currency information | Handle individual bank currency conversions | string | `quarterly.csv` | Not directly available in ONS surveys |

### 15. Compustat Firms Data

Micro-level firm financial data from Compustat.

- **Source**: Compustat Database
- **Script**: `population_data/compustat_firms_reader.py`
- **Files**:
  - `compustat/annual.csv` - Annual firm financial statements
  - `compustat/quarterly.csv` - Quarterly firm financial statements
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Employment data | Track individual firm workforce size | integer | `annual.csv` | Employment size from ASHE survey |
| Balance sheet items | Monitor individual firm assets and liabilities | float | `quarterly.csv` | Not directly available in ONS surveys |
| Income statement | Track individual firm revenue and profits | float | `quarterly.csv` | Business income from WAS survey |
| Operational data | Monitor individual firm inventory levels | float | `quarterly.csv` | Not directly available in ONS surveys |
| Sector information | Track individual firm industry classification | string | `annual.csv` | Industry classification from ASHE survey |
| Currency information | Handle individual firm currency conversions | string | `quarterly.csv` | Not directly available in ONS surveys |

### 16. World Bank Employment Data

Aggregate employment statistics from the World Bank.

- **Source**: World Bank Database
- **Script**: `population_data/employment_data.py`
- **Files**:
  - `employment_data.h5` - HDF5 file containing employment statistics
- **Variables**:

| Variable Name | Purpose in the Model | Data Type | CSV File | Potential ONS Variable |
|---------------|---------------------|-----------|-----------|----------------------|
| Unemployment rates | Track aggregate labor market conditions | float | `employment_data.h5` | Employment status from ASHE survey |
| Labor force participation rates | Monitor aggregate workforce engagement | float | `employment_data.h5` | Economic activity from ASHE survey |

## Utility Scripts

### Industry Data Processing

- **Script**: `util/industry_extraction.py`
- **Purpose**: Extract and process industry-level data from various sources
- **Features**:
  - Industry classification mapping
  - Data aggregation by sector
  - Cross-source data alignment

### Data Pruning Utilities

- **Script**: `util/prune_util.py`
- **Purpose**: Clean and filter data for model use
- **Features**:
  - Remove redundant data
  - Filter by relevance
  - Optimize data storage

### Industry Mappings

- **Script**: `io_tables/mappings.py`
- **Purpose**: Standardize industry classifications across data sources
- **Features**:
  - Cross-walk between different industry codes
  - Sector aggregation rules
  - Industry hierarchy management

### Industry Definitions

- **Script**: `io_tables/industries.py`
- **Purpose**: Define industry categories and relationships
- **Features**:
  - Industry code definitions
  - Sector groupings
  - Industry hierarchies