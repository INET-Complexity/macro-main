"""
Validate Compustat data coherence.

This script checks that the Compustat data files are internally consistent
and satisfy the requirements for the macromodel data processing pipeline.
"""

from pathlib import Path

import pandas as pd


def validate_firms_data(
    annual: pd.DataFrame, quarterly: pd.DataFrame
) -> tuple[bool, list[str]]:
    """
    Validate firms annual and quarterly data for consistency.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    print("\n" + "="*60)
    print("Validating Firms Data")
    print("="*60)

    # Check 1: conm overlap (used for merging)
    annual_conm = set(annual["conm"].dropna().unique())
    quarterly_conm = set(quarterly["conm"].dropna().unique())
    overlap = annual_conm & quarterly_conm

    print(f"\n1. Company name (conm) overlap:")
    print(f"   Annual companies: {len(annual_conm)}")
    print(f"   Quarterly companies: {len(quarterly_conm)}")
    print(f"   Overlap: {len(overlap)}")

    if len(overlap) == 0:
        errors.append("No company names overlap between annual and quarterly data - merge will fail")
    elif len(overlap) < len(annual_conm) * 0.9:
        errors.append(
            f"Only {len(overlap)}/{len(annual_conm)} annual companies have quarterly data"
        )

    # Check 2: gvkey uniqueness within files (skip if removed for spoofing)
    if "gvkey" in annual.columns:
        annual_gvkey_dups = annual["gvkey"].duplicated().sum()
        print(f"\n2. gvkey duplicates in annual: {annual_gvkey_dups}")
        if annual_gvkey_dups > 0:
            errors.append(f"Annual data has {annual_gvkey_dups} duplicate gvkey values")
    else:
        print(f"\n2. gvkey column removed (spoofed data) - skipping uniqueness check")

    # Check 3: Time period consistency
    annual_years = annual["fyear"].unique()
    quarterly_years = quarterly["fyearq"].unique()
    print(f"\n3. Time periods:")
    print(f"   Annual years: {sorted(annual_years)}")
    print(f"   Quarterly years: {sorted(quarterly_years)}")

    if not set(annual_years).intersection(set(quarterly_years)):
        errors.append("No overlapping years between annual and quarterly data")

    # Check 4: Country consistency
    annual_countries = set(annual["loc"].unique())
    quarterly_countries = set(quarterly["loc"].unique())
    print(f"\n4. Countries:")
    print(f"   Annual: {sorted(annual_countries)}")
    print(f"   Quarterly: {sorted(quarterly_countries)}")

    if not annual_countries.issubset(quarterly_countries):
        missing = annual_countries - quarterly_countries
        errors.append(f"Annual has countries not in quarterly: {missing}")

    # Check 5: Required columns exist (flexible for spoofed data)
    # Core columns needed for data processing (identifiers may be removed)
    annual_required = ["fyear", "datadate", "emp", "conm", "loc"]
    quarterly_required = [
        "curcdq", "fqtr", "fyearq", "datadate", "atq", "ceqq",
        "dlttq", "gpq", "invtq", "ltq", "revtq", "conm", "gsector", "loc"
    ]

    print(f"\n5. Required columns:")
    annual_missing = set(annual_required) - set(annual.columns)
    quarterly_missing = set(quarterly_required) - set(quarterly.columns)

    if annual_missing:
        errors.append(f"Annual missing columns: {annual_missing}")
        print(f"   ❌ Annual missing: {annual_missing}")
    else:
        print(f"   ✓ Annual has all required columns")

    if quarterly_missing:
        errors.append(f"Quarterly missing columns: {quarterly_missing}")
        print(f"   ❌ Quarterly missing: {quarterly_missing}")
    else:
        print(f"   ✓ Quarterly has all required columns")

    # Check 6: Financial values are reasonable
    print(f"\n6. Financial data sanity checks:")

    # Employment should be >= 0
    if (annual["emp"].dropna() < 0).any():
        errors.append("Annual data has negative employment values")
        print(f"   ❌ Negative employment in annual")
    else:
        print(f"   ✓ Employment is non-negative")

    # Assets should be >= 0
    if (quarterly["atq"].dropna() < 0).any():
        errors.append("Quarterly data has negative assets")
        print(f"   ❌ Negative assets in quarterly")
    else:
        print(f"   ✓ Assets are non-negative")

    # Liabilities should be >= 0
    if (quarterly["ltq"].dropna() < 0).any():
        errors.append("Quarterly data has negative liabilities")
        print(f"   ❌ Negative liabilities in quarterly")
    else:
        print(f"   ✓ Liabilities are non-negative")

    # Check 7: Currency codes match countries
    print(f"\n7. Currency-country consistency:")
    loc_currency_map = quarterly.groupby("loc")["curcdq"].unique()
    for loc, currencies in loc_currency_map.items():
        print(f"   {loc}: {list(currencies)}")

    return len(errors) == 0, errors


def validate_banks_data(banks: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate banks data for consistency.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    print("\n" + "="*60)
    print("Validating Banks Data")
    print("="*60)

    # Check 1: gvkey uniqueness per time period (skip if removed for spoofing)
    if "gvkey" in banks.columns:
        print(f"\n1. gvkey duplicates per quarter:")
        for quarter in sorted(banks["fqtr"].unique()):
            quarter_data = banks[banks["fqtr"] == quarter]
            dups = quarter_data["gvkey"].duplicated().sum()
            print(f"   Q{quarter}: {dups} duplicates")
            if dups > 0:
                errors.append(f"Quarter {quarter} has {dups} duplicate gvkey values")
    else:
        print(f"\n1. gvkey column removed (spoofed data) - skipping uniqueness check")

    # Check 2: Required columns exist (flexible for spoofed data)
    required_cols = [
        "curcdq", "fqtr", "fyearq", "datadate",
        "atq", "dlttq", "dptcq", "ltq", "teqq", "loc"
    ]

    print(f"\n2. Required columns:")
    missing = set(required_cols) - set(banks.columns)
    if missing:
        errors.append(f"Banks missing columns: {missing}")
        print(f"   ❌ Missing: {missing}")
    else:
        print(f"   ✓ All required columns present")

    # Check 3: Time periods
    years = banks["fyearq"].unique()
    quarters = sorted(banks["fqtr"].unique())
    print(f"\n3. Time periods:")
    print(f"   Years: {sorted(years)}")
    print(f"   Quarters: {quarters}")

    if len(quarters) != 4:
        errors.append(f"Banks data should have 4 quarters, has {len(quarters)}")

    # Check 4: Country distribution
    country_counts = banks["loc"].value_counts()
    print(f"\n4. Country distribution:")
    print(country_counts)

    # Check 5: Financial values are reasonable
    print(f"\n5. Financial data sanity checks:")

    # Assets should be >= 0
    if (banks["atq"].dropna() < 0).any():
        errors.append("Banks data has negative assets")
        print(f"   ❌ Negative assets")
    else:
        print(f"   ✓ Assets are non-negative")

    # Deposits should be >= 0
    if (banks["dptcq"].dropna() < 0).any():
        errors.append("Banks data has negative deposits")
        print(f"   ❌ Negative deposits")
    else:
        print(f"   ✓ Deposits are non-negative")

    # Equity should be positive (usually)
    negative_equity = (banks["teqq"].dropna() < 0).sum()
    if negative_equity > len(banks) * 0.05:  # More than 5% with negative equity
        errors.append(f"{negative_equity} banks have negative equity")
        print(f"   ⚠️  {negative_equity} banks with negative equity")
    else:
        print(f"   ✓ Equity is mostly positive ({negative_equity} negative)")

    # Check 6: Ticker symbols (tic) if present
    if "tic" in banks.columns:
        print(f"\n6. Ticker symbols:")
        tic_missing = banks["tic"].isna().sum()
        tic_unique = banks["tic"].nunique()
        print(f"   Missing: {tic_missing}/{len(banks)} ({tic_missing/len(banks)*100:.1f}%)")
        print(f"   Unique: {tic_unique}")

    return len(errors) == 0, errors


def validate_cross_file_consistency(
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    banks: pd.DataFrame
) -> tuple[bool, list[str]]:
    """
    Validate consistency across all three files.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    print("\n" + "="*60)
    print("Validating Cross-File Consistency")
    print("="*60)

    # Check 1: gvkey overlap (should be minimal - firms vs banks)
    # Skip if gvkey removed for spoofing
    if "gvkey" in annual.columns and "gvkey" in quarterly.columns and "gvkey" in banks.columns:
        firm_gvkeys = set(annual["gvkey"].unique()) | set(quarterly["gvkey"].unique())
        bank_gvkeys = set(banks["gvkey"].unique())
        overlap = firm_gvkeys & bank_gvkeys

        print(f"\n1. gvkey overlap between firms and banks:")
        print(f"   Firms: {len(firm_gvkeys)} unique gvkeys")
        print(f"   Banks: {len(bank_gvkeys)} unique gvkeys")
        print(f"   Overlap: {len(overlap)}")

        if len(overlap) > len(bank_gvkeys) * 0.1:
            errors.append(
                f"Large overlap ({len(overlap)}) between firm and bank gvkeys - may indicate data issues"
            )
    else:
        print(f"\n1. gvkey column removed (spoofed data) - skipping overlap check")

    # Check 2: Time period consistency
    annual_years = set(annual["fyear"].unique())
    quarterly_years = set(quarterly["fyearq"].unique())
    bank_years = set(banks["fyearq"].unique())

    print(f"\n2. Time period consistency:")
    print(f"   Annual: {sorted(annual_years)}")
    print(f"   Quarterly (firms): {sorted(quarterly_years)}")
    print(f"   Quarterly (banks): {sorted(bank_years)}")

    if not (annual_years == quarterly_years == bank_years):
        errors.append("Years are not consistent across all files")

    # Check 3: Country coverage
    annual_countries = set(annual["loc"].unique())
    quarterly_countries = set(quarterly["loc"].unique())
    bank_countries = set(banks["loc"].unique())

    print(f"\n3. Country coverage:")
    print(f"   Annual: {sorted(annual_countries)}")
    print(f"   Quarterly: {sorted(quarterly_countries)}")
    print(f"   Banks: {sorted(bank_countries)}")

    all_countries = annual_countries | quarterly_countries | bank_countries
    print(f"   All countries: {sorted(all_countries)}")

    return len(errors) == 0, errors


def main():
    """Main validation function."""
    data_dir = Path("tests/test_macro_data/unit/sample_raw_data/compustat")

    print("="*60)
    print("Compustat Data Coherence Validation")
    print("="*60)

    # Load data
    print("\nLoading data files...")
    firms_annual = pd.read_csv(data_dir / "firms_annual.csv")
    firms_quarterly = pd.read_csv(data_dir / "firms_quarterly.csv")
    banks = pd.read_csv(data_dir / "banks.csv")

    print(f"✓ Loaded firms_annual.csv: {firms_annual.shape}")
    print(f"✓ Loaded firms_quarterly.csv: {firms_quarterly.shape}")
    print(f"✓ Loaded banks.csv: {banks.shape}")

    # Run validations
    all_valid = True
    all_errors = []

    # Validate firms data
    firms_valid, firms_errors = validate_firms_data(firms_annual, firms_quarterly)
    all_valid = all_valid and firms_valid
    all_errors.extend(firms_errors)

    # Validate banks data
    banks_valid, banks_errors = validate_banks_data(banks)
    all_valid = all_valid and banks_valid
    all_errors.extend(banks_errors)

    # Validate cross-file consistency
    cross_valid, cross_errors = validate_cross_file_consistency(
        firms_annual, firms_quarterly, banks
    )
    all_valid = all_valid and cross_valid
    all_errors.extend(cross_errors)

    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    if all_valid:
        print("✅ All validation checks passed!")
    else:
        print(f"❌ Found {len(all_errors)} errors:")
        for i, error in enumerate(all_errors, 1):
            print(f"   {i}. {error}")

    return all_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
