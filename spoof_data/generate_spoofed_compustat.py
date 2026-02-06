"""
Generate spoofed Compustat data to replace confidential company information.

This script:
1. Removes direct identifiers (gvkey, conm, tic)
2. Spoofs financial metrics while maintaining statistical properties
3. Preserves structural relationships (e.g., company names for merging)
4. Maintains time periods, countries, and sectoral distributions
"""

from pathlib import Path

import numpy as np
import pandas as pd


class CompustatSpoofer:
    """Generate spoofed Compustat data maintaining statistical properties."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Storage for spoofed data
        self.annual_spoofed = None
        self.quarterly_spoofed = None
        self.banks_spoofed = None

        # Mapping of original to spoofed company names (for merge consistency)
        self.company_name_map = {}

    def generate_fake_company_name(self, idx: int) -> str:
        """Generate a fake but realistic-looking company name."""
        prefixes = [
            "ACME",
            "GLOBAL",
            "UNITED",
            "FIRST",
            "NATIONAL",
            "CAPITAL",
            "PREMIER",
            "ROYAL",
            "CENTRAL",
            "WESTERN",
            "EASTERN",
            "NORTHERN",
            "SOUTHERN",
            "ATLANTIC",
            "PACIFIC",
            "CONTINENTAL",
            "UNIVERSAL",
            "INTERNATIONAL",
            "GENERAL",
            "STANDARD",
            "FEDERAL",
            "COMMONWEALTH",
        ]

        types = [
            "INDUSTRIES",
            "GROUP",
            "HOLDINGS",
            "CORP",
            "INC",
            "LTD",
            "ENTERPRISES",
            "SOLUTIONS",
            "SYSTEMS",
            "SERVICES",
            "PARTNERS",
            "TECHNOLOGIES",
            "RESOURCES",
            "PRODUCTS",
            "MANUFACTURING",
        ]

        # Generate deterministic name based on index
        np.random.seed(self.random_seed + idx)
        prefix = np.random.choice(prefixes)
        company_type = np.random.choice(types)

        # Add a number for uniqueness
        return f"{prefix} {company_type} {idx:04d}"

    def spoof_categorical_column(self, series: pd.Series, preserve_values: bool = False) -> pd.Series:
        """
        Spoof a categorical column by resampling from the distribution.

        Args:
            series: Original pandas Series
            preserve_values: If True, keep original values (for time/country)

        Returns:
            Spoofed series with same distribution
        """
        if preserve_values:
            return series.copy()

        # For categorical, resample from observed distribution
        value_counts = series.value_counts(normalize=True, dropna=False)
        spoofed = np.random.choice(value_counts.index, size=len(series), p=value_counts.values, replace=True)
        return pd.Series(spoofed, index=series.index)

    def spoof_numerical_column(
        self, series: pd.Series, distribution: str = "auto", preserve_nans: bool = True
    ) -> pd.Series:
        """
        Spoof a numerical column by fitting and sampling from a distribution.

        Args:
            series: Original pandas Series
            distribution: 'auto', 'normal', or 'lognormal'
            preserve_nans: If True, maintain original NaN pattern

        Returns:
            Spoofed series with similar statistical properties
        """
        non_missing = series.dropna()

        if len(non_missing) == 0:
            return series.copy()

        # Detect if this is actually categorical (low cardinality)
        n_unique = non_missing.nunique()
        if n_unique <= 20:
            # Treat as categorical - resample from observed values
            value_counts = non_missing.value_counts(normalize=True)
            spoofed_values = np.random.choice(
                value_counts.index, size=len(non_missing), p=value_counts.values, replace=True
            )
        else:
            # Auto-detect distribution
            if distribution == "auto":
                # Use lognormal if all positive, normal otherwise
                distribution = "lognormal" if non_missing.min() >= 0 else "normal"

            if distribution == "lognormal":
                # For strictly positive values
                positive_values = non_missing[non_missing > 0]
                if len(positive_values) == 0:
                    # All zeros - keep as zeros
                    spoofed_values = np.zeros(len(non_missing))
                else:
                    log_vals = np.log(positive_values)
                    mean_log = log_vals.mean()
                    std_log = log_vals.std()

                    if std_log == 0:
                        # No variation - use constant
                        spoofed_values = np.full(len(non_missing), positive_values.mean())
                    else:
                        spoofed_values = np.random.lognormal(mean=mean_log, sigma=std_log, size=len(non_missing))
            else:  # normal
                mean = non_missing.mean()
                std = non_missing.std()

                if std == 0:
                    # No variation - use constant
                    spoofed_values = np.full(len(non_missing), mean)
                else:
                    spoofed_values = np.random.normal(loc=mean, scale=std, size=len(non_missing))

        # Create result series maintaining NaN pattern
        if preserve_nans:
            result = pd.Series(index=series.index, dtype=float)
            result.loc[~series.isna()] = spoofed_values
        else:
            result = pd.Series(spoofed_values, index=series.index)

        return result

    def spoof_firms_annual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spoof firms_annual.csv data."""
        print("\n" + "=" * 60)
        print("Spoofing firms_annual.csv")
        print("=" * 60)

        spoofed = df.copy()

        # Generate fake company names
        print("  Generating fake company names...")
        unique_companies = df["conm"].unique()
        for idx, company in enumerate(unique_companies):
            fake_name = self.generate_fake_company_name(idx)
            self.company_name_map[company] = fake_name

        # Remove/anonymize identifiers
        print("  Removing gvkey (company identifier)...")
        spoofed = spoofed.drop(columns=["gvkey"])

        print("  Replacing conm with anonymized names...")
        spoofed["conm"] = spoofed["conm"].map(self.company_name_map)

        # Preserve time and location
        print("  Preserving fyear, datadate, loc...")
        # (Already in place, no changes needed)

        # Spoof employment
        print("  Spoofing emp (employment)...")
        spoofed["emp"] = self.spoof_numerical_column(spoofed["emp"], distribution="lognormal")

        # Spoof index column
        spoofed["Unnamed: 0"] = np.arange(len(spoofed))

        print(f"  ✓ Spoofed {len(spoofed)} annual records")
        return spoofed

    def spoof_firms_quarterly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spoof firms_quarterly.csv data."""
        print("\n" + "=" * 60)
        print("Spoofing firms_quarterly.csv")
        print("=" * 60)

        spoofed = df.copy()

        # Remove/anonymize identifiers
        print("  Removing gvkey (company identifier)...")
        spoofed = spoofed.drop(columns=["gvkey"])

        print("  Replacing conm with anonymized names...")
        # Use the same mapping created from annual data
        spoofed["conm"] = spoofed["conm"].map(self.company_name_map)

        # Handle companies in quarterly but not in annual (shouldn't happen much)
        unmapped = spoofed["conm"].isna()
        if unmapped.any():
            print(f"    ⚠️  {unmapped.sum()} companies in quarterly not in annual")
            # Generate new names for these
            start_idx = len(self.company_name_map)
            for idx, orig_name in enumerate(df.loc[unmapped, "conm"].unique()):
                fake_name = self.generate_fake_company_name(start_idx + idx)
                spoofed.loc[spoofed["conm"].isna() & (df["conm"] == orig_name), "conm"] = fake_name

        print("  Removing conml (long company name)...")
        spoofed = spoofed.drop(columns=["conml"])

        # Preserve time, location, sector, currency
        print("  Preserving fqtr, fyearq, datadate, loc, gsector, curcdq...")

        # Spoof financial metrics
        financial_cols = {
            "atq": "lognormal",  # Assets
            "ceqq": "normal",  # Equity (can be negative)
            "dlttq": "lognormal",  # Debt
            "dptbq": "lognormal",  # Deposits
            "gpq": "normal",  # Profits (can be negative)
            "invtq": "lognormal",  # Inventory
            "lltq": "normal",  # Long-term liabilities (can be negative)
            "ltq": "lognormal",  # Total liabilities
            "revtq": "normal",  # Revenue (can be negative in rare cases)
        }

        for col, dist in financial_cols.items():
            print(f"  Spoofing {col}...")
            spoofed[col] = self.spoof_numerical_column(spoofed[col], distribution=dist)

        # Spoof index column
        spoofed["Unnamed: 0"] = np.arange(len(spoofed))

        print(f"  ✓ Spoofed {len(spoofed)} quarterly records")
        return spoofed

    def spoof_banks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spoof banks.csv data."""
        print("\n" + "=" * 60)
        print("Spoofing banks.csv")
        print("=" * 60)

        spoofed = df.copy()

        # Remove/anonymize identifiers
        print("  Removing gvkey (company identifier)...")
        spoofed = spoofed.drop(columns=["gvkey"])

        print("  Removing tic (ticker symbol)...")
        spoofed = spoofed.drop(columns=["tic"])

        print("  Generating fake bank names...")
        # Banks are separate - create new names
        unique_banks = df["conm"].unique()
        bank_name_map = {}
        for idx, bank in enumerate(unique_banks):
            # Use different seed offset for banks
            fake_name = self.generate_fake_company_name(10000 + idx)
            bank_name_map[bank] = fake_name

        spoofed["conm"] = spoofed["conm"].map(bank_name_map)

        # Preserve time, location, currency
        print("  Preserving fqtr, fyearq, datadate, loc, curcdq...")

        # Spoof financial metrics
        financial_cols = {
            "atq": "lognormal",  # Assets
            "ciq": "normal",  # Income (can be negative)
            "dlttq": "lognormal",  # Debt
            "dptcq": "lognormal",  # Deposits
            "ltq": "lognormal",  # Liabilities
            "teqq": "lognormal",  # Equity
            "dltisy": "lognormal",  # Debt issuance
            "dltry": "lognormal",  # Debt reduction
        }

        for col, dist in financial_cols.items():
            print(f"  Spoofing {col}...")
            spoofed[col] = self.spoof_numerical_column(spoofed[col], distribution=dist)

        # Spoof index column
        spoofed["Unnamed: 0"] = np.arange(len(spoofed))

        print(f"  ✓ Spoofed {len(spoofed)} bank records")
        return spoofed

    def spoof_all(
        self, annual_path: Path, quarterly_path: Path, banks_path: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Spoof all three Compustat files.

        Returns:
            (annual_spoofed, quarterly_spoofed, banks_spoofed)
        """
        print("=" * 60)
        print("Compustat Data Spoofing")
        print("=" * 60)

        # Load original data
        print("\nLoading original data...")
        annual_orig = pd.read_csv(annual_path)
        quarterly_orig = pd.read_csv(quarterly_path)
        banks_orig = pd.read_csv(banks_path)

        print(f"  ✓ Loaded {annual_path.name}: {annual_orig.shape}")
        print(f"  ✓ Loaded {quarterly_path.name}: {quarterly_orig.shape}")
        print(f"  ✓ Loaded {banks_path.name}: {banks_orig.shape}")

        # Spoof each file
        # IMPORTANT: Spoof annual first to build company name mapping
        self.annual_spoofed = self.spoof_firms_annual(annual_orig)
        self.quarterly_spoofed = self.spoof_firms_quarterly(quarterly_orig)
        self.banks_spoofed = self.spoof_banks(banks_orig)

        return self.annual_spoofed, self.quarterly_spoofed, self.banks_spoofed

    def save_spoofed_data(self, output_dir: Path):
        """Save spoofed data to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("Saving Spoofed Data")
        print("=" * 60)

        annual_path = output_dir / "firms_annual.csv"
        quarterly_path = output_dir / "firms_quarterly.csv"
        banks_path = output_dir / "banks.csv"

        self.annual_spoofed.to_csv(annual_path, index=False)
        print(f"  ✓ Saved {annual_path}")

        self.quarterly_spoofed.to_csv(quarterly_path, index=False)
        print(f"  ✓ Saved {quarterly_path}")

        self.banks_spoofed.to_csv(banks_path, index=False)
        print(f"  ✓ Saved {banks_path}")


def main():
    """Main spoofing function."""
    # Paths
    data_dir = Path("tests/test_macro_data/unit/sample_raw_data/compustat")
    original_dir = Path("spoof_data/original_compustat")
    output_dir = data_dir  # Replace original data

    # Move original data to backup if not already done
    if not original_dir.exists():
        print("Creating backup of original Compustat data...")
        original_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["firms_annual.csv", "firms_quarterly.csv", "banks.csv"]:
            src = data_dir / filename
            dst = original_dir / filename
            if src.exists():
                import shutil

                shutil.copy2(src, dst)
                print(f"  ✓ Backed up {filename}")

    # Create spoofer and generate data
    spoofer = CompustatSpoofer(random_seed=42)

    annual_spoofed, quarterly_spoofed, banks_spoofed = spoofer.spoof_all(
        annual_path=original_dir / "firms_annual.csv",
        quarterly_path=original_dir / "firms_quarterly.csv",
        banks_path=original_dir / "banks.csv",
    )

    # Save spoofed data
    spoofer.save_spoofed_data(output_dir)

    # Verify merge will work
    print("\n" + "=" * 60)
    print("Verification: Testing Merge Compatibility")
    print("=" * 60)

    annual_companies = set(annual_spoofed["conm"].unique())
    quarterly_companies = set(quarterly_spoofed["conm"].unique())
    overlap = annual_companies & quarterly_companies

    print(f"  Annual companies: {len(annual_companies)}")
    print(f"  Quarterly companies: {len(quarterly_companies)}")
    print(f"  Overlap (for merging): {len(overlap)}")

    if len(overlap) > 0:
        print(f"  ✅ Merge will succeed with {len(overlap)} companies")
    else:
        print(f"  ❌ WARNING: No company overlap - merge will fail!")

    print("\n" + "=" * 60)
    print("Spoofing Complete!")
    print("=" * 60)
    print("\nOriginal data backed up to:", original_dir)
    print("Spoofed data saved to:", output_dir)


if __name__ == "__main__":
    main()
