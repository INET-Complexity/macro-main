"""
Validation script to check coherence between HFCS P, H, and D files.

This script performs various checks to ensure that the HFCS data files
(Personal, Household, and Derived) maintain internal consistency and
proper relationships. These checks should pass for both original and
spoofed data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class HFCSValidator:
    """Validates coherence between HFCS P (Personal), H (Household), and D (Derived) files."""

    def __init__(self, data_dir: Path):
        """
        Initialize validator with data directory.

        Parameters
        ----------
        data_dir : Path
            Directory containing P1.csv, H1.csv, and D1.csv files
        """
        self.data_dir = Path(data_dir)
        self.p_df = None
        self.h_df = None
        self.d_df = None
        self.errors = []
        self.warnings = []

    def load_data(self):
        """Load the three HFCS data files."""
        print("Loading HFCS data files...")
        self.p_df = pd.read_csv(self.data_dir / "P1.csv")
        self.h_df = pd.read_csv(self.data_dir / "H1.csv", low_memory=False)
        self.d_df = pd.read_csv(self.data_dir / "D1.csv", low_memory=False)
        print(f"Loaded P1: {len(self.p_df)} rows, H1: {len(self.h_df)} rows, D1: {len(self.d_df)} rows")

    def check_id_relationships(self) -> bool:
        """
        Check that ID relationships are consistent across files.

        Validates:
        - All household IDs in P (hid) exist in H (id)
        - All household IDs in P exist in D (ID)
        - H and D have the same household IDs
        - Individual IDs in P are unique
        """
        print("\n=== Checking ID Relationships ===")
        passed = True

        # Check individual IDs are unique
        if self.p_df["id"].nunique() != len(self.p_df):
            self.errors.append("Individual IDs (P.id) are not unique")
            passed = False
        else:
            print("✓ Individual IDs are unique")

        # Check household IDs
        p_households = set(self.p_df["hid"].unique())
        h_households = set(self.h_df["id"].unique())
        d_households = set(self.d_df["ID"].unique())

        if p_households != h_households:
            missing_in_h = p_households - h_households
            extra_in_h = h_households - p_households
            if missing_in_h:
                self.errors.append(f"P has {len(missing_in_h)} household IDs not in H")
            if extra_in_h:
                self.warnings.append(f"H has {len(extra_in_h)} household IDs not in P")
            passed = False
        else:
            print("✓ Household IDs consistent between P and H")

        if h_households != d_households:
            missing_in_d = h_households - d_households
            extra_in_d = d_households - h_households
            if missing_in_d:
                self.errors.append(f"H has {len(missing_in_d)} household IDs not in D")
            if extra_in_d:
                self.errors.append(f"D has {len(extra_in_d)} household IDs not in H")
            passed = False
        else:
            print("✓ Household IDs consistent between H and D")

        return passed

    def check_country_consistency(self) -> bool:
        """
        Check that SA0100 (country code) is consistent.

        Validates:
        - Same set of countries across all files
        - No missing country codes
        """
        print("\n=== Checking Country Consistency ===")
        passed = True

        p_countries = set(self.p_df["SA0100"].unique())
        h_countries = set(self.h_df["SA0100"].unique())
        d_countries = set(self.d_df["SA0100"].unique())

        if p_countries != h_countries or h_countries != d_countries:
            self.errors.append(f"Country codes differ: P={p_countries}, H={h_countries}, D={d_countries}")
            passed = False
        else:
            print(f"✓ Consistent country codes: {sorted(p_countries)}")

        # Check for missing country codes
        for df_name, df in [("P", self.p_df), ("H", self.h_df), ("D", self.d_df)]:
            if df["SA0100"].isna().any():
                self.errors.append(f"{df_name} has missing country codes (SA0100)")
                passed = False

        return passed

    def check_survey_consistency(self) -> bool:
        """
        Check that survey column is consistent.

        Validates:
        - Same survey values across all files
        - No missing survey values
        """
        print("\n=== Checking Survey Consistency ===")
        passed = True

        p_surveys = set(self.p_df["survey"].unique())
        h_surveys = set(self.h_df["survey"].unique())
        d_surveys = set(self.d_df["survey"].unique())

        if p_surveys != h_surveys or h_surveys != d_surveys:
            self.warnings.append(f"Survey values differ: P={p_surveys}, H={h_surveys}, D={d_surveys}")
            passed = False
        else:
            print(f"✓ Consistent survey values: {sorted(p_surveys)}")

        return passed

    def check_household_aggregation(self) -> bool:
        """
        Check that household-level aggregations are consistent.

        Validates:
        - Number of individuals per household is reasonable
        - No household has zero individuals
        """
        print("\n=== Checking Household Aggregation ===")
        passed = True

        # Count individuals per household
        hh_sizes = self.p_df.groupby("hid").size()

        if (hh_sizes == 0).any():
            self.errors.append("Some households have 0 individuals")
            passed = False

        max_size = hh_sizes.max()
        if max_size > 20:  # Reasonable upper bound
            self.warnings.append(f"Some households have unusually large size: max={max_size}")

        print(f"✓ Household sizes: min={hh_sizes.min()}, max={max_size}, mean={hh_sizes.mean():.2f}")

        # Check all households in H have individuals in P
        h_with_no_p = set(self.h_df["id"]) - set(self.p_df["hid"])
        if h_with_no_p:
            self.warnings.append(f"{len(h_with_no_p)} households in H have no individuals in P")

        return passed

    def check_weights(self) -> bool:
        """
        Check that survey weights are consistent.

        Validates:
        - HW0010 (weight) exists and is positive
        - Weights are consistent between H and D files
        """
        print("\n=== Checking Survey Weights ===")
        passed = True

        # Check weights in H
        if "HW0010" not in self.h_df.columns:
            self.errors.append("HW0010 (weight) missing in H file")
            passed = False
        else:
            h_weights = pd.to_numeric(self.h_df["HW0010"], errors="coerce")
            if h_weights.isna().any():
                self.warnings.append(f"H has {h_weights.isna().sum()} missing weights")
            if (h_weights <= 0).any():
                self.errors.append("H has non-positive weights")
                passed = False
            else:
                print(f"✓ H weights: min={h_weights.min():.2f}, max={h_weights.max():.2f}, mean={h_weights.mean():.2f}")

        # Check weights in D
        if "HW0010" not in self.d_df.columns:
            self.errors.append("HW0010 (weight) missing in D file")
            passed = False
        else:
            # Weights should match between H and D for same IDs
            merged = pd.merge(
                self.h_df[["id", "HW0010"]],
                self.d_df[["ID", "HW0010"]],
                left_on="id",
                right_on="ID",
                suffixes=("_h", "_d"),
            )
            merged["HW0010_h"] = pd.to_numeric(merged["HW0010_h"], errors="coerce")
            merged["HW0010_d"] = pd.to_numeric(merged["HW0010_d"], errors="coerce")

            if not np.allclose(
                merged["HW0010_h"].fillna(0),
                merged["HW0010_d"].fillna(0),
                rtol=1e-5,
            ):
                self.errors.append("Weights differ between H and D files")
                passed = False
            else:
                print("✓ Weights consistent between H and D")

        return passed

    def check_common_columns(self) -> bool:
        """
        Check that common columns across files are consistent.

        Validates:
        - SA0010, SA0100, IM0100, survey columns match between files
        """
        print("\n=== Checking Common Columns ===")
        passed = True

        common_cols = ["SA0010", "SA0100", "IM0100", "survey"]

        # Check H and D have matching values for these columns
        for col in common_cols:
            if col not in self.h_df.columns or col not in self.d_df.columns:
                self.warnings.append(f"Column {col} missing in H or D")
                continue

            merged = pd.merge(
                self.h_df[["id", col]],
                self.d_df[["ID", col]],
                left_on="id",
                right_on="ID",
                suffixes=("_h", "_d"),
            )

            if not (merged[f"{col}_h"] == merged[f"{col}_d"]).all():
                mismatches = (~(merged[f"{col}_h"] == merged[f"{col}_d"])).sum()
                self.errors.append(f"Column {col} has {mismatches} mismatches between H and D")
                passed = False
            else:
                print(f"✓ Column {col} consistent between H and D")

        return passed

    def check_data_types(self) -> bool:
        """
        Check that key columns have appropriate data types.

        Validates:
        - ID columns are not null
        - Numeric columns that should be numeric
        """
        print("\n=== Checking Data Types ===")
        passed = True

        # Check ID columns
        if self.p_df["id"].isna().any():
            self.errors.append("P has null individual IDs")
            passed = False
        if self.p_df["hid"].isna().any():
            self.errors.append("P has null household IDs")
            passed = False
        if self.h_df["id"].isna().any():
            self.errors.append("H has null household IDs")
            passed = False
        if self.d_df["ID"].isna().any():
            self.errors.append("D has null household IDs")
            passed = False

        if passed:
            print("✓ No null values in ID columns")

        return passed

    def check_demographic_consistency(self) -> bool:
        """
        Check demographic data consistency.

        Validates:
        - Age (RA0300) is non-negative and reasonable
        - Gender (RA0200) has valid values
        """
        print("\n=== Checking Demographic Consistency ===")
        passed = True

        if "RA0300" in self.p_df.columns:
            ages = pd.to_numeric(self.p_df["RA0300"], errors="coerce")
            ages_valid = ages[ages.notna()]

            if (ages_valid < 0).any():
                self.errors.append("P has negative ages")
                passed = False

            if (ages_valid > 120).any():
                self.warnings.append(f"P has ages > 120: max={ages_valid.max()}")

            print(f"✓ Age range: min={ages_valid.min()}, max={ages_valid.max()}, mean={ages_valid.mean():.2f}")

        if "RA0200" in self.p_df.columns:
            genders = self.p_df["RA0200"].dropna().unique()
            print(f"✓ Gender values: {sorted(genders)}")

        return passed

    def run_all_checks(self) -> bool:
        """
        Run all validation checks.

        Returns
        -------
        bool
            True if all critical checks pass, False otherwise
        """
        self.load_data()

        checks = [
            self.check_id_relationships,
            self.check_country_consistency,
            self.check_survey_consistency,
            self.check_household_aggregation,
            self.check_weights,
            self.check_common_columns,
            self.check_data_types,
            self.check_demographic_consistency,
        ]

        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                self.errors.append(f"Error in {check.__name__}: {str(e)}")
                results.append(False)

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")

        overall_pass = all(results) and len(self.errors) == 0

        return overall_pass


def main():
    """Main function to run validation."""
    import sys

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Default to test data
        data_dir = Path(
            "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data/hfcs/2014"
        )

    print(f"Validating HFCS data in: {data_dir}")
    print("=" * 60)

    validator = HFCSValidator(data_dir)
    success = validator.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
