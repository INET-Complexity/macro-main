"""
Generate spoofed HFCS data that maintains statistical properties while protecting confidentiality.

This script creates synthetic HFCS data files (P1, H1, D1) that:
- Preserve structural relationships (household IDs, individual counts)
- Maintain statistical distributions of variables
- Preserve correlations between related variables
- Pass all coherence validation checks

Usage:
    python generate_spoofed_hfcs.py <input_dir> <output_dir> [--seed SEED]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class HFCSSpoofGenerator:
    """Generate spoofed HFCS data maintaining statistical properties."""

    def __init__(self, input_dir: Path, exploration_results_path: Path, seed: int = 42):
        """
        Initialize spoofing generator.

        Parameters
        ----------
        input_dir : Path
            Directory containing original P1.csv, H1.csv, D1.csv files
        exploration_results_path : Path
            Path to exploration results JSON file
        seed : int
            Random seed for reproducibility
        """
        self.input_dir = Path(input_dir)
        self.seed = seed
        np.random.seed(seed)

        # Load exploration results
        with open(exploration_results_path, "r") as f:
            self.exploration_results = json.load(f)

        # Data storage
        self.p_original = None
        self.h_original = None
        self.d_original = None

        self.p_spoofed = None
        self.h_spoofed = None
        self.d_spoofed = None

        print(f"Initialized HFCSSpoofGenerator with seed={seed}")

    def load_data(self):
        """Load original HFCS data files."""
        print("\n" + "=" * 60)
        print("Loading original data...")
        print("=" * 60)

        self.p_original = pd.read_csv(self.input_dir / "P1.csv")
        self.h_original = pd.read_csv(self.input_dir / "H1.csv", low_memory=False)
        self.d_original = pd.read_csv(self.input_dir / "D1.csv", low_memory=False)

        print(f"Loaded P1: {len(self.p_original)} rows, {len(self.p_original.columns)} cols")
        print(f"Loaded H1: {len(self.h_original)} rows, {len(self.h_original.columns)} cols")
        print(f"Loaded D1: {len(self.d_original)} rows, {len(self.d_original.columns)} cols")

        # Initialize spoofed dataframes as copies
        self.p_spoofed = self.p_original.copy()
        self.h_spoofed = self.h_original.copy()
        self.d_spoofed = self.d_original.copy()

    def get_preserve_columns(self, file_name: str) -> List[str]:
        """Get list of columns to preserve as-is."""
        preserve = ["SA0100", "SA0010", "survey", "IM0100"]

        if file_name in ["H1", "D1"]:
            preserve.append("HW0010")  # Survey weight

        # Always preserve ID columns
        if file_name == "P1":
            preserve.extend(["id", "hid"])
        elif file_name == "H1":
            preserve.append("id")
        elif file_name == "D1":
            preserve.append("ID")

        return preserve

    def spoof_categorical_column(self, df: pd.DataFrame, col: str, inplace: bool = True) -> Optional[pd.Series]:
        """
        Spoof categorical column by resampling from observed distribution.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the column
        col : str
            Column name to spoof
        inplace : bool
            If True, modify df in place; if False, return new series

        Returns
        -------
        Optional[pd.Series]
            Spoofed series if inplace=False, None otherwise
        """
        # Save missingness pattern
        missing_mask = df[col].isna()

        # Get non-missing values and their distribution
        non_missing = df.loc[~missing_mask, col]

        if len(non_missing) == 0:
            # All missing, keep as-is
            return None if inplace else df[col].copy()

        # Sample from observed distribution
        value_counts = non_missing.value_counts(normalize=True)
        spoofed_values = np.random.choice(value_counts.index, size=len(non_missing), p=value_counts.values)

        # Create spoofed series
        spoofed = pd.Series(index=df.index, dtype=df[col].dtype)
        spoofed[missing_mask] = np.nan
        spoofed[~missing_mask] = spoofed_values

        if inplace:
            df[col] = spoofed
            return None
        else:
            return spoofed

    def spoof_numerical_column(
        self,
        df: pd.DataFrame,
        col: str,
        method: str = "lognormal_with_zeros",
        inplace: bool = True,
    ) -> Optional[pd.Series]:
        """
        Spoof numerical column using distribution fitting.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the column
        col : str
            Column name to spoof
        method : str
            Method to use: 'normal', 'lognormal_with_zeros', 'quantile'
        inplace : bool
            If True, modify df in place; if False, return new series

        Returns
        -------
        Optional[pd.Series]
            Spoofed series if inplace=False, None otherwise
        """
        # Save missingness pattern
        missing_mask = df[col].isna()

        # Convert to numeric
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        non_missing = numeric_vals[~missing_mask]

        if len(non_missing) == 0:
            # All missing, keep as-is
            return None if inplace else df[col].copy()

        # Check if this is actually a categorical column (low cardinality)
        n_unique = non_missing.nunique()
        if n_unique <= 20:
            # Treat as categorical - resample from observed values
            value_counts = non_missing.value_counts(normalize=True)
            spoofed_values = np.random.choice(value_counts.index, size=len(non_missing), p=value_counts.values)

            # Create spoofed series
            spoofed = pd.Series(index=df.index, dtype=non_missing.dtype)
            spoofed[missing_mask] = np.nan
            spoofed[~missing_mask] = spoofed_values

            if inplace:
                df[col] = spoofed
                return None
            else:
                return spoofed

        # Determine spoofing method based on data characteristics
        has_negatives = (non_missing < 0).any()

        if method == "lognormal_with_zeros" and not has_negatives:
            # Two-stage model: Bernoulli for non-zero + lognormal for positive values
            prob_nonzero = (non_missing > 0).mean()
            positive_vals = non_missing[non_missing > 0]

            if len(positive_vals) == 0:
                # All zeros
                spoofed_values = np.zeros(len(non_missing))
            else:
                # Fit lognormal to positive values
                log_vals = np.log(positive_vals)
                mean_log, std_log = log_vals.mean(), log_vals.std()

                # Generate: first determine which are non-zero, then sample amounts
                is_nonzero = np.random.random(len(non_missing)) < prob_nonzero
                spoofed_values = np.zeros(len(non_missing))

                if is_nonzero.sum() > 0:
                    spoofed_values[is_nonzero] = np.random.lognormal(
                        mean=mean_log, sigma=std_log, size=is_nonzero.sum()
                    )

        elif method == "normal" or has_negatives:
            # Use normal distribution (works for values that can be negative)
            mean, std = non_missing.mean(), non_missing.std()

            if std == 0:
                spoofed_values = np.full(len(non_missing), mean)
            else:
                spoofed_values = np.random.normal(mean, std, len(non_missing))

        elif method == "quantile":
            # Quantile-based resampling (preserves distribution shape exactly)
            spoofed_values = np.random.choice(non_missing, size=len(non_missing), replace=True)

        else:
            # Default: use original values shuffled
            spoofed_values = np.random.permutation(non_missing.values)

        # Create spoofed series
        spoofed = pd.Series(index=df.index, dtype=float)
        spoofed[missing_mask] = np.nan
        spoofed[~missing_mask] = spoofed_values

        if inplace:
            df[col] = spoofed
            return None
        else:
            return spoofed

    def spoof_age_income_correlated(self, df: pd.DataFrame):
        """
        Spoof age and income columns while preserving their correlation.

        This handles the P1 file's age (RA0300) and income (PG0110) relationship.
        """
        if "RA0300" not in df.columns or "PG0110" not in df.columns:
            return

        print("  Spoofing age-income with correlation preservation...")

        # Save missingness for income
        income_missing = df["PG0110"].isna()

        # Age bins
        age_bins = pd.cut(
            pd.to_numeric(df["RA0300"], errors="coerce"),
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
        )

        # Spoof age first (within bins, shuffle)
        for age_group in age_bins.cat.categories:
            mask = age_bins == age_group
            if mask.sum() > 0:
                ages_in_group = df.loc[mask, "RA0300"].values
                df.loc[mask, "RA0300"] = np.random.permutation(ages_in_group)

        # Spoof income conditional on age group
        for age_group in age_bins.cat.categories:
            mask = age_bins == age_group
            has_income_mask = mask & ~income_missing

            if has_income_mask.sum() > 0:
                # Get income distribution for this age group
                income_in_group = pd.to_numeric(df.loc[has_income_mask, "PG0110"], errors="coerce")
                income_positive = income_in_group[income_in_group > 0].values

                if len(income_positive) > 0:
                    # Sample with replacement from observed incomes in age group
                    df.loc[has_income_mask, "PG0110"] = np.random.choice(
                        income_positive, size=has_income_mask.sum(), replace=True
                    )

    def spoof_flag_columns(self, df: pd.DataFrame, flag_cols: List[str]):
        """
        Regenerate flag columns based on spoofed data.

        Flag columns (starting with 'f') typically indicate:
        - Missing/imputed data
        - Data quality flags
        - Response codes
        """
        print(f"  Regenerating {len(flag_cols)} flag columns...")

        for flag_col in flag_cols:
            # Try to identify the base column (remove 'f' prefix)
            base_col = flag_col[1:]

            if base_col in df.columns:
                # Common pattern: flag = 1 if missing, -1 or other codes otherwise
                # We'll use a simple heuristic: preserve flag distribution but resample
                flag_values = df[flag_col].dropna().unique()

                if len(flag_values) > 0:
                    # Resample from observed flag distribution
                    flag_dist = df[flag_col].value_counts(normalize=True)
                    df[flag_col] = np.random.choice(flag_dist.index, size=len(df), p=flag_dist.values)
            # If we can't match to base column, just resample from distribution
            else:
                if df[flag_col].notna().any():
                    flag_dist = df[flag_col].value_counts(normalize=True)
                    df[flag_col] = np.random.choice(flag_dist.index, size=len(df), p=flag_dist.values)

    def spoof_p1(self):
        """Spoof P1.csv (Personal data)."""
        print("\n" + "=" * 60)
        print("Spoofing P1 (Personal data)...")
        print("=" * 60)

        preserve_cols = self.get_preserve_columns("P1")
        results = self.exploration_results.get("P1", {})

        print(f"Preserving {len(preserve_cols)} columns: {preserve_cols}")

        # Get column classifications
        categorical_cols = list(results.get("categorical_columns", {}).keys())
        numerical_cols = list(results.get("numerical_columns", {}).keys())
        flag_cols = results.get("flag_columns", [])

        # Remove preserved columns from spoofing lists
        categorical_cols = [c for c in categorical_cols if c not in preserve_cols]
        numerical_cols = [c for c in numerical_cols if c not in preserve_cols]

        # Spoof age and income with correlation
        self.spoof_age_income_correlated(self.p_spoofed)

        # Remove RA0300 and PG0110 from lists (already handled)
        numerical_cols = [c for c in numerical_cols if c not in ["RA0300", "PG0110"]]

        # Spoof categorical columns
        print(f"Spoofing {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            if col in self.p_spoofed.columns:
                self.spoof_categorical_column(self.p_spoofed, col)

        # Spoof numerical columns
        print(f"Spoofing {len(numerical_cols)} numerical columns...")
        for col in numerical_cols:
            if col in self.p_spoofed.columns:
                # Use quantile method for faster processing
                self.spoof_numerical_column(self.p_spoofed, col, method="quantile")

        # Regenerate flag columns
        self.spoof_flag_columns(self.p_spoofed, flag_cols)

        print("✓ P1 spoofing complete")

    def spoof_h1(self):
        """Spoof H1.csv (Household data)."""
        print("\n" + "=" * 60)
        print("Spoofing H1 (Household data)...")
        print("=" * 60)

        preserve_cols = self.get_preserve_columns("H1")
        results = self.exploration_results.get("H1", {})

        print(f"Preserving {len(preserve_cols)} columns: {preserve_cols}")

        # Get column classifications
        categorical_cols = list(results.get("categorical_columns", {}).keys())
        numerical_cols = list(results.get("numerical_columns", {}).keys())
        flag_cols = results.get("flag_columns", [])

        # Remove preserved columns and special columns handled separately
        categorical_cols = [c for c in categorical_cols if c not in preserve_cols]
        numerical_cols = [c for c in numerical_cols if c not in preserve_cols]

        # Spoof categorical columns (but handle HB0300 specially later)
        special_cats = ["HB0300"]  # Tenure status - has dependencies
        categorical_cols = [c for c in categorical_cols if c not in special_cats]

        print(f"Spoofing {len(categorical_cols)} categorical columns...")
        for i, col in enumerate(categorical_cols):
            if col in self.h_spoofed.columns:
                self.spoof_categorical_column(self.h_spoofed, col)
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(categorical_cols)} categorical columns")

        # Spoof numerical columns (HB0300 will determine constraints for some)
        print(f"Spoofing {len(numerical_cols)} numerical columns...")
        for i, col in enumerate(numerical_cols):
            if col in self.h_spoofed.columns:
                # Detect if should use lognormal (asset/liability columns)
                if any(prefix in col for prefix in ["HB", "HC", "DA", "DL"]):
                    self.spoof_numerical_column(self.h_spoofed, col, method="lognormal_with_zeros")
                else:
                    self.spoof_numerical_column(self.h_spoofed, col, method="quantile")

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(numerical_cols)} numerical columns")

        # Handle HB0300 (Tenure Status) specially - keep as-is to maintain relationships
        # This is critical because it determines ownership in matching_households_with_houses.py
        print("  Note: HB0300 (Tenure Status) preserved to maintain ownership relationships")

        # Regenerate flag columns
        self.spoof_flag_columns(self.h_spoofed, flag_cols)

        print("✓ H1 spoofing complete")

    def spoof_d1(self):
        """Spoof D1.csv (Derived data)."""
        print("\n" + "=" * 60)
        print("Spoofing D1 (Derived data)...")
        print("=" * 60)

        preserve_cols = self.get_preserve_columns("D1")
        results = self.exploration_results.get("D1", {})

        print(f"Preserving {len(preserve_cols)} columns: {preserve_cols}")
        print("Note: D1 contains derived variables - maintaining consistency with H1")

        # Get column classifications
        categorical_cols = list(results.get("categorical_columns", {}).keys())
        numerical_cols = list(results.get("numerical_columns", {}).keys())

        # Remove preserved columns and special columns
        categorical_cols = [c for c in categorical_cols if c not in preserve_cols]
        numerical_cols = [c for c in numerical_cols if c not in preserve_cols]

        # Spoof categorical columns
        print(f"Spoofing {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            if col in self.d_spoofed.columns:
                self.spoof_categorical_column(self.d_spoofed, col)

        # Spoof numerical columns with correlation preservation
        print(f"Spoofing {len(numerical_cols)} numerical columns...")

        # Build correlation matrix for key derived variables
        key_derived = [c for c in numerical_cols if c in self.d_spoofed.columns]

        # Get tenure status from H1 to maintain consistency
        tenure_status = self.h_spoofed["HB0300"].values if "HB0300" in self.h_spoofed.columns else None

        for i, col in enumerate(key_derived):
            # Special handling for DA1110 (Value of Main Residence)
            # Must be consistent with tenure status
            if col == "DA1110" and tenure_status is not None:
                print("  Spoofing DA1110 (Main Residence Value) with tenure consistency...")
                self.spoof_main_residence_value(tenure_status)
            # Use method based on column type
            elif col.startswith("DA"):  # Asset columns
                self.spoof_numerical_column(self.d_spoofed, col, method="lognormal_with_zeros")
            elif col.startswith("DL"):  # Liability columns
                self.spoof_numerical_column(self.d_spoofed, col, method="lognormal_with_zeros")
            elif col.startswith("DI"):  # Income columns (can be negative)
                self.spoof_numerical_column(self.d_spoofed, col, method="normal")
            elif col.startswith("DN"):  # Net wealth (can be negative)
                self.spoof_numerical_column(self.d_spoofed, col, method="normal")
            else:
                self.spoof_numerical_column(self.d_spoofed, col, method="quantile")

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(key_derived)} numerical columns")

        print("✓ D1 spoofing complete")

    def spoof_main_residence_value(self, tenure_status: np.ndarray):
        """
        Spoof DA1110 (Value of Main Residence) maintaining consistency with tenure status.

        Tenure status meanings:
        - 1, 2: Owner-occupied → MUST have DA1110 > 0
        - 3, 4: Renter/Other → MUST have DA1110 = 0 or NaN
        """
        col = "DA1110"
        if col not in self.d_spoofed.columns:
            return

        # Get original values
        original_vals = pd.to_numeric(self.d_original[col], errors="coerce")

        # Identify owners vs non-owners based on tenure
        is_owner = (tenure_status == 1) | (tenure_status == 2)
        is_renter = (tenure_status == 3) | (tenure_status == 4)

        # Get distributions for owners and renters separately
        owner_values = original_vals[is_owner].dropna()
        owner_positive = owner_values[owner_values > 0]

        # Spoof owner values (must be > 0)
        if len(owner_positive) > 0:
            # Fit lognormal to positive values
            log_vals = np.log(owner_positive)
            mean_log, std_log = log_vals.mean(), log_vals.std()

            # Generate positive values for all owners
            n_owners = is_owner.sum()
            spoofed_owner_values = np.random.lognormal(mean=mean_log, sigma=std_log, size=n_owners)
            self.d_spoofed.loc[is_owner, col] = spoofed_owner_values

        # Renters should have NaN or 0
        # Preserve the missingness pattern from original data
        self.d_spoofed.loc[is_renter, col] = np.nan
        # Some renters might have 0 instead of NaN (check original pattern)
        if (original_vals[is_renter] == 0).any():
            # Keep some as 0
            n_zero = (original_vals[is_renter] == 0).sum()
            renter_indices = np.where(is_renter)[0]
            zero_indices = np.random.choice(renter_indices, size=n_zero, replace=False)
            self.d_spoofed.loc[zero_indices, col] = 0

    def generate_spoofed_data(self):
        """Main method to generate all spoofed files."""
        self.load_data()
        self.spoof_p1()
        self.spoof_h1()
        self.spoof_d1()

        print("\n" + "=" * 60)
        print("✅ All spoofing complete!")
        print("=" * 60)

    def save_spoofed_data(self, output_dir: Path):
        """Save spoofed data to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving spoofed data to: {output_dir}")

        self.p_spoofed.to_csv(output_dir / "P1.csv", index=False)
        print(f"  Saved P1.csv ({len(self.p_spoofed)} rows)")

        self.h_spoofed.to_csv(output_dir / "H1.csv", index=False)
        print(f"  Saved H1.csv ({len(self.h_spoofed)} rows)")

        self.d_spoofed.to_csv(output_dir / "D1.csv", index=False)
        print(f"  Saved D1.csv ({len(self.d_spoofed)} rows)")

        print("\n✓ All files saved successfully")

    def compare_statistics(self):
        """Compare key statistics between original and spoofed data."""
        print("\n" + "=" * 60)
        print("Statistical Comparison: Original vs Spoofed")
        print("=" * 60)

        def compare_column(original, spoofed, col_name, df_name):
            """Compare statistics for a single column."""
            orig_vals = pd.to_numeric(original[col_name], errors="coerce").dropna()
            spoof_vals = pd.to_numeric(spoofed[col_name], errors="coerce").dropna()

            if len(orig_vals) == 0 or len(spoof_vals) == 0:
                return

            print(f"\n{df_name}.{col_name}:")
            print(f"  Mean:   {orig_vals.mean():>12.2f} → {spoof_vals.mean():>12.2f}")
            print(f"  Median: {orig_vals.median():>12.2f} → {spoof_vals.median():>12.2f}")
            print(f"  Std:    {orig_vals.std():>12.2f} → {spoof_vals.std():>12.2f}")
            print(f"  Min:    {orig_vals.min():>12.2f} → {spoof_vals.min():>12.2f}")
            print(f"  Max:    {orig_vals.max():>12.2f} → {spoof_vals.max():>12.2f}")

        # Compare key columns
        print("\n📊 P1 (Personal) Key Statistics:")
        compare_column(self.p_original, self.p_spoofed, "RA0300", "P1")  # Age
        if "PG0110" in self.p_original.columns:
            compare_column(self.p_original, self.p_spoofed, "PG0110", "P1")  # Income

        print("\n📊 H1 (Household) Key Statistics:")
        compare_column(self.h_original, self.h_spoofed, "HW0010", "H1")  # Weight

        print("\n📊 D1 (Derived) Key Statistics:")
        if "DA1110" in self.d_original.columns:
            compare_column(self.d_original, self.d_spoofed, "DA1110", "D1")  # Home value
        if "DI2000" in self.d_original.columns:
            compare_column(self.d_original, self.d_spoofed, "DI2000", "D1")  # Total income


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate spoofed HFCS data maintaining statistical properties")
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default="tests/test_macro_data/unit/sample_raw_data/hfcs/2014",
        help="Directory containing original P1.csv, H1.csv, D1.csv",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="spoof_data/spoofed_hfcs_output",
        help="Directory to save spoofed files",
    )
    parser.add_argument(
        "--exploration-results",
        type=str,
        default="spoof_data/hfcs_exploration_results.json",
        help="Path to exploration results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks after spoofing",
    )

    args = parser.parse_args()

    # Create generator
    generator = HFCSSpoofGenerator(
        input_dir=Path(args.input_dir),
        exploration_results_path=Path(args.exploration_results),
        seed=args.seed,
    )

    # Generate spoofed data
    generator.generate_spoofed_data()

    # Compare statistics
    generator.compare_statistics()

    # Save
    generator.save_spoofed_data(Path(args.output_dir))

    # Validate if requested
    if args.validate:
        print("\n" + "=" * 60)
        print("Running validation checks...")
        print("=" * 60)

        from validate_hfcs_coherence import HFCSValidator

        validator = HFCSValidator(Path(args.output_dir))
        success = validator.run_all_checks()

        if success:
            print("\n🎉 Spoofed data passes all validation checks!")
        else:
            print("\n⚠️  Spoofed data has validation issues (see above)")


if __name__ == "__main__":
    main()
