"""
Data exploration script for HFCS files.

This script analyzes the HFCS data to understand:
- Column types and distributions
- Value ranges for numerical columns
- Categorical variable values
- Relationships between variables

This information guides the spoofing strategy.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


class HFCSExplorer:
    """Explores HFCS data to understand structure and distributions."""

    def __init__(self, data_dir: Path):
        """
        Initialize explorer with data directory.

        Parameters
        ----------
        data_dir : Path
            Directory containing P1.csv, H1.csv, and D1.csv files
        """
        self.data_dir = Path(data_dir)
        self.p_df = None
        self.h_df = None
        self.d_df = None
        self.exploration_results = {}

    def load_data(self):
        """Load the three HFCS data files."""
        print("Loading HFCS data files...")
        self.p_df = pd.read_csv(self.data_dir / "P1.csv")
        self.h_df = pd.read_csv(self.data_dir / "H1.csv", low_memory=False)
        self.d_df = pd.read_csv(self.data_dir / "D1.csv", low_memory=False)
        print(f"Loaded P1: {len(self.p_df)} rows, H1: {len(self.h_df)} rows, D1: {len(self.d_df)} rows\n")

    def analyze_column_types(self, df: pd.DataFrame, file_name: str):
        """
        Analyze and categorize column types.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze
        file_name : str
            Name of the file (P1, H1, or D1)
        """
        print(f"\n{'=' * 60}")
        print(f"Analyzing {file_name}")
        print("=" * 60)

        results = {
            "total_columns": len(df.columns),
            "id_columns": [],
            "categorical_columns": {},
            "numerical_columns": {},
            "high_cardinality_columns": [],
            "flag_columns": [],  # Columns starting with 'f' (flags)
            "columns_with_missing": {},
        }

        for col in df.columns:
            # Identify ID columns
            if col.lower() in ["id", "hid", "iid"]:
                results["id_columns"].append(col)
                continue

            # Identify flag columns (columns starting with 'f')
            if col.startswith("f") and col[1:] in df.columns:
                results["flag_columns"].append(col)
                continue

            # Check for missing values
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 0:
                results["columns_with_missing"][col] = round(missing_pct, 2)

            # Try to determine if numerical or categorical
            try:
                # Attempt to convert to numeric
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                non_null_numeric = numeric_vals.dropna()

                # If most values can be converted to numeric, treat as numerical
                if len(non_null_numeric) > 0.5 * len(df[col].dropna()):
                    # Store statistics
                    results["numerical_columns"][col] = {
                        "min": float(non_null_numeric.min()),
                        "max": float(non_null_numeric.max()),
                        "mean": float(non_null_numeric.mean()),
                        "median": float(non_null_numeric.median()),
                        "std": float(non_null_numeric.std()),
                        "n_unique": int(df[col].nunique()),
                        "has_negatives": bool((non_null_numeric < 0).any()),
                        "has_zeros": bool((non_null_numeric == 0).any()),
                        "missing_pct": round(missing_pct, 2),
                    }
                else:
                    # Categorical
                    unique_vals = df[col].dropna().unique()
                    n_unique = len(unique_vals)

                    if n_unique <= 20:  # Low cardinality
                        results["categorical_columns"][col] = {
                            "n_unique": n_unique,
                            "values": sorted([str(v) for v in unique_vals]),
                            "value_counts": df[col].value_counts().to_dict(),
                            "missing_pct": round(missing_pct, 2),
                        }
                    else:  # High cardinality
                        results["high_cardinality_columns"].append(
                            {
                                "column": col,
                                "n_unique": n_unique,
                                "missing_pct": round(missing_pct, 2),
                            }
                        )
            except Exception as e:
                # If can't process, note it
                print(f"Warning: Could not process column {col}: {e}")

        self.exploration_results[file_name] = results
        return results

    def print_summary(self, results: Dict, file_name: str):
        """Print summary of exploration results."""
        print(f"\n📊 Summary for {file_name}:")
        print(f"  Total columns: {results['total_columns']}")
        print(f"  ID columns: {len(results['id_columns'])} - {results['id_columns']}")
        print(f"  Categorical columns: {len(results['categorical_columns'])}")
        print(f"  Numerical columns: {len(results['numerical_columns'])}")
        print(f"  High cardinality columns: {len(results['high_cardinality_columns'])}")
        print(f"  Flag columns: {len(results['flag_columns'])}")
        print(f"  Columns with missing data: {len(results['columns_with_missing'])}")

    def print_key_columns(self, results: Dict, file_name: str):
        """Print details about key columns."""
        print(f"\n🔍 Key Categorical Columns in {file_name}:")

        # Show a few important categorical columns
        important_cats = [
            "SA0100",
            "SA0010",
            "survey",
            "RA0200",
            "DHHTYPE",
            "HB0300",
        ]
        for col in important_cats:
            if col in results["categorical_columns"]:
                info = results["categorical_columns"][col]
                print(f"\n  {col}:")
                print(f"    Unique values: {info['n_unique']}")
                print(f"    Values: {info['values']}")
                if info["missing_pct"] > 0:
                    print(f"    Missing: {info['missing_pct']}%")

        print(f"\n📈 Key Numerical Columns in {file_name}:")

        # Show a few important numerical columns
        important_nums = [
            "RA0300",
            "HW0010",
            "PG0110",
            "DA1110",
            "DI2000",
        ]  # Age, Weight, Employee Income, Home Value, Total Income
        for col in important_nums:
            if col in results["numerical_columns"]:
                info = results["numerical_columns"][col]
                print(f"\n  {col}:")
                print(f"    Range: [{info['min']:.2f}, {info['max']:.2f}]")
                print(f"    Mean: {info['mean']:.2f}, Median: {info['median']:.2f}")
                print(f"    Std: {info['std']:.2f}")
                print(f"    Has negatives: {info['has_negatives']}, Has zeros: {info['has_zeros']}")
                print(f"    Unique values: {info['n_unique']}")
                if info["missing_pct"] > 0:
                    print(f"    Missing: {info['missing_pct']}%")

    def identify_spoofing_strategy(self, results: Dict, file_name: str):
        """Suggest spoofing strategies for different column types."""
        print(f"\n💡 Spoofing Strategies for {file_name}:")

        strategies = {
            "preserve": [],  # Keep as-is (IDs, country codes)
            "permute": [],  # Randomly shuffle within column
            "sample_from_distribution": [],  # Generate from fitted distribution
            "categorical_resample": [],  # Resample from observed categories
            "linked": [],  # Must maintain relationships
        }

        # ID columns - linked
        strategies["linked"].extend(results["id_columns"])

        # Country, survey codes - preserve
        preserve_cols = ["SA0100", "SA0010", "survey", "IM0100"]
        strategies["preserve"].extend([col for col in preserve_cols if col in results["categorical_columns"]])

        # Low-cardinality categorical - resample
        for col, info in results["categorical_columns"].items():
            if col not in preserve_cols and info["n_unique"] <= 20:
                strategies["categorical_resample"].append(col)

        # Numerical columns
        for col, info in results["numerical_columns"].items():
            if col == "HW0010":  # Weight - special handling
                strategies["preserve"].append(col)
            elif info["has_negatives"] or col.startswith("DL"):  # Debt columns
                # Can't just use positive distribution
                strategies["sample_from_distribution"].append(col)
            else:
                strategies["sample_from_distribution"].append(col)

        # Flag columns - linked to their base columns
        if results["flag_columns"]:
            strategies["linked"].extend(results["flag_columns"])

        print(f"\n  Preserve as-is: {len(strategies['preserve'])} columns")
        print(f"    Examples: {strategies['preserve'][:5]}")

        print(f"\n  Permute (shuffle): {len(strategies['permute'])} columns")
        if strategies["permute"]:
            print(f"    Examples: {strategies['permute'][:5]}")

        print(f"\n  Sample from distribution: {len(strategies['sample_from_distribution'])} columns")
        print(f"    Examples: {strategies['sample_from_distribution'][:5]}")

        print(f"\n  Resample categorical: {len(strategies['categorical_resample'])} columns")
        print(f"    Examples: {strategies['categorical_resample'][:5]}")

        print(f"\n  Linked/dependent: {len(strategies['linked'])} columns (maintain relationships)")
        print(f"    Examples: {strategies['linked'][:5]}")

        self.exploration_results[f"{file_name}_strategies"] = strategies

    def check_special_relationships(self):
        """Check for special relationships that need to be maintained."""
        print(f"\n{'=' * 60}")
        print("Checking Special Relationships")
        print("=" * 60)

        # Check household size vs number of individuals
        hh_sizes = self.p_df.groupby("hid").size()
        print(f"\n📊 Household size distribution:")
        print(hh_sizes.describe())

        # Check if D columns are derived from H columns
        # For example, check if DA columns (assets) might be sums
        if "DA1000" in self.d_df.columns:  # Total real assets
            print(f"\n💰 Checking if derived columns are aggregations from household data...")
            print("  Note: D file contains derived variables that may be calculated from H and P")
            print("  Strategy: Regenerate D file after spoofing H and P")

    def save_results(self, output_path: Path):
        """Save exploration results to JSON file."""

        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.exploration_results, f, default=convert_types, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

    def run_exploration(self):
        """Run full exploration pipeline."""
        self.load_data()

        # Analyze each file
        for df, name in [
            (self.p_df, "P1"),
            (self.h_df, "H1"),
            (self.d_df, "D1"),
        ]:
            results = self.analyze_column_types(df, name)
            self.print_summary(results, name)
            self.print_key_columns(results, name)
            self.identify_spoofing_strategy(results, name)

        # Check special relationships
        self.check_special_relationships()

        # Save results
        output_path = self.data_dir.parent / "hfcs_exploration_results.json"
        self.save_results(output_path)


def main():
    """Main function to run exploration."""
    import sys

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Default to test data
        data_dir = Path("tests/test_macro_data/unit/sample_raw_data/hfcs/2014")

    print(f"Exploring HFCS data in: {data_dir}")
    print("=" * 60)

    explorer = HFCSExplorer(data_dir)
    explorer.run_exploration()


if __name__ == "__main__":
    main()
