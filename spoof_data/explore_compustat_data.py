"""
Explore Compustat data structure to understand what needs to be spoofed.

This script analyzes the three Compustat files:
- firms_annual.csv
- firms_quarterly.csv
- banks.csv

It examines column types, distributions, relationships between files,
and generates a spoofing strategy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_dataframe(df: pd.DataFrame, name: str) -> dict:
    """Analyze a single dataframe and return summary statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    results = {
        "name": name,
        "shape": df.shape,
        "columns": {},
    }

    for col in df.columns:
        col_data = df[col]
        col_info = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isna().sum()),
            "null_pct": float(col_data.isna().sum() / len(col_data) * 100),
            "unique_count": int(col_data.nunique()),
        }

        # Analyze based on type
        if pd.api.types.is_numeric_dtype(col_data):
            non_null = col_data.dropna()
            if len(non_null) > 0:
                col_info["type"] = "numeric"
                col_info["min"] = float(non_null.min())
                col_info["max"] = float(non_null.max())
                col_info["mean"] = float(non_null.mean())
                col_info["median"] = float(non_null.median())
                col_info["std"] = float(non_null.std())

                # Check if it's actually categorical (low cardinality)
                if col_info["unique_count"] <= 20:
                    col_info["is_categorical"] = True
                    col_info["unique_values"] = sorted(non_null.unique().tolist())
                    value_counts = col_data.value_counts()
                    col_info["value_distribution"] = {str(k): int(v) for k, v in value_counts.items()}
        else:
            col_info["type"] = "categorical"
            value_counts = col_data.value_counts()
            col_info["top_values"] = {str(k): int(v) for k, v in value_counts.head(10).items()}

        results["columns"][col] = col_info

        # Print summary
        print(f"\n{col}:")
        print(f"  Type: {col_info.get('type', 'categorical')}")
        print(f"  Nulls: {col_info['null_count']} ({col_info['null_pct']:.1f}%)")
        print(f"  Unique: {col_info['unique_count']}")

        if col_info.get("type") == "numeric":
            if col_info.get("is_categorical"):
                print(f"  → Actually categorical (≤20 unique values)")
                print(f"  Unique values: {col_info['unique_values']}")
            else:
                print(f"  Range: [{col_info['min']:.2f}, {col_info['max']:.2f}]")
                print(f"  Mean: {col_info['mean']:.2f}, Median: {col_info['median']:.2f}")
        elif "top_values" in col_info:
            print(f"  Top values: {dict(list(col_info['top_values'].items())[:3])}")

    return results


def analyze_relationships(firms_annual: pd.DataFrame, firms_quarterly: pd.DataFrame, banks: pd.DataFrame) -> dict:
    """Analyze relationships between the three dataframes."""
    print(f"\n{'='*60}")
    print("Analyzing Relationships")
    print(f"{'='*60}")

    relationships = {}

    # Check gvkey overlap
    annual_gvkeys = set(firms_annual["gvkey"].dropna().unique())
    quarterly_gvkeys = set(firms_quarterly["gvkey"].dropna().unique())
    bank_gvkeys = set(banks["gvkey"].dropna().unique())

    print(f"\ngvkey (company identifier) analysis:")
    print(f"  Annual firms: {len(annual_gvkeys)} unique companies")
    print(f"  Quarterly firms: {len(quarterly_gvkeys)} unique companies")
    print(f"  Banks: {len(bank_gvkeys)} unique companies")
    print(f"  Annual ∩ Quarterly: {len(annual_gvkeys & quarterly_gvkeys)} companies")
    print(f"  Banks ∩ Firms: {len(bank_gvkeys & (annual_gvkeys | quarterly_gvkeys))} companies")

    relationships["gvkey_overlap"] = {
        "annual_unique": len(annual_gvkeys),
        "quarterly_unique": len(quarterly_gvkeys),
        "bank_unique": len(bank_gvkeys),
        "annual_quarterly_overlap": len(annual_gvkeys & quarterly_gvkeys),
        "banks_firms_overlap": len(bank_gvkeys & (annual_gvkeys | quarterly_gvkeys)),
    }

    # Check conm (company name) usage
    if "conm" in firms_annual.columns and "conm" in firms_quarterly.columns:
        annual_conm = set(firms_annual["conm"].dropna().unique())
        quarterly_conm = set(firms_quarterly["conm"].dropna().unique())
        bank_conm = set(banks["conm"].dropna().unique()) if "conm" in banks.columns else set()

        print(f"\nconm (company name) analysis:")
        print(f"  Annual: {len(annual_conm)} unique names")
        print(f"  Quarterly: {len(quarterly_conm)} unique names")
        print(f"  Banks: {len(bank_conm)} unique names")
        print(f"  Annual ∩ Quarterly: {len(annual_conm & quarterly_conm)} names")

        # The code merges on conm, so this is critical
        print(f"\n  ⚠️  CRITICAL: Firms are merged on 'conm' (company name)")
        print(f"  Companies in both annual and quarterly: {len(annual_conm & quarterly_conm)}")

        relationships["conm_overlap"] = {
            "annual_unique": len(annual_conm),
            "quarterly_unique": len(quarterly_conm),
            "bank_unique": len(bank_conm),
            "annual_quarterly_overlap": len(annual_conm & quarterly_conm),
        }

    # Check country distribution
    annual_countries = firms_annual["loc"].value_counts()
    quarterly_countries = firms_quarterly["loc"].value_counts()
    bank_countries = banks["loc"].value_counts()

    print(f"\nCountry distribution:")
    print(f"  Annual:\n{annual_countries}")
    print(f"  Quarterly:\n{quarterly_countries}")
    print(f"  Banks:\n{bank_countries}")

    relationships["country_distribution"] = {
        "annual": {str(k): int(v) for k, v in annual_countries.items()},
        "quarterly": {str(k): int(v) for k, v in quarterly_countries.items()},
        "banks": {str(k): int(v) for k, v in bank_countries.items()},
    }

    return relationships


def generate_spoofing_strategy(analysis_results: dict) -> dict:
    """Generate a spoofing strategy based on the analysis."""
    print(f"\n{'='*60}")
    print("Generating Spoofing Strategy")
    print(f"{'='*60}")

    strategy = {
        "overview": (
            "Compustat data contains confidential company financial information. "
            "Key identifiers like gvkey (company ID), conm (company name), and "
            "tic (ticker symbol) should be anonymized or removed. "
            "Financial metrics should be spoofed while maintaining distributions."
        ),
        "critical_constraints": [],
        "columns_to_remove": [],
        "columns_to_spoof": {},
    }

    # Analyze each file
    for file_result in analysis_results["files"]:
        file_name = file_result["name"]
        print(f"\n{file_name}:")

        for col_name, col_info in file_result["columns"].items():
            # Identify columns to remove (direct identifiers)
            if col_name in ["gvkey", "tic", "conm", "conml"]:
                print(f"  ⚠️  {col_name}: REMOVE (direct company identifier)")
                strategy["columns_to_remove"].append(f"{file_name}:{col_name}")

            # Categorical columns - resample from distribution
            elif col_info.get("type") == "categorical" or col_info.get("is_categorical"):
                print(f"  ✓ {col_name}: Categorical resampling")
                strategy["columns_to_spoof"][f"{file_name}:{col_name}"] = "categorical_resample"

            # Numeric columns - fit distribution
            elif col_info.get("type") == "numeric":
                # Check if values are strictly positive (use lognormal)
                if col_info["min"] >= 0:
                    print(f"  ✓ {col_name}: Lognormal distribution (positive values)")
                    strategy["columns_to_spoof"][f"{file_name}:{col_name}"] = "lognormal"
                else:
                    print(f"  ✓ {col_name}: Normal distribution (can be negative)")
                    strategy["columns_to_spoof"][f"{file_name}:{col_name}"] = "normal"

    # Add critical constraints
    strategy["critical_constraints"].extend(
        [
            "conm must be consistent between firms_annual and firms_quarterly (used for merging)",
            "gvkey should be unique per company within each file",
            "loc (country) should be preserved or consistently spoofed",
            "fyear/fyearq and fqtr should be preserved (time identifiers)",
            "Currency codes (curcdq) should match country locations",
        ]
    )

    print("\n" + "=" * 60)
    print("Key Constraints:")
    for constraint in strategy["critical_constraints"]:
        print(f"  • {constraint}")

    return strategy


def main():
    """Main exploration function."""
    # Paths
    data_dir = Path("tests/test_macro_data/unit/sample_raw_data/compustat")

    print("=" * 60)
    print("Compustat Data Exploration")
    print("=" * 60)

    # Load data
    firms_annual = pd.read_csv(data_dir / "firms_annual.csv")
    firms_quarterly = pd.read_csv(data_dir / "firms_quarterly.csv")
    banks = pd.read_csv(data_dir / "banks.csv")

    # Analyze each file
    results = {
        "files": [
            analyze_dataframe(firms_annual, "firms_annual.csv"),
            analyze_dataframe(firms_quarterly, "firms_quarterly.csv"),
            analyze_dataframe(banks, "banks.csv"),
        ],
        "relationships": analyze_relationships(firms_annual, firms_quarterly, banks),
    }

    # Generate spoofing strategy
    strategy = generate_spoofing_strategy(results)
    results["spoofing_strategy"] = strategy

    # Save results
    output_path = Path("spoof_data/compustat_exploration_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
