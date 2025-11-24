"""
Compare original and spoofed HFCS data to verify quality.

This script generates a detailed comparison report showing:
- Key statistics for important variables
- Distribution comparisons
- Correlation preservation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def compare_distributions(original_dir: Path, spoofed_dir: Path):
    """Compare distributions between original and spoofed data."""
    print("=" * 80)
    print("HFCS Data Comparison: Original vs Spoofed")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    p_orig = pd.read_csv(original_dir / "P1.csv")
    h_orig = pd.read_csv(original_dir / "H1.csv", low_memory=False)
    d_orig = pd.read_csv(original_dir / "D1.csv", low_memory=False)

    p_spoof = pd.read_csv(spoofed_dir / "P1.csv")
    h_spoof = pd.read_csv(spoofed_dir / "H1.csv", low_memory=False)
    d_spoof = pd.read_csv(spoofed_dir / "D1.csv", low_memory=False)

    print(f"  Original: P={len(p_orig)}, H={len(h_orig)}, D={len(d_orig)}")
    print(f"  Spoofed:  P={len(p_spoof)}, H={len(h_spoof)}, D={len(d_spoof)}")

    def compare_col(orig_df, spoof_df, col_name, df_name):
        """Compare a single column."""
        if col_name not in orig_df.columns or col_name not in spoof_df.columns:
            return None

        orig_vals = pd.to_numeric(orig_df[col_name], errors="coerce")
        spoof_vals = pd.to_numeric(spoof_df[col_name], errors="coerce")

        orig_clean = orig_vals.dropna()
        spoof_clean = spoof_vals.dropna()

        if len(orig_clean) == 0 or len(spoof_clean) == 0:
            return None

        # Calculate statistics
        stats = {
            "column": f"{df_name}.{col_name}",
            "n_original": len(orig_clean),
            "n_spoofed": len(spoof_clean),
            "missing_pct_orig": (orig_vals.isna().sum() / len(orig_vals)) * 100,
            "missing_pct_spoof": (spoof_vals.isna().sum() / len(spoof_vals)) * 100,
            "mean_orig": orig_clean.mean(),
            "mean_spoof": spoof_clean.mean(),
            "median_orig": orig_clean.median(),
            "median_spoof": spoof_clean.median(),
            "std_orig": orig_clean.std(),
            "std_spoof": spoof_clean.std(),
            "min_orig": orig_clean.min(),
            "min_spoof": spoof_clean.min(),
            "max_orig": orig_clean.max(),
            "max_spoof": spoof_clean.max(),
        }

        # Calculate percent differences
        stats["mean_diff_pct"] = (
            (stats["mean_spoof"] - stats["mean_orig"]) / abs(stats["mean_orig"]) * 100
            if stats["mean_orig"] != 0
            else 0
        )
        stats["std_diff_pct"] = (
            (stats["std_spoof"] - stats["std_orig"]) / abs(stats["std_orig"]) * 100
            if stats["std_orig"] != 0
            else 0
        )

        return stats

    # Important columns to compare
    important_cols = {
        "P1": ["RA0300", "RA0200", "PA0200", "PG0110", "PG0210"],
        "H1": ["HW0010", "HB0300", "HB2300"],
        "D1": ["DA1110", "DA1120", "DA2101", "DL1110", "DI2000", "DN3001"],
    }

    all_stats = []

    print("\n" + "=" * 80)
    print("KEY STATISTICS COMPARISON")
    print("=" * 80)

    for file_name, cols in important_cols.items():
        if file_name == "P1":
            orig_df, spoof_df = p_orig, p_spoof
        elif file_name == "H1":
            orig_df, spoof_df = h_orig, h_spoof
        else:
            orig_df, spoof_df = d_orig, d_spoof

        print(f"\n{file_name}:")
        print("-" * 80)

        for col in cols:
            stats = compare_col(orig_df, spoof_df, col, file_name)
            if stats:
                all_stats.append(stats)

                print(f"\n  {col}:")
                print(
                    f"    Mean:   {stats['mean_orig']:>12.2f} → {stats['mean_spoof']:>12.2f} "
                    f"({stats['mean_diff_pct']:+.1f}%)"
                )
                print(
                    f"    Median: {stats['median_orig']:>12.2f} → {stats['median_spoof']:>12.2f}"
                )
                print(
                    f"    Std:    {stats['std_orig']:>12.2f} → {stats['std_spoof']:>12.2f} "
                    f"({stats['std_diff_pct']:+.1f}%)"
                )
                print(
                    f"    Range:  [{stats['min_orig']:.2f}, {stats['max_orig']:.2f}] → "
                    f"[{stats['min_spoof']:.2f}, {stats['max_spoof']:.2f}]"
                )
                print(
                    f"    Missing: {stats['missing_pct_orig']:.1f}% → {stats['missing_pct_spoof']:.1f}%"
                )

    # Check correlations
    print("\n" + "=" * 80)
    print("CORRELATION PRESERVATION")
    print("=" * 80)

    # Age vs Income correlation in P1
    if "RA0300" in p_orig.columns and "PG0110" in p_orig.columns:
        orig_age = pd.to_numeric(p_orig["RA0300"], errors="coerce")
        orig_income = pd.to_numeric(p_orig["PG0110"], errors="coerce")
        spoof_age = pd.to_numeric(p_spoof["RA0300"], errors="coerce")
        spoof_income = pd.to_numeric(p_spoof["PG0110"], errors="coerce")

        # Create mask for non-missing in both
        orig_mask = orig_age.notna() & orig_income.notna()
        spoof_mask = spoof_age.notna() & spoof_income.notna()

        if orig_mask.sum() > 0 and spoof_mask.sum() > 0:
            orig_corr = orig_age[orig_mask].corr(orig_income[orig_mask])
            spoof_corr = spoof_age[spoof_mask].corr(spoof_income[spoof_mask])

            print(f"\nAge vs Income (P1):")
            print(f"  Original correlation: {orig_corr:.4f}")
            print(f"  Spoofed correlation:  {spoof_corr:.4f}")
            print(f"  Difference:           {abs(orig_corr - spoof_corr):.4f}")

    # Asset vs Income correlation in D1
    if "DA1110" in d_orig.columns and "DI2000" in d_orig.columns:
        orig_asset = pd.to_numeric(d_orig["DA1110"], errors="coerce")
        orig_income = pd.to_numeric(d_orig["DI2000"], errors="coerce")
        spoof_asset = pd.to_numeric(d_spoof["DA1110"], errors="coerce")
        spoof_income = pd.to_numeric(d_spoof["DI2000"], errors="coerce")

        orig_mask = orig_asset.notna() & orig_income.notna()
        spoof_mask = spoof_asset.notna() & spoof_income.notna()

        if orig_mask.sum() > 0 and spoof_mask.sum() > 0:
            orig_corr = orig_asset[orig_mask].corr(orig_income[orig_mask])
            spoof_corr = spoof_asset[spoof_mask].corr(spoof_income[spoof_mask])

            print(f"\nHome Value vs Income (D1):")
            print(f"  Original correlation: {orig_corr:.4f}")
            print(f"  Spoofed correlation:  {spoof_corr:.4f}")
            print(f"  Difference:           {abs(orig_corr - spoof_corr):.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_stats:
        mean_diffs = [abs(s["mean_diff_pct"]) for s in all_stats]
        std_diffs = [abs(s["std_diff_pct"]) for s in all_stats]

        print(f"\nAverage absolute difference in means:  {np.mean(mean_diffs):.2f}%")
        print(f"Average absolute difference in stds:   {np.mean(std_diffs):.2f}%")
        print(f"\nMax difference in means:  {np.max(mean_diffs):.2f}%")
        print(f"Max difference in stds:   {np.max(std_diffs):.2f}%")

    print("\n✅ Comparison complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare original and spoofed HFCS data"
    )
    parser.add_argument(
        "--original",
        type=str,
        default="tests/test_macro_data/unit/sample_raw_data/hfcs/2014",
        help="Directory with original data",
    )
    parser.add_argument(
        "--spoofed",
        type=str,
        default="spoof_data/spoofed_hfcs_output",
        help="Directory with spoofed data",
    )

    args = parser.parse_args()

    compare_distributions(Path(args.original), Path(args.spoofed))


if __name__ == "__main__":
    main()
