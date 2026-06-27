#!/usr/bin/env python3
"""Run the Canada provincial MacroABM workflow (equivalent to Sample_macroabm-Canada_provincial_run.ipynb).

End-to-end steps:
1. Build the data pickle from raw inputs (skipped if the pickle already exists).
2. Load the pickle and run the simulation with TFP and technical-coefficient growth.
3. Plot nominal GDP, real GDP, and top-industry production; save PNGs to output/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macro_data import DataWrapper, configuration_utils
from macro_data.configuration.dataconfiguration import DataConfiguration
from macro_data.configuration.countries import Country as CountryCode
from macro_data.configuration.region import Region
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

CANADIAN_PROVINCES = [
    Region.from_code("CAN_AB", "Alberta"),
    Region.from_code("CAN_BC", "British Columbia"),
    Region.from_code("CAN_MB", "Manitoba"),
    Region.from_code("CAN_NB", "New Brunswick"),
    Region.from_code("CAN_NL", "Newfoundland and Labrador"),
    Region.from_code("CAN_NS", "Nova Scotia"),
    Region.from_code("CAN_ON", "Ontario"),
    Region.from_code("CAN_PE", "Prince Edward Island"),
    Region.from_code("CAN_QC", "Quebec"),
    Region.from_code("CAN_SK", "Saskatchewan"),
]

PROVINCE_LABELS = {
    "CAN_AB": "Alberta",
    "CAN_BC": "British Columbia",
    "CAN_MB": "Manitoba",
    "CAN_NB": "New Brunswick",
    "CAN_NL": "Newfoundland",
    "CAN_NS": "Nova Scotia",
    "CAN_ON": "Ontario",
    "CAN_PE": "PEI",
    "CAN_QC": "Quebec",
    "CAN_SK": "Saskatchewan",
}

TreeNode = Dict[str, Any]


def build_data_configuration() -> DataConfiguration:
    data_config = configuration_utils.default_data_configuration(
        countries=["CAN"],
        aggregate_industries=False,
        proxy_country_dict={"CAN": "FRA"},
        use_disagg_can_2014_reader=True,
    )

    data_config.year = 2014
    data_config.time_unit = 3
    data_config.can_disaggregation = True
    data_config.aggregate_industries = False
    data_config.prune_date = None
    data_config.seed = 0

    base_config = data_config.country_configs[CountryCode("CAN")]
    base_config.single_firm_per_industry = True
    base_config.single_bank = True
    base_config.single_government_entity = True
    base_config.firms_configuration.constructor = "Default"
    base_config.scale = 1000

    for province in CANADIAN_PROVINCES:
        data_config.country_configs[province] = base_config
        data_config.country_configs[province].eu_proxy_country = CountryCode("FRA")

    data_config.aggregation_structure = {CountryCode("CAN"): CANADIAN_PROVINCES}
    return data_config


def create_data_pickle(
    input_path: Path,
    pkl_path: Path,
    *,
    force: bool = False,
) -> None:
    if pkl_path.exists() and not force:
        print(f"Pickle already exists at {pkl_path} — skipping data creation.")
        return

    data_config = build_data_configuration()
    t0 = time.time()
    creator = DataWrapper.from_config(
        configuration=data_config,
        raw_data_path=input_path,
        single_hfcs_survey=True,
    )
    creator.save(pkl_path)
    print(f"Pickle saved to {pkl_path}  ({(time.time() - t0) / 60:.1f} min)")


def build_simulation_configuration(
    n_industries: int,
    *,
    timesteps: int,
    seed: int,
) -> SimulationConfiguration:
    tfp_base_growth_rate = 0.001
    tfp_investment_elasticity = 0.5
    productivity_growth_investment_effectiveness = 0.3
    technical_coefficients_investment_effectiveness = 0.3
    diminishing_returns_factor = 0.1
    hurdle_rate = 0.01
    investment_effectiveness = 0.3
    technical_investment_effectiveness = 0.3
    technical_diminishing_returns = 0.1
    tfp_investment_share = 0.5
    max_investment_fraction = 0.2

    config = SimulationConfiguration(
        seed=seed,
        country_configurations={
            province: CountryConfiguration.n_industry_default(n_industries=n_industries)
            for province in CANADIAN_PROVINCES
        },
        t_max=timesteps,
    )

    for province in CANADIAN_PROVINCES:
        firms = config.country_configurations[province].firms

        firms.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
        firms.functions.productivity_investment_planner.parameters.update(
            {
                "n_firms": n_industries,
                "tfp_investment_share": tfp_investment_share,
                "max_investment_fraction": max_investment_fraction,
                "investment_effectiveness": investment_effectiveness,
                "technical_investment_effectiveness": technical_investment_effectiveness,
                "technical_diminishing_returns": technical_diminishing_returns,
                "hurdle_rate": hurdle_rate,
            }
        )

        firms.functions.productivity_growth.name = "SimpleTFPGrowth"
        firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": productivity_growth_investment_effectiveness,
        }
        firms.parameters.tfp_base_growth_rate = tfp_base_growth_rate
        firms.parameters.tfp_investment_elasticity = tfp_investment_elasticity

        firms.functions.technical_coefficients_growth.name = "SimpleTechnicalGrowth"
        firms.functions.technical_coefficients_growth.parameters = {
            "investment_effectiveness": technical_coefficients_investment_effectiveness,
            "diminishing_returns_factor": diminishing_returns_factor,
        }

    return config


def run_simulation(
    data: DataWrapper,
    config: SimulationConfiguration,
    h5_path: Path,
    *,
    save_h5: bool = True,
) -> Optional[Path]:
    model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=config)

    t0 = time.time()
    model.run()
    print(f"Simulation complete in {(time.time() - t0) / 60:.1f} min")

    if not save_h5:
        print("Skipping HDF5 write (--no-save-h5).")
        return None

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_dir=h5_path.parent, file_name=h5_path.name)
    print(f"Results saved to {h5_path}")
    return h5_path


def load_industry_mapping(input_path: Path) -> Dict[int, str]:
    industry_names_csv = input_path / "can_industries_wnames.csv"
    if not industry_names_csv.exists():
        return {}
    df = pd.read_csv(industry_names_csv)
    return dict(zip(df["Firm_ID"], df["Industry_Name"]))


def _build_tree(group: h5py.Group) -> TreeNode:
    tree: TreeNode = {}
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            tree[key] = {"type": "dataset"}
        else:
            tree[key] = {"type": "group", "children": _build_tree(obj)}
    return tree


def get_tree(file_path: Path) -> TreeNode:
    with h5py.File(file_path, "r") as f:
        return _build_tree(f)


def list_dataset_paths(tree: TreeNode, prefix: str = "") -> List[str]:
    paths: List[str] = []
    for key in sorted(tree.keys()):
        entry = tree[key]
        current_path = f"{prefix}/{key}" if prefix else key
        if entry["type"] == "dataset":
            paths.append(current_path)
        else:
            paths.extend(list_dataset_paths(entry["children"], current_path))
    return paths


def load_dataset(file_path: Path, dataset_path: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        return f[dataset_path][()]


def compute_real_gdp(file_path: Path, province: str) -> Optional[np.ndarray]:
    gdp_path = f"{province}/economy/gdp_output"
    cpi_path = f"{province}/economy/cpi"
    with h5py.File(file_path, "r") as f:
        if gdp_path not in f or cpi_path not in f:
            return None
        gdp = f[gdp_path][()].astype(float)
        cpi = f[cpi_path][()].astype(float)
    return np.divide(gdp, cpi, out=np.full_like(gdp, np.nan), where=cpi != 0)


def get_column_labels(
    file_path: Path,
    dataset_path: str,
    industry_mapping: Optional[Dict[int, str]] = None,
) -> Optional[List[str]]:
    columns_path = f"{dataset_path}_columns"
    with h5py.File(file_path, "r") as f:
        if columns_path not in f:
            return None
        cols = np.asarray(f[columns_path][()])
    if cols.ndim == 1:
        if industry_mapping:
            return [industry_mapping.get(int(i), str(int(i))) for i in cols]
        return [str(int(i)) for i in cols]
    if cols.ndim == 2 and cols.shape[1] == 2:
        if industry_mapping:
            return [
                f"[{industry_mapping.get(int(a), str(int(a)))}, {industry_mapping.get(int(b), str(int(b)))}]"
                for a, b in cols
            ]
        return [f"[{int(a)}, {int(b)}]" for a, b in cols]
    return [str(x) for x in cols.flatten()]


def merge_trees(trees: Iterable[TreeNode]) -> TreeNode:
    merged: TreeNode = {}
    for tree in trees:
        for key, entry in tree.items():
            if key not in merged:
                merged[key] = entry
                continue
            if merged[key]["type"] == "group" or entry["type"] == "group":
                merged[key] = {
                    "type": "group",
                    "children": merge_trees(
                        [
                            merged[key].get("children", {}),
                            entry.get("children", {}),
                        ]
                    ),
                }
    return merged


def add_can_region(tree: TreeNode) -> List[str]:
    province_keys = [k for k, v in tree.items() if v["type"] == "group" and k.startswith("CAN_")]
    if province_keys:
        can_children = merge_trees([tree[p]["children"] for p in province_keys])
        tree["CAN"] = {"type": "group", "children": can_children}
    return province_keys


def sum_regions(file_path: Path, province_keys: List[str], tail_path: str) -> Optional[np.ndarray]:
    total: Optional[np.ndarray] = None
    with h5py.File(file_path, "r") as f:
        for province in province_keys:
            full = f"{province}/{tail_path}"
            if full not in f:
                continue
            data = f[full][()].astype(float)
            total = data.copy() if total is None else total + data
    return total


def inspect_results(h5_path: Path, input_path: Path) -> tuple[List[str], List[str]]:
    industry_mapping = load_industry_mapping(input_path)
    print(f"Industry mapping loaded: {len(industry_mapping)} industries")

    tree = get_tree(h5_path)
    regions = add_can_region(tree)
    all_paths = list_dataset_paths(tree)

    print(f"\nProvinces found : {regions}")
    print(f"Total datasets  : {len(all_paths)}")
    print("\nTop-level keys:")
    for k, v in sorted(tree.items()):
        print(f"  {k}  [{v['type']}]")

    return regions, all_paths


def plot_results(
    h5_path: Path,
    output_directory: Path,
    regions: List[str],
    input_path: Path,
    *,
    show_plots: bool = False,
) -> None:
    industry_mapping = load_industry_mapping(input_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    with h5py.File(h5_path, "r") as f:
        for region in regions:
            path = f"{region}/economy/gdp_output"
            if path in f:
                gdp = f[path][()].astype(float)
                ax.plot(gdp, label=PROVINCE_LABELS.get(region, region))
    ax.set_title("Nominal GDP output by province")
    ax.set_xlabel("Timestep (3 months)")
    ax.set_ylabel("Nominal GDP output (model units)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "gdp_by_province.png", dpi=120)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for region in regions:
        real_gdp = compute_real_gdp(h5_path, region)
        if real_gdp is not None:
            ax.plot(real_gdp, label=PROVINCE_LABELS.get(region, region))
    ax.set_title("Real GDP by province (GDP / CPI)")
    ax.set_xlabel("Timestep (3 months)")
    ax.set_ylabel("Real GDP (model units)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "real_gdp_by_province.png", dpi=120)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    production_total = sum_regions(h5_path, regions, "firms/production")
    if production_total is not None:
        labels = get_column_labels(h5_path, f"{regions[0]}/firms/production", industry_mapping)
        n_show = min(10, production_total.shape[1])
        mean_prod = production_total.mean(axis=0)
        top_idx = np.argsort(mean_prod)[::-1][:n_show]

        fig, ax = plt.subplots(figsize=(12, 5))
        for i in top_idx:
            name = labels[i] if labels else str(i)
            ax.plot(production_total[:, i], label=name)
        ax.set_title(f"Top {n_show} industries by production (sum across provinces)")
        ax.set_xlabel("Timestep (3 months)")
        ax.set_ylabel("Production (model units)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_directory / "production_top_industries.png", dpi=120)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Canada provincial MacroABM demo (see Sample_macroabm-Canada_provincial_run.ipynb).",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to raw_data/ containing icio/, wiod_sea/, hfcs/, exchange_rates/, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for pickle, HDF5 results, and plots (default: output/).",
    )
    parser.add_argument(
        "--output-filename",
        default="results_provinces_wTFP.h5",
        help="HDF5 results filename when --save-h5 is not used (default: results_provinces_wTFP.h5).",
    )
    h5_group = parser.add_mutually_exclusive_group()
    h5_group.add_argument(
        "--save-h5",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Write simulation results to HDF5. Optionally pass PATH; otherwise uses "
        "<output-dir>/<output-filename>. Omit this flag entirely to use the same default save location.",
    )
    h5_group.add_argument(
        "--no-save-h5",
        action="store_true",
        help="Run the simulation without writing an HDF5 file.",
    )
    parser.add_argument(
        "--pkl-path",
        type=Path,
        default=None,
        help="Pickle path (default: <output-dir>/data_provincial_model.pkl).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=16,
        help="Number of 3-month timesteps to simulate (default: 16 = 4 years).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the simulation (default: 0).",
    )
    parser.add_argument(
        "--force-rebuild-pickle",
        action="store_true",
        help="Rebuild the data pickle even if it already exists.",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Only build/load the pickle; do not run the simulation.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plotting after the simulation.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving PNGs.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print all dataset paths in the HDF5 file after plotting.",
    )
    return parser.parse_args()


def resolve_h5_path(args: argparse.Namespace, output_directory: Path) -> tuple[Path, bool]:
    if args.no_save_h5:
        return output_directory / args.output_filename, False
    if args.save_h5 is not None:
        if args.save_h5 == "":
            return output_directory / args.output_filename, True
        return Path(args.save_h5), True
    return output_directory / args.output_filename, True


def main() -> None:
    args = parse_args()

    input_path: Path = args.input_path.resolve()
    output_directory: Path = args.output_dir.resolve()
    pkl_path: Path = (args.pkl_path or output_directory / "data_provincial_model.pkl").resolve()

    output_directory.mkdir(parents=True, exist_ok=True)
    h5_path, save_h5 = resolve_h5_path(args, output_directory)
    h5_path = h5_path.resolve()

    print(f"Input path : {input_path}  exists={input_path.exists()}")
    print(f"Pickle     : {pkl_path}")
    print(f"H5 output  : {h5_path}  save={save_h5}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    create_data_pickle(input_path, pkl_path, force=args.force_rebuild_pickle)

    if args.skip_simulation:
        print("Skipping simulation (--skip-simulation).")
        return

    data = DataWrapper.init_from_pickle(pkl_path)
    n_industries = data.n_industries
    print(f"n_industries : {n_industries}")
    print(f"industries   : {list(data.industries)}")
    print(f"Timesteps    : {args.timesteps}")
    print(f"Seed         : {args.seed}")

    config = build_simulation_configuration(n_industries, timesteps=args.timesteps, seed=args.seed)
    print("Configuration built.")

    h5_path = run_simulation(data, config, h5_path, save_h5=save_h5)

    if args.skip_plots or h5_path is None:
        if h5_path is None and not args.skip_plots:
            print("Skipping plots because no HDF5 file was written.")
        return

    regions, all_paths = inspect_results(h5_path, input_path)
    plot_results(
        h5_path,
        output_directory,
        regions,
        input_path,
        show_plots=args.show_plots,
    )

    if args.list_datasets:
        print("\nAll dataset paths in H5 file:")
        for path in all_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
