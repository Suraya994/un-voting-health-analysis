#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic data generator for the UN GA Voting × Global Health Governance project.

This script creates reproducible, schema-compatible CSV files for testing the
analysis pipeline when the real research dataset cannot be shared publicly.

The generated values are artificial. They are useful for:
    1. checking whether the analysis code runs end-to-end;
    2. demonstrating the expected input file structure;
    3. allowing reviewers or collaborators to reproduce the workflow.

They must not be interpreted as empirical findings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = list(range(2004, 2025))

REGIONS = {
    "DZA": "Middle East & North Africa", "AGO": "Sub-Saharan Africa",
    "ARG": "Latin America & Caribbean", "AUS": "East Asia & Pacific",
    "BRA": "Latin America & Caribbean", "CHN": "East Asia & Pacific",
    "DEU": "Europe & Central Asia", "EGY": "Middle East & North Africa",
    "ETH": "Sub-Saharan Africa", "FRA": "Europe & Central Asia",
    "GBR": "Europe & Central Asia", "GHA": "Sub-Saharan Africa",
    "IDN": "East Asia & Pacific", "IND": "South Asia",
    "IRN": "Middle East & North Africa", "IRQ": "Middle East & North Africa",
    "KEN": "Sub-Saharan Africa", "KOR": "East Asia & Pacific",
    "MEX": "Latin America & Caribbean", "MYS": "East Asia & Pacific",
    "NGA": "Sub-Saharan Africa", "PAK": "South Asia",
    "PHL": "East Asia & Pacific", "POL": "Europe & Central Asia",
    "RUS": "Europe & Central Asia", "SAU": "Middle East & North Africa",
    "THA": "East Asia & Pacific", "TUR": "Middle East & North Africa",
    "TZA": "Sub-Saharan Africa", "UGA": "Sub-Saharan Africa",
    "USA": "North America", "VNM": "East Asia & Pacific",
    "ZAF": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa",
    "CAN": "North America", "ESP": "Europe & Central Asia",
    "ITA": "Europe & Central Asia", "JPN": "East Asia & Pacific",
    "NLD": "Europe & Central Asia", "SWE": "Europe & Central Asia",
}

GROUP_MAP = {
    region: ("West" if region in {"North America", "Europe & Central Asia"} else "Global South")
    for region in set(REGIONS.values())
}

OUTPUT_FILES = {
    "panel": "bm_who_augmented_panel_100countries_2004_2024.csv",
    "edges": "un_ga_voting_similarity_edges_100countries_2004_2024.csv",
    "network_metrics": "un_ga_voting_network_metrics_100countries_2004_2024.csv",
    "country_year": "un_ga_voting_country_year_summary_100countries.csv",
}


def build_panel(rng: np.random.Generator) -> pd.DataFrame:
    """Build a synthetic country-year panel matching the empirical schema."""
    rows: list[dict] = []
    isos = list(REGIONS.keys())

    for iso in isos:
        region = REGIONS[iso]
        group = GROUP_MAP[region]
        base_uhc = rng.uniform(30, 80)
        base_gdp = np.exp(rng.uniform(7.5, 11.5))
        base_yes = rng.uniform(0.55, 0.95)
        base_eigen = rng.uniform(0.0, 0.5)

        for year in YEARS:
            t = (year - min(YEARS)) / (max(YEARS) - min(YEARS))
            uhc = np.clip(base_uhc + t * 20 + rng.normal(0, 3), 20, 100)
            yes_share = np.clip(base_yes + rng.normal(0, 0.05), 0, 1)
            no_share = np.clip(rng.uniform(0, 0.05), 0, 1 - yes_share)
            abstain_share = np.clip(1 - yes_share - no_share, 0, 1)
            gdp_pc_ppp = base_gdp * (1 + t * 0.3 + rng.normal(0, 0.05))

            physicians = rng.uniform(0.05, 4.5)
            hospital_beds = rng.uniform(0.5, 8)
            un_eigen = np.clip(base_eigen + rng.normal(0, 0.1), 0, 1)
            un_strength = rng.uniform(5, 40)

            rows.append({
                "iso3c": iso,
                "country": iso,
                "year": year,
                "country_name_wdi": iso,
                "global_group": group,
                "region": region,
                "include_flag": "yes",
                "un_yes_share": yes_share,
                "un_no_share": no_share,
                "un_abstain_share": abstain_share,
                "un_degree": rng.uniform(0.05, 0.40),
                "un_strength": un_strength,
                "un_eigen": un_eigen,
                "un_betweenness": rng.uniform(0, 0.05),
                "un_closeness": rng.uniform(0.3, 0.8),
                "un_local_clustering": rng.uniform(0.3, 0.9),
                "un_network_density": rng.uniform(0.10, 0.20),
                "un_network_clustering": rng.uniform(0.4, 0.8),
                "un_network_components": rng.choice([1, 2]),
                "un_votes_total": rng.integers(60, 120),
                "un_votes_yes": int(yes_share * 80),
                "un_votes_no": int(no_share * 80),
                "un_votes_abstain": rng.integers(0, 10),
                "health_spending": rng.uniform(1.5, 12),
                "physicians": physicians,
                "nurses": rng.uniform(0.5, 12),
                "hospital_beds": hospital_beds,
                "internet": rng.uniform(5, 95),
                "gdp_pc_ppp": gdp_pc_ppp,
                "population": rng.integers(5_000_000, 1_400_000_000),
                "life_expectancy": rng.uniform(55, 83),
                "UHC_INDEX_REPORTED": int(uhc),
                "UHC_SCI_INFECT": rng.integers(20, 90),
                "GHED_EXTCHE_SHA2011": rng.uniform(0.5, 30),
                "l_un_eigen": un_eigen,
                "l_un_degree": rng.uniform(0.05, 0.40),
                "l_un_strength": un_strength,
                "log_gdp": np.log1p(gdp_pc_ppp),
                "voting_entropy": 0.0,
                "network_influence": un_eigen * un_strength,
                "physicians_beds": physicians * hospital_beds,
                "notes.x": np.nan,
                "notes.y": np.nan,
                "who_name_override": np.nan,
                "wgi_name_override": np.nan,
            })

    panel = pd.DataFrame(rows)
    panel["voting_entropy"] = (
        -panel["un_yes_share"].clip(1e-9) * np.log(panel["un_yes_share"].clip(1e-9))
        -panel["un_no_share"].clip(1e-9) * np.log(panel["un_no_share"].clip(1e-9))
        -panel["un_abstain_share"].clip(1e-9) * np.log(panel["un_abstain_share"].clip(1e-9))
    )
    return panel


def build_edges(rng: np.random.Generator, isos: list[str]) -> pd.DataFrame:
    """Build synthetic dyadic UN voting similarity edges."""
    edge_rows: list[dict] = []
    for year in YEARS:
        for i, iso_origin in enumerate(isos):
            for iso_destination in isos[i + 1:]:
                if rng.random() > 0.55:
                    continue
                edge_rows.append({
                    "year": year,
                    "iso_o": iso_origin,
                    "iso_d": iso_destination,
                    "common_votes": rng.integers(40, 120),
                    "un_voting_agreement": rng.beta(5, 2),
                    "un_voting_cosine": rng.beta(6, 2),
                })
    return pd.DataFrame(edge_rows)


def build_network_metrics(rng: np.random.Generator, isos: list[str]) -> pd.DataFrame:
    """Build synthetic country-year network metrics."""
    rows: list[dict] = []
    for iso in isos:
        region = REGIONS[iso]
        for year in YEARS:
            rows.append({
                "iso3c": iso,
                "year": year,
                "region": region,
                "global_group": GROUP_MAP[region],
                "un_degree": rng.uniform(0.05, 0.40),
                "un_strength": rng.uniform(5, 40),
                "un_eigen": rng.uniform(0, 0.5),
                "un_betweenness": rng.uniform(0, 0.05),
                "un_closeness": rng.uniform(0.3, 0.8),
                "un_local_clustering": rng.uniform(0.3, 0.9),
                "un_network_density": rng.uniform(0.10, 0.20),
                "un_network_clustering": rng.uniform(0.4, 0.8),
                "un_network_components": 2,
                "country_name_wdi": iso,
                "include_flag": "yes",
                "who_name_override": np.nan,
                "wgi_name_override": np.nan,
                "notes": np.nan,
            })
    return pd.DataFrame(rows)


def build_country_year_summary(rng: np.random.Generator, isos: list[str]) -> pd.DataFrame:
    """Build synthetic country-year UN voting summary data."""
    rows: list[dict] = []
    for iso in isos:
        region = REGIONS[iso]
        for year in YEARS:
            yes_share = rng.uniform(0.55, 0.95)
            no_share = rng.uniform(0, 0.05)
            rows.append({
                "iso3c": iso,
                "country_name_wdi": iso,
                "global_group": GROUP_MAP[region],
                "region": region,
                "year": year,
                "un_votes_total": rng.integers(60, 120),
                "un_votes_yes": int(yes_share * 80),
                "un_votes_no": rng.integers(0, 5),
                "un_votes_abstain": rng.integers(0, 10),
                "un_yes_share": yes_share,
                "un_no_share": no_share,
                "un_abstain_share": rng.uniform(0, 0.10),
            })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    """Write a CSV file and print a concise status message."""
    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"✓ {filename}: {len(df):,} rows")


def write_manifest(output_dir: Path, seed: int, datasets: dict[str, pd.DataFrame]) -> None:
    """Write a small machine-readable summary of the synthetic run."""
    manifest = {
        "seed": seed,
        "years": [min(YEARS), max(YEARS)],
        "countries": len(REGIONS),
        "warning": "Synthetic data are artificial and must not be interpreted as empirical evidence.",
        "files": {
            name: {
                "filename": OUTPUT_FILES[name],
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            }
            for name, frame in datasets.items()
        },
    }
    path = output_dir / "synthetic_data_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✓ synthetic_data_manifest.json: {len(datasets)} dataset summaries")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate schema-compatible synthetic data for the UN GA voting analysis."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the synthetic CSV files will be written. Default: data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic data. Default: 42",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    isos = list(REGIONS.keys())

    datasets = {
        "panel": build_panel(rng),
        "edges": build_edges(rng, isos),
        "network_metrics": build_network_metrics(rng, isos),
        "country_year": build_country_year_summary(rng, isos),
    }

    for name, frame in datasets.items():
        write_csv(frame, output_dir, OUTPUT_FILES[name])
    write_manifest(output_dir, args.seed, datasets)

    print("\nSynthetic dataset generation completed.")
    print("Note: these files are artificial and should only be used for testing/reproducibility.")


if __name__ == "__main__":
    main()
