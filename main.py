"""Entry point for reproducible conformal prediction experiments.

Run with:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import load_all_datasets
from experiments import aggregate_results, run_all_experiments
from plots import generate_all_figures
from tables import save_tables



def ensure_directories() -> dict[str, Path]:
    paths = {
        "results": Path("results"),
        "figures": Path("figures"),
        "tables": Path("tables"),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths



def main() -> None:
    paths = ensure_directories()
    seeds = [0, 1, 2, 3, 4]

    print("Loading datasets...")
    datasets = load_all_datasets()
    print(f"Loaded datasets: {', '.join(datasets.keys())}")

    print("Running experiments (this may take several minutes)...")
    raw_df = run_all_experiments(datasets=datasets, seeds=seeds, alpha=0.1)
    agg_df = aggregate_results(raw_df)

    raw_path = paths["results"] / "raw_results.csv"
    agg_path = paths["results"] / "aggregated_results.csv"
    raw_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print("Generating figures...")
    generate_all_figures(raw_df, paths["figures"])

    print("Generating tables...")
    tables = save_tables(raw_df, paths["tables"])

    # Save pretty text versions for convenience in manuscript drafting.
    for name, table in tables.items():
        pretty_path = paths["tables"] / f"{name}_pretty.txt"
        with open(pretty_path, "w", encoding="utf-8") as f:
            f.write(table.to_string(index=False))

    summary = pd.DataFrame(
        {
            "artifact": ["raw_results", "aggregated_results", "figures", "tables"],
            "path": [str(raw_path), str(agg_path), str(paths["figures"]), str(paths["tables"])],
        }
    )
    summary.to_csv(paths["results"] / "artifact_summary.csv", index=False)

    print("Done. All reproducible outputs have been generated.")


if __name__ == "__main__":
    main()
