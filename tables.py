"""Table generation utilities for paper-ready conformal summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COVERAGE = 0.9



def _round_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns
    out = df.copy()
    out[num_cols] = out[num_cols].round(decimals)
    return out



def make_table1_distribution_shift(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[(raw_df["label_noise"] == 0.0) & (raw_df["sample_size"].astype(str) == "full")]
    grouped = (
        df.groupby(["dataset", "model", "distribution_noise"], as_index=False)
        .agg(coverage=("coverage", "mean"), stability=("coverage", "std"))
    )

    cov_wide = grouped.pivot_table(
        index=["dataset", "model"],
        columns="distribution_noise",
        values="coverage",
    )
    cov_wide.columns = [f"cov_noise_{c}" for c in cov_wide.columns]
    stability = grouped.groupby(["dataset", "model"], as_index=True)["coverage"].std().rename("stability_across_levels")

    table = cov_wide.join(stability).reset_index()
    table["best_noise_robustness"] = table[[c for c in table.columns if c.startswith("cov_noise_")]].mean(axis=1)
    return _round_df(table)



def make_table2_label_noise_sample_size(raw_df: pd.DataFrame) -> pd.DataFrame:
    label_df = raw_df[(raw_df["distribution_noise"] == 0.0) & (raw_df["sample_size"].astype(str) == "full")]
    label = (
        label_df.groupby(["dataset", "model", "label_noise"], as_index=False)["coverage"]
        .mean()
        .pivot_table(index=["dataset", "model"], columns="label_noise", values="coverage")
    )
    label.columns = [f"cov_label_noise_{c}" for c in label.columns]

    sample_df = raw_df[(raw_df["distribution_noise"] == 0.0) & (raw_df["label_noise"] == 0.0)]
    sample_df = sample_df.copy()
    sample_df["sample_label"] = sample_df["sample_size"].astype(str)
    sample = (
        sample_df.groupby(["dataset", "model", "sample_label"], as_index=False)["coverage"]
        .mean()
        .pivot_table(index=["dataset", "model"], columns="sample_label", values="coverage")
    )

    ordered_cols = [c for c in ["100", "500", "1000", "full"] if c in sample.columns]
    sample = sample[ordered_cols]
    sample.columns = [f"cov_sample_size_{c}" for c in sample.columns]

    table = label.join(sample, how="outer").reset_index()
    return _round_df(table)



def make_table3_setsize_failures(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["dataset", "model"], as_index=False).agg(
        avg_set_size_clean=(
            "avg_set_size",
            lambda s: raw_df.loc[s.index][
                (raw_df.loc[s.index, "distribution_noise"] == 0.0)
                & (raw_df.loc[s.index, "label_noise"] == 0.0)
                & (raw_df.loc[s.index, "sample_size"].astype(str) == "full")
            ]["avg_set_size"].mean(),
        ),
        avg_set_size_high_shift=(
            "avg_set_size",
            lambda s: raw_df.loc[s.index][
                (raw_df.loc[s.index, "distribution_noise"] == 1.0)
                & (raw_df.loc[s.index, "label_noise"] == 0.0)
                & (raw_df.loc[s.index, "sample_size"].astype(str) == "full")
            ]["avg_set_size"].mean(),
        ),
        avg_set_size_high_label_noise=(
            "avg_set_size",
            lambda s: raw_df.loc[s.index][
                (raw_df.loc[s.index, "distribution_noise"] == 0.0)
                & (raw_df.loc[s.index, "label_noise"] == 0.3)
                & (raw_df.loc[s.index, "sample_size"].astype(str) == "full")
            ]["avg_set_size"].mean(),
        ),
        failed_settings=("coverage", lambda s: int((s < TARGET_COVERAGE).sum())),
        mean_coverage_gap=("coverage_gap", "mean"),
        worst_coverage=("coverage", "min"),
    )
    return _round_df(grouped)



def save_tables(raw_df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = make_table1_distribution_shift(raw_df)
    t2 = make_table2_label_noise_sample_size(raw_df)
    t3 = make_table3_setsize_failures(raw_df)

    t1.to_csv(out_dir / "table1_distribution_shift.csv", index=False)
    t2.to_csv(out_dir / "table2_label_noise_sample_size.csv", index=False)
    t3.to_csv(out_dir / "table3_setsize_failures.csv", index=False)

    return {"table1": t1, "table2": t2, "table3": t3}
