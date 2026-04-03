"""Publication-quality plotting utilities for conformal experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_COVERAGE = 0.9



def _setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )



def _save(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def figure1_coverage_vs_shift(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    df = raw_df[(raw_df["label_noise"] == 0.0) & (raw_df["sample_size"].astype(str) == "full")]

    for dataset in sorted(df["dataset"].unique()):
        dsub = df[df["dataset"] == dataset]
        summary = dsub.groupby(["model", "distribution_noise"], as_index=False)["coverage"].mean()

        fig, ax = plt.subplots()
        for model in summary["model"].unique():
            m = summary[summary["model"] == model]
            ax.plot(m["distribution_noise"], m["coverage"], marker="o", label=model)
        ax.axhline(TARGET_COVERAGE, linestyle="--", color="black", linewidth=1.2, label="target=0.9")
        ax.set_title(f"Figure 1: Coverage vs Distribution Shift ({dataset})")
        ax.set_xlabel("Gaussian noise level")
        ax.set_ylabel("Empirical coverage")
        ax.legend(frameon=False)
        _save(fig, out_dir / f"figure1_coverage_vs_shift_{dataset}")



def figure2_setsize_vs_shift(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    df = raw_df[(raw_df["label_noise"] == 0.0) & (raw_df["sample_size"].astype(str) == "full")]

    for dataset in sorted(df["dataset"].unique()):
        dsub = df[df["dataset"] == dataset]
        summary = dsub.groupby(["model", "distribution_noise"], as_index=False)["avg_set_size"].mean()

        fig, ax = plt.subplots()
        for model in summary["model"].unique():
            m = summary[summary["model"] == model]
            ax.plot(m["distribution_noise"], m["avg_set_size"], marker="o", label=model)
        ax.set_title(f"Figure 2: Prediction Set Size vs Distribution Shift ({dataset})")
        ax.set_xlabel("Gaussian noise level")
        ax.set_ylabel("Average prediction set size")
        ax.legend(frameon=False)
        _save(fig, out_dir / f"figure2_setsize_vs_shift_{dataset}")



def figure3_coverage_vs_label_noise(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    df = raw_df[(raw_df["distribution_noise"] == 0.0) & (raw_df["sample_size"].astype(str) == "full")]

    for dataset in sorted(df["dataset"].unique()):
        dsub = df[df["dataset"] == dataset]
        summary = dsub.groupby(["model", "label_noise"], as_index=False)["coverage"].mean()

        fig, ax = plt.subplots()
        for model in summary["model"].unique():
            m = summary[summary["model"] == model]
            ax.plot(m["label_noise"], m["coverage"], marker="o", label=model)
        ax.axhline(TARGET_COVERAGE, linestyle="--", color="black", linewidth=1.2, label="target=0.9")
        ax.set_title(f"Figure 3: Coverage vs Label Noise ({dataset})")
        ax.set_xlabel("Training label noise level")
        ax.set_ylabel("Empirical coverage")
        ax.legend(frameon=False)
        _save(fig, out_dir / f"figure3_coverage_vs_label_noise_{dataset}")



def figure4_coverage_vs_sample_size(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    df = raw_df[(raw_df["distribution_noise"] == 0.0) & (raw_df["label_noise"] == 0.0)].copy()

    for dataset in sorted(df["dataset"].unique()):
        dsub = df[df["dataset"] == dataset].copy()
        full_size = dsub.loc[dsub["sample_size"].astype(str) == "full", "sample_size"]
        dsub.loc[dsub["sample_size"].astype(str) == "full", "sample_size_num"] = 999999
        dsub.loc[dsub["sample_size"].astype(str) != "full", "sample_size_num"] = dsub.loc[
            dsub["sample_size"].astype(str) != "full", "sample_size"
        ].astype(int)
        summary = dsub.groupby(["model", "sample_size_num"], as_index=False)["coverage"].mean()

        fig, ax = plt.subplots()
        for model in summary["model"].unique():
            m = summary[summary["model"] == model].sort_values("sample_size_num")
            x_vals = m["sample_size_num"].values
            labels = ["full" if x == 999999 else str(int(x)) for x in x_vals]
            ax.plot(range(len(x_vals)), m["coverage"], marker="o", label=model)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(labels)
        ax.axhline(TARGET_COVERAGE, linestyle="--", color="black", linewidth=1.2, label="target=0.9")
        ax.set_title(f"Figure 4: Coverage vs Sample Size ({dataset})")
        ax.set_xlabel("Training sample size")
        ax.set_ylabel("Empirical coverage")
        ax.legend(frameon=False)
        _save(fig, out_dir / f"figure4_coverage_vs_sample_size_{dataset}")



def figure5_seed_stability_boxplot(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    # Representative setting: highest shift to expose instability.
    df = raw_df[
        (raw_df["distribution_noise"] == 1.0)
        & (raw_df["label_noise"] == 0.0)
        & (raw_df["sample_size"].astype(str) == "full")
    ]

    for dataset in sorted(df["dataset"].unique()):
        dsub = df[df["dataset"] == dataset]
        models = sorted(dsub["model"].unique())
        data = [dsub[dsub["model"] == m]["coverage"].values for m in models]

        fig, ax = plt.subplots()
        ax.boxplot(data, labels=models, patch_artist=False)
        ax.axhline(TARGET_COVERAGE, linestyle="--", color="black", linewidth=1.2)
        ax.set_title(f"Figure 5: Coverage Stability Across Seeds ({dataset}, shift=1.0)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Empirical coverage")
        _save(fig, out_dir / f"figure5_seed_stability_{dataset}")



def figure6_coverage_gap_heatmap(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    # One heatmap per model using distribution-shift settings.
    df = raw_df[(raw_df["label_noise"] == 0.0) & (raw_df["sample_size"].astype(str) == "full")]

    for model in sorted(df["model"].unique()):
        msub = df[df["model"] == model]
        pivot = (
            msub.groupby(["dataset", "distribution_noise"]) ["coverage_gap"].mean().reset_index()
            .pivot(index="dataset", columns="distribution_noise", values="coverage_gap")
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-0.3, vmax=0.3)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_title(f"Figure 6: Coverage Gap Heatmap ({model})")
        ax.set_xlabel("Distribution shift noise level")
        ax.set_ylabel("Dataset")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, label="Coverage gap (empirical - 0.9)")
        _save(fig, out_dir / f"figure6_coverage_gap_heatmap_{model}")



def figure7_failure_case_summary(raw_df: pd.DataFrame, out_dir: Path) -> None:
    _setup_matplotlib()
    summary = (
        raw_df.assign(failure=(raw_df["coverage"] < TARGET_COVERAGE).astype(int))
        .groupby(["dataset", "model"], as_index=False)
        .agg(failure_pct=("failure", "mean"))
    )

    datasets = sorted(summary["dataset"].unique())
    models = sorted(summary["model"].unique())
    x = np.arange(len(datasets))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [
            summary[(summary["dataset"] == ds) & (summary["model"] == model)]["failure_pct"].iloc[0]
            for ds in datasets
        ]
        ax.bar(x + (i - 1) * width, vals, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Failure percentage (coverage < 0.9)")
    ax.set_title("Figure 7: Failure Case Summary Across Settings")
    ax.legend(frameon=False)
    _save(fig, out_dir / "figure7_failure_case_summary")



def generate_all_figures(raw_df: pd.DataFrame, out_dir: Path) -> None:
    figure1_coverage_vs_shift(raw_df, out_dir)
    figure2_setsize_vs_shift(raw_df, out_dir)
    figure3_coverage_vs_label_noise(raw_df, out_dir)
    figure4_coverage_vs_sample_size(raw_df, out_dir)
    figure5_seed_stability_boxplot(raw_df, out_dir)
    figure6_coverage_gap_heatmap(raw_df, out_dir)
    figure7_failure_case_summary(raw_df, out_dir)
