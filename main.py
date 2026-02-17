#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from statsmodels.stats.multitest import multipletests


def _load_data(path_to_areas_csv, path_to_metadata_csv):
    # Load data
    areas = pd.read_csv(path_to_areas_csv, index_col=0)
    metadata = pd.read_csv(path_to_metadata_csv, index_col=0)

    # Drop useless values
    areas = areas.iloc[3:]

    # Transpose so proteins are columns
    areas = areas.T

    # Ensure metadata values are strings
    metadata = metadata.astype(str)

    # Create extended phenotype column
    metadata["Pheno_ext"] = (
        metadata["Estimulacion"] + "-" + metadata["Vacunado_Placebo"]
    )

    # Simplify protein names to just the UniProt accession
    areas.columns = areas.columns.str.extract(r"^([^_]+)_")[0]

    # Ensure areas are numeric
    areas = areas.apply(pd.to_numeric, errors="coerce")

    # Ensure index are the same
    areas = areas.loc[metadata.index]

    return areas, metadata


def _normalize(areas):
    # Normalize areas using Quantile Transformer
    return pd.DataFrame(
        QuantileTransformer(
            n_quantiles=min(1000, areas.shape[0]),
            output_distribution="normal",
            random_state=42,
        ).fit_transform(areas),
        index=areas.index,
        columns=areas.columns,
    )


def _compute_relative_differences(areas, metadata, stimulus):
    # Get baseline samples for each patient
    baseline_mask = metadata["Estimulacion"] == stimulus
    baseline_indices = []
    for _, group in metadata[baseline_mask].groupby("ID_paciente"):
        # Pick the first baseline sample for each patient
        baseline_indices.append(group.index[0])
    baseline_values = areas.loc[baseline_indices]

    # Compute % zeros per protein in baseline samples
    zero_proportions = (baseline_values == 0).sum() / baseline_values.shape[0]

    # Keep proteins with <= 20% zeros
    valid_proteins = zero_proportions[zero_proportions <= 0.2].index
    areas = areas[valid_proteins]

    # Add 1 to avoid zeros
    areas = areas + 1

    # Compute relative differences
    relative_diffs = []
    for _, patient_data in metadata.groupby("ID_paciente"):
        baseline_indices = patient_data[patient_data["Estimulacion"] == stimulus].index
        if len(baseline_indices) == 0:
            continue  # Skip this patient if no baseline sample
        baseline_idx = baseline_indices[0]
        baseline_sample = areas.loc[baseline_idx]

        # Get non-baseline samples for this patient
        other_samples = areas.loc[patient_data.index.difference([baseline_idx])]

        # Compute relative differences and handle division by zero
        diffs = (other_samples - baseline_sample) / baseline_sample
        relative_diffs.append(diffs)

    return pd.concat(relative_diffs)


def _compute_log2fc(areas, metadata, stimulus):
    # Use ΔΔ method to compute the log2 fold change

    # First, compute the median and standard deviation of placebo samples in
    # RPMI condition and in stimulated condition
    placebo_mask = (metadata["Estimulacion"] == "RPMI") & (
        metadata["Vacunado_Placebo"] == "Placebo"
    )
    placebo_medians = areas_normalized.loc[placebo_mask].median()
    placebo_stds = areas_normalized.loc[placebo_mask].std()
    placebo_stimulated_mask = (metadata["Vacunado_Placebo"] == "Placebo") & (
        metadata["Estimulacion"] == stimulus
    )
    placebo_stimulated_medians = areas_normalized.loc[placebo_stimulated_mask].median()
    placebo_stimulated_stds = areas_normalized.loc[placebo_stimulated_mask].std()

    # Then, compute the median and standard deviation of vaccinated samples in
    # RPMI condition and in stimulated condition
    vaccinated_mask = (metadata["Estimulacion"] == "RPMI") & (
        metadata["Vacunado_Placebo"] == "Vacunado"
    )
    vaccinated_medians = areas_normalized.loc[vaccinated_mask].median()
    vaccinated_stds = areas_normalized.loc[vaccinated_mask].std()
    vaccinated_stimulated_mask = (metadata["Vacunado_Placebo"] == "Vacunado") & (
        metadata["Estimulacion"] == stimulus
    )
    vaccinated_stimulated_medians = areas_normalized.loc[
        vaccinated_stimulated_mask
    ].median()
    vaccinated_stimulated_stds = areas_normalized.loc[vaccinated_stimulated_mask].std()

    # Third, compute the differences and their standard deviations
    delta_vaccinated_minus_placebo = vaccinated_medians - placebo_medians
    delta_vaccinated_minus_placebo_stds = np.sqrt(vaccinated_stds**2 + placebo_stds**2)
    delta_vaccinated_minus_placebo_stimulated = (
        vaccinated_stimulated_medians - placebo_stimulated_medians
    )
    delta_vaccinated_minus_placebo_stimulated_stds = np.sqrt(
        vaccinated_stimulated_stds**2 + placebo_stimulated_stds**2
    )

    # Fourth, compute the ΔΔ (logFC) and its standard deviation
    delta_delta = (
        delta_vaccinated_minus_placebo_stimulated - delta_vaccinated_minus_placebo
    )
    _delta_delta_stds = np.sqrt(
        delta_vaccinated_minus_placebo_stds**2
        + delta_vaccinated_minus_placebo_stimulated_stds**2
    )

    # And last, generate dataframe with log2FC
    return pd.DataFrame(
        {
            "UniProt": areas.columns,
            "log2FC": delta_delta,
        },
    )


def _compute_pvalues(areas, metadata, stimulus):
    # Transpose areas and select samples from stimulus
    stimulus_mask = metadata["Estimulacion"] == stimulus
    areas = areas.T.loc[:, metadata.index[stimulus_mask]]

    # Get groups
    groups = metadata.loc[stimulus_mask, "Pheno_ext"]
    group1, group2 = np.sort(groups.unique())
    print(group1, group2)

    expr_1 = areas.loc[:, groups == group2].astype(float)  # Vaccinated
    expr_2 = areas.loc[:, groups == group1].astype(float)  # Placebo

    # Apply Saphiro test per row (protein)
    pvals_norm_vaccinated = expr_1.apply(
        lambda row: shapiro(pd.to_numeric(row, errors="coerce").dropna())[1], axis=1
    )

    pvals_norm_placebo = expr_2.apply(
        lambda row: shapiro(pd.to_numeric(row, errors="coerce").dropna())[1],axis=1
    )

    # Combinar los resultados en un DataFrame
    pvals_normality = pd.DataFrame({
        "pval_norm_vaccinated": pvals_norm_vaccinated,
        "pval_norm_placebo": pvals_norm_placebo
    })

    # Compute p-values based on normality test
    def choose_test(pval_vac, pval_pla, expr1, expr2):
        if (pval_vac < 0.05) or (pval_pla < 0.05):
            return mannwhitneyu(expr1, expr2, alternative="two-sided")[1]
        else:
            return ttest_ind(expr1, expr2, nan_policy="omit", equal_var=False)[1]

    pvals = pd.Series(
        [
            choose_test(pval_vac, pval_pla, e1, e2)
            for pval_vac, pval_pla, e1, e2 in zip(
                pvals_norm_vaccinated, pvals_norm_placebo, expr_1.values, expr_2.values
            )
        ],
        index=areas.index,
        )

    # Adjust p-values
    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    # Create results DataFrame
    return pd.DataFrame(
        {
            "UniProt": areas.index,
            "p-value": pvals,
            "p-value_adjusted": pvals_adj,
        }
    )


def _plot_volcano(results, stimulus, output_dir):
    # Extract logFC and p-values
    logFC, pvals = results["log2FC"], results["p-value"]

    # Determine significance
    significance = (pvals < 0.05) & (abs(logFC) > 0.5)

    # Create volcano plot
    plt.figure(figsize=(8, 10))
    plt.scatter(
        logFC[~significance],
        -np.log10(pvals[~significance]),
        color="#3b528b",
        alpha=0.25,
        s=150,
    )
    plt.scatter(
        logFC[significance],
        -np.log10(pvals[significance]),
        color="#5ec962",
        alpha=0.75,
        s=150,
    )

    # LaTeX-style axis labels
    plt.xlabel(r"$\log_{2}(\mathrm{Fold\ Change})$", fontsize=40)
    plt.ylabel(r"$-\log_{10}(p\ \mathrm{value})$", fontsize=40)

    # Significance threshold lines
    y_thresh = -np.log10(0.05)
    plt.axhline(y_thresh, color="gray", linestyle="--")
    plt.axvline(0.5, color="gray", linestyle="--")
    plt.axvline(-0.5, color="gray", linestyle="--")

    # Only tick at significance thresholds
    plt.xticks([-0.5, 0.5], labels=["-0.5", "0.5"])
    plt.yticks([y_thresh], labels=[f"{y_thresh:.2f}"])

    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, "volcano_{0}.png".format(stimulus)),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def _plot_pca(areas, metadata, results, stimulus, output_dir):
    # Filter metadata and area based on stimulus
    metadata_filtered = metadata[metadata["Estimulacion"] == stimulus].copy()
    areas_filtered = areas.loc[areas.index.intersection(metadata_filtered.index)].copy()

    # Select significantly different proteins
    diff_proteins = results[
        (results["p-value"] < 0.05) & (results["log2FC"].abs() > 1)
    ]["UniProt"]

    # Filter areas dataframe
    areas_filtered = areas_filtered.loc[
        :, areas_filtered.columns.intersection(diff_proteins)
    ]

    # PCA
    pca = PCA(n_components=2)
    areas_pca = pca.fit_transform(areas_filtered)

    # Label groups
    metadata_filtered["Vacunado_Placebo"] = metadata_filtered[
        "Vacunado_Placebo"
    ].replace({"Vacunado": "VMI", "Placebo": "PMI"})
    metadata_filtered["Cond-Stimulus"] = (
        metadata_filtered["Estimulacion"].astype(str)
        + "-"
        + metadata_filtered["Vacunado_Placebo"].astype(str)
    )

    # Group colors
    group_colors = {
        f"{stimulus}-VMI": "#9847B8FB",
        f"{stimulus}-PMI": "#F07D1F",
    }

    sample_colors = metadata_filtered.loc[areas_filtered.index, "Cond-Stimulus"].map(
        group_colors
    )

    # PCA plot
    fig, ax = plt.subplots(figsize=(8, 10))

    ax.scatter(
        areas_pca[:, 0],
        areas_pca[:, 1],
        c=sample_colors,
        edgecolor="k",
        s=350,
        alpha=0.9,
    )

    # LaTeX-style axis labels (matching volcano)
    ax.set_xlabel(
        rf"$\mathrm{{PC1}}\ (\mathrm{{{pca.explained_variance_ratio_[0] * 100:.1f}\%}})$",
        fontsize=40,
    )
    ax.set_ylabel(
        rf"$\mathrm{{PC2}}\ (\mathrm{{{pca.explained_variance_ratio_[1] * 100:.1f}\%}})$",
        fontsize=40,
    )

    # Remove ticks for consistency with volcano
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, f"pca_{stimulus}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_heatmap(areas, metadata, results, stimulus, output_dir):
    # Filter by stimulus
    metadata_filtered = metadata[metadata["Estimulacion"] == stimulus].copy()
    metadata_filtered = metadata_filtered.sort_values(
        "Vacunado_Placebo", ascending=False
    )
    areas_filtered = areas.loc[metadata_filtered.index].copy()

    # Select differentially expressed proteins
    diff_proteins = results[
        (results["p-value"] < 0.05) & (results["log2FC"].abs() > 1)
    ]["UniProt"]

    # Filter areas dataframe
    diff_areas = areas_filtered.loc[
        :, areas_filtered.columns.intersection(diff_proteins)
    ]

    # Generate Stimulus-HMV/HMP condition
    metadata_filtered["Vacunado_Placebo"] = metadata_filtered[
        "Vacunado_Placebo"
    ].replace({"Vacunado": "VMI", "Placebo": "PMI"})

    # Assign colors
    group_colors = {"VMI": "#9847B8FB", "PMI": "#F07D1F"}
    col_colors = metadata_filtered["Vacunado_Placebo"].map(group_colors).tolist()

    # Generate heatmap
    g = sns.clustermap(
        diff_areas.transpose(),
        cmap="viridis",
        vmin=-2,
        vmax=2,
        center=0,
        linewidths=0.5,
        linecolor="black",
        yticklabels=True,
        xticklabels=False,
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=True,
        figsize=(8, 10),
        z_score=1,
        dendrogram_ratio=(0.1, 0.05),
        colors_ratio=(0.02, 0.02),
    )
    g.figure.subplots_adjust(top=1, bottom=0, left=0, right=1)

    # Access main heatmap axis
    ax = g.ax_heatmap

    # Remove all ticks and labels
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
 
    # Remove any automatically set axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove colorbar, titles, and legends if any
    if g.cax is not None:
        g.cax.remove()

    for sub_ax in [g.ax_col_dendrogram, g.ax_row_dendrogram, g.ax_heatmap]:
        legend = sub_ax.get_legend()
        if legend is not None:
            legend.remove()
        sub_ax.set_title("")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, f"heatmap_{stimulus}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    areas, metadata = _load_data(
        (script_dir / "data" / "PHL_areas_MLR.csv"),
        (script_dir / "data" / "metadata_boostrix.csv")

    )

    relative_diffs = _compute_relative_differences(areas, metadata, "RPMI")
    relative_diffs_norm = _normalize(relative_diffs)

    output_dir = script_dir / "output"
    for stimulus in [
        "Bexcero",
        "LPS",
        "Poly_C",
        "Varivax",
        "Diftavax",
        "BCG",
        "Boostrix",
        "HBVaxpro",
        "candida",
    ]:
        # Normalize areas and compute log2FC and p-values
        areas_normalized = _normalize(areas)
        log2fc = _compute_log2fc(areas_normalized, metadata, stimulus)
        pvalues = _compute_pvalues(relative_diffs_norm, metadata, stimulus)

        # Merge results
        results = pd.merge(log2fc, pvalues, on="UniProt", how="inner")
        results = results[["UniProt", "log2FC", "p-value"]]

        # Plot data into a volcano plot, a PCA and a heatmap
        _plot_volcano(results, stimulus, output_dir)
        _plot_pca(areas_normalized, metadata, results, stimulus, output_dir)
        _plot_heatmap(relative_diffs_norm, metadata, results, stimulus, output_dir)
