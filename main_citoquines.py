#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
import subprocess

from pathlib import Path

def _load_data(filepath):
    start = False
    data = []

    with open(filepath) as f:
        for line in f:
            if start:
                if line.strip() == "":
                    break
                data.append(line)
            if line.strip() == '"DataType:","Avg Result"':
                start = True

    # Join lines and read as csv
    csv_text = "".join(data)
    result = pd.read_csv(pd.io.common.StringIO(csv_text), index_col=0)
    return result

def clean_value(x):
    # Convert NaN a 0
    if pd.isna(x):
        return 0.0

    if isinstance(x, str):
        x = x.strip()
        # Values less than the limit -> half the limit
        if re.match(r"^<\s*\d+(\.\d+)?$", x):
            return float(x.replace("<", "").strip()) / 2
        # Values greater than the limit -> limit
        elif re.match(r"^>\s*\d+(\.\d+)?$", x):
            return float(x.replace(">", "").strip())
    return x

def _join_data(df1, df2, df3, metadata):
    # Add Plaque column
    df1["Plaque"] = 1
    df2["Plaque"] = 2
    df3["Plaque"] = 3

     # Join citoquines dataframe
    citoquines = pd.concat([df1, df2, df3], axis=0)

    # Ensure index are strings
    metadata["Codigo_citoquinas"] = metadata["Codigo_citoquinas"].astype(str).str.strip()
    cytokines.index = cytokines.index.astype(str).str.strip()

    common_index = cytokines.index.intersection(metadata["Codigo_citoquinas"])
  
    # Reorder both dataframes to have the same order
    citoquines = citoquines.loc[common_index]
    metadata_filter = metadata.set_index("Codigo_citoquinas").loc[common_index]

    columns_add = ["Vacunado_Placebo", "código lactante", "Estimulacion"]

    # Add columns to cytokynes dataframe
    for col in columns_add:
        citoquines[col] = metadata_filter[col].values

    citoquines = citoquines.applymap(clean_value)
    citoquines = citoquines.apply(pd.to_numeric, errors="ignore")

    return citoquines

def _plot_pca(cytokynes, color_col="Batch"):
    
    # Select the numerical columns (cytokines)
    cytokine_data = cytokynes.iloc[:, :13].copy()

    # Impute NA values with mean value for each column (cytokine)
    cytokine_data = cytokine_data.fillna(cytokine_data.mean())

    # Scale the data (mean=0, var=1)
    scaler = StandardScaler()
    cytokine_scaled = scaler.fit_transform(cytokine_data)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cytokine_scaled)

    # DataFrame with results
    pca_df = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    color_col: cytokynes[color_col].values
    }, index=cytokynes.index)

    # Group colors
    custom_colors = ["#FFD700", "#FF4500", "#32CD32", "#1E90FF", "#9370DB"]
    unique_batches = pca_df[color_col].unique()
    color_map = {batch: custom_colors[i % len(custom_colors)] for i, batch in enumerate(unique_batches)}

    # PCA plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for batch in unique_batches:
        subset = pca_df[pca_df[color_col] == batch]
        ax.scatter(
            subset["PC1"], subset["PC2"],
            color=color_map[batch],
            label=batch,
            s=120,
            edgecolor="k",
            alpha=0.85
        )

    # LaTeX-style axis labels (matching volcano)
    ax.set_xlabel(
        rf"$\mathrm{{PC1}}\ (\mathrm{{{pca.explained_variance_ratio_[0] * 100:.1f}\%}})$",fontsize=16,
    )
    ax.set_ylabel(rf"$\mathrm{{PC2}}\ (\mathrm{{{pca.explained_variance_ratio_[1] * 100:.1f}\%}})$",fontsize=16,
    )

    ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join("PCA_cytokynes_batch.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _normalize(cytokines):
    # Normalize areas using Quantile Transformer
    cytokines = cytokines.copy()  # para no modificar el original

    # Seleccionamos solo las columnas numéricas (las 13 primeras)
    numeric_cols = cytokines.columns[:13]

    # Aplicamos la normalización
    qt = QuantileTransformer(
        n_quantiles=min(1000, cytokines.shape[0]),
        output_distribution="normal",
        random_state=42
    )

    cytokines[numeric_cols] = qt.fit_transform(cytokines[numeric_cols])

    return cytokines

def _compute_pvalues(cytokines, metadata, stimulus):

    stimulus_mask = metadata["Estimulacion"] == stimulus
    cytokines_selected = cytokines[cytokines_combat["Estimulacion"] == stimulus]

    cols_expr = cytokines_selected.select_dtypes(include=np.number).columns
    expr = cytokines_selected[cols_expr]

    # Get groups
    groups = metadata.loc[stimulus_mask, "Vacunado_Placebo"]
    group1, group2 = np.sort(groups.unique())
    expr = expr.T

    expr_1 = expr.loc[:, groups == group2].astype(float)  # Vaccinated
    expr_2 = expr.loc[:, groups == group1].astype(float)  # Placebo

    # Apply Saphiro test per row (protein)
    pvals_norm_vaccinated = expr_1.apply(
        lambda row: stats.shapiro(pd.to_numeric(row, errors="coerce").dropna())[1], axis=1
    )

    pvals_norm_placebo = expr_2.apply(
        lambda row: stats.shapiro(pd.to_numeric(row, errors="coerce").dropna())[1],axis=1
    )

    # Combinar los resultados en un DataFrame
    pvals_normality = pd.DataFrame({
        "pval_norm_vaccinated": pvals_norm_vaccinated,
        "pval_norm_placebo": pvals_norm_placebo
    })

    # Compute p-values based on normality test
    def choose_test(pval_vac, pval_pla, expr1, expr2):
        if (pval_vac < 0.05) or (pval_pla < 0.05):
            return stats.mannwhitneyu(expr1, expr2, alternative="two-sided")[1]
        else:
            return stats.ttest_ind(expr1, expr2, nan_policy="omit", equal_var=False)[1]

    pvals = pd.Series(
        [
            choose_test(pval_vac, pval_pla, e1, e2)
            for pval_vac, pval_pla, e1, e2 in zip(
                pvals_norm_vaccinated, pvals_norm_placebo, expr_1.values, expr_2.values
            )
        ],
        index=expr.index,
        )


    # Adjust p-values
    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    # Create results DataFrame
    return pd.DataFrame(
        {
            "Cytokine": expr.index,
            "p-value": pvals,
            "p-value_adjusted": pvals_adj,
        }
    )

if __name__== "__main__":

    script_dir = Path(__file__).parent

    df1 = _load_data(script_dir / "data" / "New_Batch_22.csv")
    df2 = _load_data(script_dir / "data" / "New_Batch_23.csv")
    df3 = _load_data(script_dir / "data" / "New_Batch_25.csv")
    metadata = pd.read_csv(script_dir / "data" / "metadata_boostrix_modificado.csv", index_col=0, dtype=str)

    cytokines = _join_data(df1, df2, df3, metadata)

    # Normalize the input data
    cytokines_norm = _normalize(cytokines)

    # Correct batch effect
    cytokines_norm.to_csv(script_dir / "data" / "cytokines_normalized.csv", index=True)

    input_file = script_dir / "data" / "cytokines_normalized.csv"
    output_file = script_dir / "data" / "cytokines_combat.csv"
    
    subprocess.run(
    ["Rscript", "correct_batch_effect.R", str(input_file), str(output_file)],
    check=True
    )

    cytokines_combat = pd.read_csv(output_file, index_col=0)

    # Ensure Batch is str and not a numerical column
    cytokines_combat["Batch"] = cytokines_combat["Batch"].astype(str)


    for stimulus in [
        "RPMI",
        "Bexcero",
        "LPS",
        "Poly_C",
        "HBVaxpro"
    ]:
        pvalues = _compute_pvalues(cytokines_combat, metadata, stimulus)
        print(pvalues)
        #pvalues.to_csv("output" / f"pval_cytokines_{stimulus}.txt", index=True)
