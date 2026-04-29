#!/usr/bin/env python

import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from inmoose.pycombat import pycombat_norm
from scipy.stats import shapiro, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

BASE_DIR = pl.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CYTOKINES = ["INFgama", "IL1beta", "IL6", "IL8", "IL12 p70", "TNFalpha"]
STIMULI = ["RPMI", "LPS", "Bexcero", "HBVaxpro", "Poly_C"]

# Límites oficiales Bio-Plex Pro Human (Bulletin 5803)
LIMITS = {
    "INFgama": {"low": 46.3, "high": 52719},
    "IL1beta": {"low": 1.6, "high": 3261},
    "IL6": {"low": 1.15, "high": 18880},
    "IL8": {"low": 0.95, "high": 26403},
    "IL12 p70": {"low": 1.65, "high": 13099},
    "TNFalpha": {"low": 3.0, "high": 19531},
}



def load_and_clean_data(path_conc, path_count, path_meta):
    conc = pd.read_csv(path_conc, index_col=0).drop(
        columns=["Location"], errors="ignore"
    )
    count = pd.read_csv(path_count, index_col=0).drop(
        columns=["Location"], errors="ignore"
    )
    meta = pd.read_csv(path_meta)

    for cyto, val in LIMITS.items():
        col = conc[cyto].astype(str)

        col = col.mask(col.str.contains("<", na=False), val["low"])
        col = col.mask(col.str.contains(">", na=False), val["high"])

        col = pd.to_numeric(col, errors="coerce")

        # Filtro de calidad
        col = col.mask(count[cyto] <= 35, np.nan)

        conc[cyto] = col

    # Media de las réplicas
    conc = conc.groupby(level=0)[CYTOKINES].mean().dropna(how="all")
    merged = pd.merge(meta, conc, left_on="Codigo_citoquinas", right_index=True)
    merged = merged[merged["Estimulacion"].isin(STIMULI)].copy()

    # Se pasa a formato ancho
    merged["ID_corrida"] = merged["Codigo_citoquinas"].str.rsplit("_", n=1).str[0]
    wide_df = merged.pivot(index="ID_corrida", columns="Estimulacion", values=CYTOKINES)
    wide_df.columns = [f"{cyto}_{stim}" for cyto, stim in wide_df.columns]

    meta_info = (
        merged[["ID_corrida", "Batch", "Vacunado_Placebo"]]
        .drop_duplicates()
        .set_index("ID_corrida")
    )

    return pd.concat([meta_info, wide_df], axis=1)


def normalize_data(df):
    meta_cols = ["Batch", "Vacunado_Placebo"]
    batch_labels = df["Batch"].astype(str)
    numeric_data = df.drop(columns=meta_cols)

    # Log2 e Imputación
    data_log = np.log2(numeric_data.replace(0, np.nan))
    data_imputed = data_log.groupby(df["Vacunado_Placebo"]).transform(
        lambda x: x.fillna(x.median())
    )

    df_pre = pd.concat([df[meta_cols], data_imputed], axis=1)

    # ComBat
    corrected_mat = pycombat_norm(data_imputed.T, batch=batch_labels).T
    df_post = pd.concat([df[meta_cols], corrected_mat], axis=1)

    return df_pre, df_post

def plot_pca(df, title="Análisis PCA"):
    features = df.drop(columns=["Batch", "Vacunado_Placebo"])
    x = StandardScaler().fit_transform(features.fillna(0))

    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    var = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=df.index)
    pca_df = pd.concat([pca_df, df[["Batch", "Vacunado_Placebo"]]], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)

    for i, col in enumerate(["Batch", "Vacunado_Placebo"]):
        sns.scatterplot(ax=axes[i], data=pca_df, x="PC1", y="PC2", hue=col, s=80)
        axes[i].set_title(f"Agrupación por {col}")
        axes[i].set_xlabel(f"PC1 ({var[0]:.1f}%)")
        axes[i].set_ylabel(f"PC2 ({var[1]:.1f}%)")

    plt.tight_layout()
    plt.show()


def compute_statistics(df):

    stats_rpmi = []
    for group in df["Vacunado_Placebo"].unique():
        df_g = df[df["Vacunado_Placebo"] == group]
        for stim in [s for s in STIMULI if s != "RPMI"]:
            for cyto in CYTOKINES:
                c_stim, c_rpmi = f"{cyto}_{stim}", f"{cyto}_RPMI"
                data = df_g[[c_stim, c_rpmi]].dropna()
                if len(data) > 5:
                    # Test de Shapiro-Wilks para comprobar la normalidad de los datos
                    diff = data[c_stim] - data[c_rpmi]
                    _, p_norm = shapiro(diff)
                    
                    if p_norm > 0.05:
                        _, p = ttest_rel(data[c_stim], data[c_rpmi])
                    else:
                        _, p = wilcoxon(data[c_stim], data[c_rpmi])
                        
                    stats_rpmi.append(
                        {
                            "Group": group,
                            "Stimulus": stim,
                            "Cytokine": cyto,
                            "Log2FC": diff.mean(),
                            "p_val": p,
                        }
                    )

    res_rpmi = pd.DataFrame(stats_rpmi)
    if not res_rpmi.empty:
        res_rpmi["p_adj"] = multipletests(res_rpmi["p_val"], method="fdr_bh")[1]

    stats_net = []

    for stim in [s for s in STIMULI if s != "RPMI"]:
        for cyto in CYTOKINES:
            c_stim, c_rpmi = f"{cyto}_{stim}", f"{cyto}_RPMI"

            # Delta Citoquin~Estimulo - Basal
            net = df[c_stim] - df[c_rpmi]
            v_net = net[df["Vacunado_Placebo"] == "Vacunado"].dropna()
            p_net = net[df["Vacunado_Placebo"] == "Placebo"].dropna()

            if len(v_net) > 3 and len(p_net) > 3:

                _, p_norm_v = shapiro(v_net)
                _, p_norm_p = shapiro(p_net)
                
                if p_norm_v > 0.05 and p_norm_p > 0.05:
                    _, p_val = ttest_ind(v_net, p_net)
                else:
                    _, p_val = mannwhitneyu(v_net, p_net)
                    
                stats_net.append(
                    {
                        "Stimulus": stim,
                        "Cytokine": cyto,
                        "Diff_Median_Delta": v_net.median() - p_net.median(),
                        "Median_Vac": v_net.median(),
                        "Median_Pla": p_net.median(),
                        "p_val": p_val,
                    }
                )


    res_net = pd.DataFrame(stats_net)
    if not res_net.empty:
        res_net["p_adj"] = multipletests(res_net["p_val"], method="fdr_bh")[1]


    return res_rpmi, res_net


def plot_fold_change(res_df):
    if res_df.empty:
        return

    df_plot_base = res_df.copy()

    df_plot_base["FC_Vac"] = 2 ** df_plot_base["Median_Vac"]
    df_plot_base["FC_Pla"] = 2 ** df_plot_base["Median_Pla"]

    # Se ordena por estimulo y luego por citoquina
    df_plot_base = df_plot_base.sort_values(
        ["Stimulus", "Cytokine"], ascending=[True, True]
    )
    
    df_plot_base["Unique_ID"] = (
        df_plot_base["Stimulus"] + "_" + df_plot_base["Cytokine"]
    )

    plot_data = df_plot_base.melt(
        id_vars=["Stimulus", "Cytokine", "Unique_ID", "p_val"],
        value_vars=["FC_Vac", "FC_Pla"],
        var_name="Grupo",
        value_name="FoldChange",
    )

    plt.figure(figsize=(20, 8))

    ax = sns.barplot(
        data=plot_data,
        x="Unique_ID",
        y="FoldChange",
        hue="Grupo",
        palette=["#FF0000", "#000000"],
        edgecolor="black",
        dodge=True,
    )

    plt.axhline(1, color="black", linewidth=2, zorder=1)

    current_pos = 0
    stimuli_list = df_plot_base["Stimulus"].unique()

    y_max_data = plot_data["FoldChange"].max()
    ax.set_ylim(plot_data["FoldChange"].min() * 0.8, y_max_data * 4)

    for i, stim in enumerate(stimuli_list):
        n_cytos = len(df_plot_base[df_plot_base["Stimulus"] == stim])

        if i < len(stimuli_list) - 1:
            plt.axvline(
                current_pos + n_cytos - 0.5, color="gray", linestyle="--", alpha=0.5
            )

        center = current_pos + (n_cytos / 2) - 0.5
        plt.text(
            center,
            ax.get_ylim()[1] * 0.85,
            stim,
            ha="center",
            va="top",
            weight="bold",
            fontsize=12,
            color="black",
        )

        current_pos += n_cytos

    for i, (idx, row) in enumerate(df_plot_base.iterrows()):
        p = row["p_val"]
        if p < 0.05:
            y_val = max(row["FC_Vac"], row["FC_Pla"])
            star = "*" if p > 0.01 else "**" if p > 0.001 else "***"
            plt.text(i, y_val * 1.2, star, ha="center", fontsize=18, fontweight="bold")

    plt.xticks(
        range(len(df_plot_base)), df_plot_base["Cytokine"], rotation=45, ha="right"
    )

    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles[0:2], ["VMI", "PMI"], frameon=False, loc="upper right")

    plt.yscale("log", base=2)
    plt.ylabel("Fold Change", labelpad=15, fontsize=12, fontweight='bold')
    plt.xlabel("")

    plt.legend(handles[0:2], ["VMI", "PMI"], 
            frameon=False, 
            loc="upper right",
            bbox_to_anchor=(1, 1))

    sns.despine()
    plt.tight_layout(rect=[0.01, 0, 1, 1])
    plt.show()


if __name__ == "__main__":
    df_raw = load_and_clean_data(
        DATA_DIR / "concentration.csv",
        DATA_DIR / "count.csv",
        DATA_DIR / "metadata.csv",
    )

    df_imputed, df_norm = normalize_data(df_raw)

    res_rpmi, res_net= compute_statistics(df_norm)

    print("Resultados significativos vs RPMI:")
    print(res_rpmi[res_rpmi["p_val"] < 0.05].sort_values(["Group", "p_val"]))

    print("\nResultados significativos Vacunado vs Placebo NETO:")
    print(res_net[res_net["p_val"] < 0.05].sort_values(["Stimulus", "p_val"]))


    plot_fold_change(res_net)
