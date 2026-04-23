#!/usr/bin/env python3

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pathlib as pl

from inmoose.pycombat import pycombat_norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


STIMULI = ["RPMI", "LPS", "Bexcero", "HBVaxpro", "Poly_C"]


def _find(predicate, container, start=0):
    for i, item in enumerate(container[start:], start):
        if predicate(item):
            return i
    return None


def _read_table(filepath, header):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    start = _find(lambda line: line.strip() == header, lines)
    assert start is not None, f"Header '{header}' no encontrado en {filepath}"
    end = _find(lambda line: line.strip() == "", lines, start)
    assert end is not None, f"Fin de tabla no encontrado después de {header}"

    df = pd.read_csv(
        pd.io.common.StringIO("".join(lines[start + 1 : end])),
        index_col=0,
        dtype=str,
    )
    df.index = df.index.astype(str).str.strip()
    return df


def _load_data(filepath):
    count = _read_table(filepath, '"DataType:","Count"')
    result = _read_table(filepath, '"DataType:","Obs Concentration"')
    return count, result


def _join_data(*dfs):
    return pd.concat(dfs, axis=0, verify_integrity=False)


def _remove_calibration_values_location_and_events(df):
    pattern = r"^(Background|Standard)\d*$"
    df = df[~df.index.str.match(pattern, na=False)].copy()
    df = df.drop(columns=["Location"], errors="ignore")
    return df


def _process_replicates(df):
    data_dict = {}
    for cytokine in df.columns:
        long = df[cytokine].reset_index()
        long.columns = ["sample", "value"]
        grouped = long.groupby("sample")["value"].apply(list)

        rows = []
        for sample, vals in grouped.items():
            rep1 = vals[0] if len(vals) > 0 else np.nan
            rep2 = vals[1] if len(vals) > 1 else np.nan
            rows.append({"sample": sample, "replicate_1": rep1, "replicate_2": rep2})

        df_cyt = pd.DataFrame(rows).set_index("sample")
        data_dict[cytokine] = df_cyt
    return data_dict


def _duplicate_replicate_if_any_is_nan(data_dict):
    for _, df in data_dict.items():
        for idx in df.index:
            r1, r2 = df.loc[idx, ["replicate_1", "replicate_2"]]
            if pd.isna(r1) and not pd.isna(r2):
                df.loc[idx, "replicate_1"] = r2
            elif pd.isna(r2) and not pd.isna(r1):
                df.loc[idx, "replicate_2"] = r1
    return data_dict


def _clamp_value(x):
    if isinstance(x, str):
        x = x.strip()
        x = x.replace(",", ".")
        if re.match(r"^<\s*\d", x):
            val = float(re.sub(r"^<\s*", "", x))
            return val * 0.5
        elif re.match(r"^>\s*\d", x):
            val = float(re.sub(r"^>\s*", "", x))
            return val * 2.0
        elif re.match(r"^\*\d*", x):
            return float(re.sub(r"^\*\s*", "", x))
        elif re.match(r"^OOR <", x):
            return 0.5
        elif re.match(r"^OOR >", x):
            return 10000.0
    if isinstance(x, (int, float)) and x <= 0:
        return 0.5
    return x


def _clamp_values(data_dict):
    for _, df in data_dict.items():
        df["replicate_1"] = df["replicate_1"].map(_clamp_value)
        df["replicate_2"] = df["replicate_2"].map(_clamp_value)
    return data_dict


def _convert_to_numeric(data_dict):
    for _, df in data_dict.items():
        df["replicate_1"] = pd.to_numeric(df["replicate_1"], errors="coerce")
        df["replicate_2"] = pd.to_numeric(df["replicate_2"], errors="coerce")
    return data_dict


def plot_pca(
    data,
    metadata,
    title="PCA",
    filename="pca.png",
    output_dir="results/pca",
    hue="Batch",
):
    os.makedirs(output_dir, exist_ok=True)

    # Si metadata no se pasa, intentamos extraerla de data
    if metadata is None:
        meta = data.copy()
        data = data.select_dtypes(include=[np.number])  # solo columnas numéricas
    else:
        meta = metadata.copy()

    # Asegurar que los índices coincidan
    common_idx = data.index.intersection(meta.index)
    data = data.loc[common_idx].copy()
    meta = meta.loc[common_idx].copy()

    print(
        f"→ PCA: {data.shape[0]} muestras × {data.shape[1]} citoquinas | Título: {title}"
    )

    # Escalar
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(data_scaled)

    pca_df = pd.DataFrame(pca_coords, index=data.index, columns=["PC1", "PC2", "PC3"])
    pca_df = pca_df.join(meta)

    # ====================== PLOT ======================
    plt.figure(figsize=(12, 9))
    color_map = {"1": "#21918c", "2": "#fd7825", "3": "#fde725"}

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=hue,
        palette=color_map,
        s=200,
        edgecolor="black",
        alpha=0.95,
        linewidth=1.2,
    )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    plt.title(title, fontsize=20, pad=20)
    plt.xlabel(f"PC1 ({var1:.1f}%)", fontsize=16)
    plt.ylabel(f"PC2 ({var2:.1f}%)", fontsize=16)

    plt.legend(title="Batch", title_fontsize=13, fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"   Guardado: {filename}")
    print(f"   Varianza explicada → PC1: {var1:.1f}% | PC2: {var2:.1f}%")

    return pca_df


if __name__ == "__main__":
    script_dir = pl.Path(__file__).parent

    count_1, result_1 = _load_data(script_dir / "data" / "citokines_1.csv")
    count_2, result_2 = _load_data(script_dir / "data" / "citokines_2.csv")
    count_3, result_3 = _load_data(script_dir / "data" / "citokines_3.csv")

    count = _join_data(count_1, count_2, count_3)
    result = _join_data(result_1, result_2, result_3)

    count = _remove_calibration_values_location_and_events(count)
    result = _remove_calibration_values_location_and_events(result)

    count_dict = _process_replicates(count)
    result_dict = _process_replicates(result)

    count_dict = _duplicate_replicate_if_any_is_nan(count_dict)
    result_dict = _duplicate_replicate_if_any_is_nan(result_dict)

    count_dict = _clamp_values(count_dict)
    result_dict = _clamp_values(result_dict)

    count_dict = _convert_to_numeric(count_dict)
    result_dict = _convert_to_numeric(result_dict)

    metadata = pd.read_csv(script_dir / "data" / "metadata.csv", index_col=0)
    metadata["Batch"] = metadata["Batch"].astype(str)

    # Construir matriz: muestras × citoquinas
    all_cytokines = list(result_dict.keys())
    samples = sorted(set().union(*(df.index for df in result_dict.values())))

    data_rows = []
    for sample in samples:
        row = {}
        for cyto in all_cytokines:
            if sample in result_dict[cyto].index:
                r1 = result_dict[cyto].loc[sample, "replicate_1"]
                r2 = result_dict[cyto].loc[sample, "replicate_2"]
                values = [v for v in [r1, r2] if not pd.isna(v)]
                row[cyto] = np.mean(values) if values else np.nan
            else:
                row[cyto] = np.nan
        data_rows.append(row)

    results = pd.DataFrame(data_rows, index=samples)

    # Alinear con metadata usando "Codigo_citoquinas"
    common_samples = metadata["Codigo_citoquinas"].isin(results.index)
    metadata_red = metadata[common_samples].copy()
    metadata_red = metadata_red.set_index("Codigo_citoquinas").loc[results.index].copy()

    # Filtrar y imputar
    results = results.loc[metadata_red.index].copy()  # asegurar mismo orden

    results = results.dropna(thresh=len(results) * 0.7, axis=1)
    print(f"→ {results.shape[1]} citoquinas después de filtrado NaNs")

    imputer = SimpleImputer(strategy="median")
    results_imputed = pd.DataFrame(
        imputer.fit_transform(results), index=results.index, columns=results.columns
    )

    # Log-transform (necesario antes de ComBat y PCA)
    results_log = np.log1p(results_imputed)

    # ====================== PCA ANTES de ComBat ======================
    print("\n=== PCA ANTES de ComBat ===")
    plot_pca(
        data=results_log,
        metadata=metadata_red,
        title="PCA - ANTES de ComBat\nTodos los estímulos",
        filename="pca_antes_combat.png",
    )

    # ====================== COMBAT ======================
    print("\n=== Aplicando ComBat ===")
    combat_t = pycombat_norm(
        counts=results_log.T,  # features (citoquinas) en filas
        batch=metadata_red["Batch"],
        covar_mod=metadata_red[
            ["Vacunado_Placebo", "Estimulacion"]
        ],  # protege el grupo biológico
        par_prior=True,
        # mean_only=False,                       # puedes probar True si solo quieres corregir media
        # prior_plots=False
    )

    results_combat = combat_t.T  # volver a muestras × citoquinas

    print(f"✓ ComBat completado → Shape: {results_combat.shape}")

    # ====================== PCA DESPUÉS de ComBat ======================
    print("\n=== PCA DESPUÉS de ComBat ===")
    plot_pca(
        data=results_combat,
        metadata=metadata_red,
        title="PCA - DESPUÉS de ComBat\nTodos los estímulos",
        filename="pca_despues_combat.png",
    )
